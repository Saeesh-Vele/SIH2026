"""Prompted-diff change detection.

Two halves, one checkpoint:

1. **Where.** Both frames go through the VQA model's own vision tower. The patch
   embeddings are compared cosine-wise, giving a coarse difference map on the
   encoder's patch grid (24x24 for LLaVA-1.5's CLIP ViT-L/14-336). Thresholding
   that map and grouping the surviving cells produces the `evidence` masks the
   canvas draws.
2. **What.** Both frames are then handed to the VQA model with a change-focused
   instruction, and its answer is the prose the user reads.

Using one checkpoint for both is the point: there is no second set of weights to
ship, and swapping `base_model` in model_config.yaml swaps the detector and the
describer together.

Caveat worth knowing: LLaVA-1.5 was trained on single images. It accepts two and
will answer, but its comparison is weaker than a model trained on image pairs.
The difference map does not depend on that — it comes from the vision tower,
which is applied to each frame separately.

Heavy imports (torch, transformers) stay inside `load()`, so importing this
module costs nothing on a machine that will never run inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.models.geochat import (
    VICUNA_SYSTEM,
    GeoChatConfig,
    GeoChatEngine,
    ModelUnavailable,
    read_cpu_fallback,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

#: Asked of the model once both frames are in the prompt. Deliberately names
#: both the what and the where — the difference map locates the change, and this
#: asks the model to say the same thing in words.
CHANGE_INSTRUCTION = (
    "Compare these two images of the same place, taken at different times. "
    "Describe what changed and where in the scene the change is."
)


@dataclass
class ChangeConfig:
    """The `change_detection` block of model_config.yaml."""

    method: str = "prompted_diff"
    base_model: str = DEFAULT_MODEL
    quantization: Literal["none", "8bit", "4bit"] = "4bit"
    device: str = "cuda:0"
    diff_threshold: float = 0.35
    min_region_area: float = 0.01
    max_regions: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.2
    #: `vqa_grounding.cpu_fallback.enabled` — one switch for every engine that
    #: wraps the VQA model. The registry copies it into this block.
    cpu_fallback_enabled: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ChangeConfig":
        return cls(
            method=raw.get("method", "prompted_diff"),
            base_model=raw.get("base_model") or DEFAULT_MODEL,
            quantization=raw.get("quantization", "4bit"),
            device=raw.get("device", "cuda:0"),
            diff_threshold=float(raw.get("diff_threshold", 0.35)),
            min_region_area=float(raw.get("min_region_area", 0.01)),
            max_regions=int(raw.get("max_regions", 8)),
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
            temperature=float(raw.get("temperature", 0.2)),
            cpu_fallback_enabled=read_cpu_fallback(raw),
        )

    def as_vqa_config(self) -> GeoChatConfig:
        """The same checkpoint, described the way GeoChatEngine expects."""
        return GeoChatConfig(
            base_model=self.base_model,
            quantization=self.quantization,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            cpu_fallback_enabled=self.cpu_fallback_enabled,
        )


def build_change_prompt(question: str | None = None) -> str:
    """Vicuna-v1 prompt carrying both frames.

    Two `<image>` tokens, earlier frame first. A user question is appended after
    the standard instruction so a change-VQA query ("did the pier get longer?")
    is answered directly rather than replaced by a generic summary.
    """
    ask = CHANGE_INSTRUCTION
    if question and question.strip():
        ask = f"{ask} Then answer: {question.strip()}"
    return f"{VICUNA_SYSTEM} USER: <image>\n<image>\n{ask} ASSISTANT:"


# --------------------------------------------------------------------------
# Difference map — pure Python, so it is testable without torch or a GPU.
# --------------------------------------------------------------------------
def threshold_map(scores: list[list[float]], threshold: float) -> list[list[bool]]:
    """Which cells of the difference map count as changed."""
    return [[value >= threshold for value in row] for row in scores]


def _components(mask: list[list[bool]]) -> list[list[tuple[int, int]]]:
    """Connected groups of changed cells, 4-connectivity, iterative flood fill."""
    if not mask:
        return []
    rows, cols = len(mask), len(mask[0])
    seen = [[False] * cols for _ in range(rows)]
    groups: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r][c] or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            group: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                group.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            groups.append(group)

    return groups


def mask_evidence(
    scores: list[list[float]],
    *,
    threshold: float = 0.35,
    min_region_area: float = 0.01,
    max_regions: int = 8,
) -> list[dict[str, Any]]:
    """Turn a difference map into the mask overlays the canvas draws.

    Each connected group of changed cells becomes one `mask` overlay whose
    polygon is the group's bounding rectangle in 0-1 scene space. A rectangle,
    not a traced contour: the map is a coarse patch grid — 24x24 for LLaVA-1.5 —
    so a pixel-accurate outline would imply precision the features do not carry.
    `confidence` is the mean difference score inside the group.

    Regions are returned strongest first, smaller than `min_region_area` of the
    scene dropped as noise, and at most `max_regions` of them.
    """
    if not scores or not scores[0]:
        return []

    rows, cols = len(scores), len(scores[0])
    cell_area = 1.0 / (rows * cols)
    regions: list[dict[str, Any]] = []

    for group in _components(threshold_map(scores, threshold)):
        if len(group) * cell_area < min_region_area:
            continue
        top = min(r for r, _ in group)
        bottom = max(r for r, _ in group) + 1
        left = min(c for _, c in group)
        right = max(c for _, c in group) + 1
        mean = sum(scores[r][c] for r, c in group) / len(group)

        x0, x1 = left / cols, right / cols
        y0, y1 = top / rows, bottom / rows
        regions.append(
            {
                "kind": "mask",
                "id": "",  # assigned below, after ranking
                "label": "changed region",
                "confidence": round(min(max(mean, 0.0), 1.0), 4),
                "polygon": [
                    [round(x0, 4), round(y0, 4)],
                    [round(x1, 4), round(y0, 4)],
                    [round(x1, 4), round(y1, 4)],
                    [round(x0, 4), round(y1, 4)],
                ],
                "area": round(len(group) * cell_area, 4),
            }
        )

    regions.sort(key=lambda r: (r["confidence"], r["area"]), reverse=True)
    for i, region in enumerate(regions[:max_regions]):
        region["id"] = f"c{i}"
    return regions[:max_regions]


def changed_fraction(scores: list[list[float]], threshold: float) -> float:
    """Share of the scene above the threshold, for the metrics strip."""
    cells = [value for row in scores for value in row]
    if not cells:
        return 0.0
    return round(sum(1 for value in cells if value >= threshold) / len(cells), 4)


#: Where a LLaVA-family checkpoint keeps its vision encoder, current layout
#: first. transformers 5.x moved the core submodules down a level —
#: `LlavaForConditionalGeneration.model` is a `LlavaModel`, and that is what
#: owns the tower — where 4.x hung it straight off the top-level class.
_VISION_TOWER_PATHS = ("model.vision_tower", "vision_tower")


def vision_tower(model: Any) -> Any:
    """The vision encoder inside a LLaVA-family model, wherever it now lives.

    Nothing in transformers promises this location, and it has already moved
    once: 5.16.1 answers to neither `model.vision_tower` nor the
    `get_vision_tower()` the LLaVA repo defines, both of which used to work.
    So the paths are tried in turn and a miss is reported as a
    `ModelUnavailable` naming the class that was searched — an AttributeError
    five frames into a diff says nothing about which layout arrived.
    """
    for path in _VISION_TOWER_PATHS:
        found: Any = model
        for attribute in path.split("."):
            found = getattr(found, attribute, None)
            if found is None:
                break
        if found is not None and callable(found):
            return found

    # The GeoChat fork and the original LLaVA repo expose a method instead.
    getter = getattr(model, "get_vision_tower", None)
    if callable(getter):
        found = getter()
        if found is not None:
            return found

    try:  # pragma: no cover - only for the message
        from transformers import __version__ as transformers_version
    except ImportError:  # pragma: no cover - unreachable once the model exists
        transformers_version = "unknown"
    raise ModelUnavailable(
        f"no vision tower found on {type(model).__name__} (transformers "
        f"{transformers_version}); looked at {', '.join(_VISION_TOWER_PATHS)} and "
        "get_vision_tower(). The prompted diff reads patch features straight off "
        "the encoder, so it needs the tower itself; if this checkpoint keeps it "
        "somewhere else, add that path to _VISION_TOWER_PATHS."
    )


def pixel_difference_map(t0_path: str | Path, t1_path: str | Path, grid: int = 24) -> list[list[float]]:
    """Per-cell RGB difference, normalised to 0-1. CPU fallback only.

    A stand-in for the vision-tower map when `cpu_fallback` is on and the
    checkpoint cannot load. It measures raw colour change — lighting, season
    and registration error all count — so it is never the default.
    """
    import numpy as np
    from PIL import Image

    a, b = (
        np.asarray(Image.open(p).convert("RGB").resize((grid, grid)), dtype=np.float32) / 255.0
        for p in (t0_path, t1_path)
    )
    diff = np.mean(np.abs(a - b), axis=2)
    low, high = float(diff.min()), float(diff.max())
    diff = (diff - low) / (high - low) if high > low else np.zeros_like(diff)
    return [[float(v) for v in row] for row in diff]


#: What a fallback change answer is attributed to. The prose is a template
#: filled from the pixel diff; no language model reads the frames.
CPU_FALLBACK_CHANGE_MODEL = "cpu_fallback (rgb pixel diff + templated answer)"


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class ChangeDetectorEngine:
    """Runs the prompted diff over one bi-temporal pair.

    Holds a `GeoChatEngine` rather than subclassing it: the VQA checkpoint is
    reused as-is, and everything change-specific lives here.
    """

    def __init__(self, config: ChangeConfig, vqa: GeoChatEngine | None = None):
        self.config = config
        self.vqa = vqa if vqa is not None else GeoChatEngine(config.as_vqa_config())

    @property
    def loaded(self) -> bool:
        return self.vqa.loaded

    @property
    def model_used(self) -> str:
        return f"{self.config.base_model} ({self.config.method})"

    def load(self) -> None:
        self.vqa.load()

    # -- difference map ---------------------------------------------------
    def difference_map(self, t0_path: str | Path, t1_path: str | Path) -> list[list[float]]:
        """Per-patch cosine distance between the two frames' visual features.

        Raises ModelUnavailable when the checkpoint cannot be loaded here, the
        same way the VQA path does — a missing GPU is an environment fact.
        """
        self.load()

        import torch
        from PIL import Image

        processor = self.vqa._processor
        model = self.vqa._model
        if processor is None:
            raise ModelUnavailable(
                "the prompted diff needs the transformers LLaVA path; the geochat "
                "loader exposes no separate vision tower here"
            )

        images = [Image.open(p).convert("RGB") for p in (t0_path, t1_path)]
        pixels = processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(model.device, dtype=next(model.parameters()).dtype)

        tower = vision_tower(model)
        with torch.inference_mode():
            features = tower(pixels, output_hidden_states=False)
            features = getattr(features, "last_hidden_state", features)

        # Drop the CLS token; what is left is one embedding per image patch.
        patches = features[:, 1:, :].float()
        similarity = torch.nn.functional.cosine_similarity(patches[0], patches[1], dim=-1)
        distance = ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)

        side = int(distance.shape[0] ** 0.5)
        if side * side != distance.shape[0]:  # pragma: no cover - non-square towers
            raise ModelUnavailable(
                f"vision tower returned {distance.shape[0]} patches, which is not a square grid"
            )
        grid = distance.reshape(side, side).tolist()
        return [[float(v) for v in row] for row in grid]

    # -- inference --------------------------------------------------------
    def infer(
        self,
        t0_path: str | Path,
        t1_path: str | Path,
        question: str | None = None,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Answer one change question about one bi-temporal pair.

        Returns the shape every specialist produces:
        ``{answer, evidence, confidence, model_used}``.
        """
        started = time.perf_counter()
        # Raises ModelUnavailable unless cpu_fallback is on, in which case the
        # VQA engine comes back degraded instead of loaded.
        self.load()
        if self.vqa.degraded:
            return self._infer_cpu_fallback(t0_path, t1_path, started)

        scores = self.difference_map(t0_path, t1_path)
        evidence = mask_evidence(
            scores,
            threshold=self.config.diff_threshold,
            min_region_area=self.config.min_region_area,
            max_regions=self.config.max_regions,
        )

        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in (t0_path, t1_path)]
        prompt = build_change_prompt(question)
        processor = self.vqa._processor
        model = self.vqa._model

        batch = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
        prompt_len = batch["input_ids"].shape[1]
        temp = self.config.temperature if temperature is None else temperature

        with torch.inference_mode():
            output = model.generate(
                **dict(batch),
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        from app.models.geochat import _sequence_confidence

        generated = output.sequences[0][prompt_len:]
        answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        confidence = _sequence_confidence(output, generated)

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": confidence,
            "model_used": self.model_used,
            "changed_fraction": changed_fraction(scores, self.config.diff_threshold),
            "grid": f"{len(scores)}x{len(scores[0]) if scores else 0}",
            "tokens": int(generated.shape[0]),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    def _infer_cpu_fallback(
        self, t0_path: str | Path, t1_path: str | Path, started: float
    ) -> dict[str, Any]:
        """Opt-in demo answer from a pixel diff. Marked degraded, no confidence."""
        scores = pixel_difference_map(t0_path, t1_path)
        evidence = mask_evidence(
            scores,
            threshold=self.config.diff_threshold,
            min_region_area=self.config.min_region_area,
            max_regions=self.config.max_regions,
        )
        frac = changed_fraction(scores, self.config.diff_threshold)
        pct = round(frac * 100, 1)
        n = len(evidence)
        if pct < 1.0:
            answer = (
                f"Minimal change detected between earlier (T0) and later (T1) captures "
                f"({pct}% of cells changed in raw colour)."
            )
        elif pct < 15.0:
            answer = (
                f"Localised colour change across {pct}% of the area "
                f"({n} changed region{'s' if n != 1 else ''})."
            )
        else:
            answer = (
                f"Widespread colour change across {pct}% of the scene "
                f"({n} changed region{'s' if n != 1 else ''})."
            )
        answer += " CPU fallback: raw pixel difference only, not the fine-tuned model."

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": None,
            "model_used": CPU_FALLBACK_CHANGE_MODEL,
            "degraded": True,
            "degraded_reason": self.vqa.fallback_reason,
            "changed_fraction": frac,
            "grid": f"{len(scores)}x{len(scores[0]) if scores else 0}",
            "tokens": len(answer.split()),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
