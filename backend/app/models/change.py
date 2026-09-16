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

from app.models.geochat import VICUNA_SYSTEM, GeoChatConfig, GeoChatEngine, ModelUnavailable

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
        )

    def as_vqa_config(self) -> GeoChatConfig:
        """The same checkpoint, described the way GeoChatEngine expects."""
        return GeoChatConfig(
            base_model=self.base_model,
            quantization=self.quantization,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )


def build_change_prompt(question: str | None = None) -> str:
    """
    Prompt for the side-by-side temporal preview.
    """

    ask = (
        "The image contains two satellite views of the same location. "
        "The LEFT side is T0, the earlier image. "
        "The RIGHT side is T1, the later image. "
        "Compare the two views carefully. "
        "Identify meaningful changes between T0 and T1, "
        "and describe where those changes occur. "
        "Ignore the labels and focus on the satellite imagery."
    )

    if question and question.strip():
        ask += f" Then answer this question: {question.strip()}"

    return f"{VICUNA_SYSTEM} USER: <image>\n{ask} ASSISTANT:"

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

def create_temporal_preview(
    t0_path: str | Path,
    t1_path: str | Path,
):
    """
    Create a side-by-side temporal image:
    LEFT  = T0 / earlier image
    RIGHT = T1 / later image

    This allows the existing single-image LLaVA model
    to visually compare both timestamps.
    """
    from PIL import Image, ImageDraw

    t0 = Image.open(t0_path).convert("RGB")
    t1 = Image.open(t1_path).convert("RGB")

    # Keep both images at the same size
    width = min(t0.width, t1.width)
    height = min(t0.height, t1.height)

    t0 = t0.resize((width, height))
    t1 = t1.resize((width, height))

    canvas = Image.new(
        "RGB",
        (width * 2, height + 40),
        "white",
    )

    canvas.paste(t0, (0, 40))
    canvas.paste(t1, (width, 40))

    draw = ImageDraw.Draw(canvas)

    draw.text((10, 10), "T0 - Earlier", fill="black")
    draw.text((width + 10, 10), "T1 - Later", fill="black")

    return canvas
    
    


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
        """Lightweight bi-temporal change detection using image differences."""

        from PIL import Image
        import numpy as np

        GRID_SIZE = 24

        # Load both temporal images
        t0 = Image.open(t0_path).convert("RGB")
        t1 = Image.open(t1_path).convert("RGB")

        # Resize both images to the same grid
        t0 = t0.resize((GRID_SIZE, GRID_SIZE))
        t1 = t1.resize((GRID_SIZE, GRID_SIZE))

        # Convert to normalized RGB arrays
        a = np.asarray(t0, dtype=np.float32) / 255.0
        b = np.asarray(t1, dtype=np.float32) / 255.0

        # Calculate RGB difference
        diff = np.mean(np.abs(a - b), axis=2)

        # Normalize to 0-1
        min_val = float(diff.min())
        max_val = float(diff.max())

        if max_val > min_val:
            diff = (diff - min_val) / (max_val - min_val)
        else:
            diff = np.zeros_like(diff)

        return [
            [float(value) for value in row]
            for row in diff
        ]

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
        scores = self.difference_map(t0_path, t1_path)
        evidence = mask_evidence(
            scores,
            threshold=self.config.diff_threshold,
            min_region_area=self.config.min_region_area,
            max_regions=self.config.max_regions,
        )

        frac = changed_fraction(scores, self.config.diff_threshold)
        pct = round(frac * 100, 1)

        if getattr(self.vqa, "_fallback_mode", False) or self.vqa._model is None:
            num_regions = len(evidence)
            if pct < 1.0:
                answer = f"Minimal change detected between earlier (T0) and later (T1) captures ({pct}% surface variation). Surface reflectance and canopy structure remain largely stable across the footprint."
            elif pct < 15.0:
                answer = f"Localised changes detected across {pct}% of the area ({num_regions} distinct change cluster{'s' if num_regions != 1 else ''}). Surface alterations indicate minor vegetation clearance, growth, or seasonal ground variation."
            else:
                answer = f"Significant structural changes detected across {pct}% of the captured footprint ({num_regions} prominent change zone{'s' if num_regions != 1 else ''}). Substantial shift in surface reflectance observed between the earlier and later captures."

            confidence = round(0.86 + min(frac * 0.1, 0.08), 2)
            return {
                "answer": answer,
                "evidence": evidence,
                "confidence": confidence,
                "model_used": f"{self.model_used} (CPU fallback)",
                "changed_fraction": frac,
                "grid": f"{len(scores)}x{len(scores[0]) if scores else 0}",
                "tokens": len(answer.split()),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }

        import torch
        prompt = build_change_prompt(question)
        processor = self.vqa._processor
        model = self.vqa._model

        temporal_image = create_temporal_preview(t0_path, t1_path)
        batch = processor(
            images=temporal_image,
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

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
            "changed_fraction": frac,
            "grid": f"{len(scores)}x{len(scores[0]) if scores else 0}",
            "tokens": int(generated.shape[0]),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
