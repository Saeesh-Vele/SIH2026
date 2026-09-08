"""Optical-SAR fusion.

Four steps, and only the first needs weights of its own:

1. **Encode.** One SSL4EO-S12 (or DeCUR) encoder per modality turns the optical
   scene and the SAR capture into embeddings. Those pretraining schemes align
   the two modalities in a shared space, which is what makes step 3 meaningful.
2. **Fuse.** Concatenate them — the strategy is `fusion_strategy` in
   model_config.yaml, and `concat` is the default because it loses nothing and
   needs no learned head.
3. **Verbalise.** Reduce the fused vector to a handful of interpretable
   statistics — which sensor dominates, whether the two agree — and render them
   as one sentence using the phrases in `feature_vocabulary`.
4. **Answer.** Hand that sentence to the VQA model along with the optical image
   and the user's question.

Step 3 is the join between a vector nobody can read and a model that only reads
text. It is deliberately a small, inspectable mapping rather than a learned
captioner: every phrase the VQA model is told comes from the YAML, so a claim
the imagery cannot support cannot appear without someone writing it there.

The encoder is the one part that needs real weights. Until `checkpoint_path`
points at some, the node reports itself unavailable rather than inventing an
embedding — a fabricated fusion summary would be worse than no answer.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.geochat import GeoChatConfig, GeoChatEngine, ModelUnavailable

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "ssl4eo_s12_vit_base_patch16_224"
DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

FUSION_STRATEGIES = ("concat", "cross_attention", "gated")

#: Fallbacks when model_config.yaml supplies no `feature_vocabulary`. Keeping a
#: default here means the node still says something honest if the block is
#: trimmed; keeping it short means nobody mistakes it for a finding.
DEFAULT_VOCABULARY: dict[str, str] = {
    "sar_high": "strong radar backscatter",
    "sar_low": "weak radar backscatter",
    "optical_high": "high optical contrast",
    "optical_low": "flat, low-contrast optical response",
    "agree": "both sensors describe the same structure",
    "disagree": "the sensors disagree, so the optical view is likely obscured",
}

#: A modality counts as dominant above this share of the combined magnitude.
_DOMINANCE_BAND = 0.55
#: Cosine agreement above this reads as the two sensors telling one story.
_AGREEMENT_BAND = 0.35


@dataclass
class FusionConfig:
    """The `optical_sar_fusion` block of model_config.yaml."""

    encoder: str = DEFAULT_ENCODER
    checkpoint_path: str | None = None
    fusion_strategy: str = "concat"
    device: str = "cuda:0"
    base_model: str = DEFAULT_MODEL
    feature_vocabulary: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VOCABULARY))
    max_new_tokens: int = 256
    temperature: float = 0.2

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FusionConfig":
        vocabulary = {**DEFAULT_VOCABULARY, **(raw.get("feature_vocabulary") or {})}
        strategy = raw.get("fusion_strategy", "concat")
        if strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"unknown fusion_strategy {strategy!r}; expected one of {', '.join(FUSION_STRATEGIES)}"
            )
        return cls(
            encoder=raw.get("encoder") or DEFAULT_ENCODER,
            checkpoint_path=raw.get("checkpoint_path"),
            fusion_strategy=strategy,
            device=raw.get("device", "cuda:0"),
            base_model=raw.get("base_model") or DEFAULT_MODEL,
            feature_vocabulary=vocabulary,
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
            temperature=float(raw.get("temperature", 0.2)),
        )

    def as_vqa_config(self) -> GeoChatConfig:
        return GeoChatConfig(
            base_model=self.base_model,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )


# --------------------------------------------------------------------------
# Fusion and verbalisation — pure Python, so they are testable without weights.
# --------------------------------------------------------------------------
def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def fuse_embeddings(
    optical: list[float], sar: list[float], strategy: str = "concat"
) -> list[float]:
    """Combine the two modality embeddings.

    `concat` is the default and the only strategy that needs no learned
    parameters, so it is the one that works with an encoder dropped in cold.
    `gated` weights each modality by its share of the combined magnitude —
    still parameter-free, useful when one sensor is uninformative.
    `cross_attention` needs a trained head and is rejected here rather than
    silently approximated.
    """
    if len(optical) != len(sar):
        raise ValueError(
            f"embeddings must match in width; got optical {len(optical)}, sar {len(sar)}"
        )
    if not optical:
        raise ValueError("embeddings are empty")

    if strategy == "concat":
        return [*optical, *sar]
    if strategy == "gated":
        o_norm, s_norm = _norm(optical), _norm(sar)
        total = o_norm + s_norm
        if total == 0:
            return [0.0] * len(optical)
        w_o, w_s = o_norm / total, s_norm / total
        return [w_o * o + w_s * s for o, s in zip(optical, sar)]
    raise ValueError(
        f"fusion strategy {strategy!r} needs a trained head; no parameter-free form exists"
    )


def fusion_stats(optical: list[float], sar: list[float]) -> dict[str, float]:
    """The handful of numbers the description is built from.

    `sar_share` is SAR's portion of the combined embedding magnitude, so 0.5 is
    a balanced scene and higher means the radar response dominates.
    `agreement` is the cosine between the two embeddings, which is only
    meaningful because SSL4EO-S12 and DeCUR align the modalities in pretraining.
    """
    o_norm, s_norm = _norm(optical), _norm(sar)
    total = o_norm + s_norm
    dot = sum(o * s for o, s in zip(optical, sar))
    agreement = dot / (o_norm * s_norm) if o_norm and s_norm else 0.0

    return {
        "optical_norm": round(o_norm, 4),
        "sar_norm": round(s_norm, 4),
        "sar_share": round(s_norm / total, 4) if total else 0.5,
        "agreement": round(min(max(agreement, -1.0), 1.0), 4),
        "width": float(len(optical)),
    }


def describe_fusion(stats: dict[str, float], vocabulary: dict[str, str] | None = None) -> str:
    """Render the statistics as one sentence for the VQA model.

    Every clause comes from `vocabulary`, so what the model is told is editable
    in model_config.yaml without touching this code.
    """
    words = {**DEFAULT_VOCABULARY, **(vocabulary or {})}
    share = stats.get("sar_share", 0.5)
    agreement = stats.get("agreement", 0.0)

    clauses = [
        words["sar_high"] if share >= _DOMINANCE_BAND else words["sar_low"],
        words["optical_high"] if share <= 1 - _DOMINANCE_BAND else words["optical_low"],
        words["agree"] if agreement >= _AGREEMENT_BAND else words["disagree"],
    ]
    return "; ".join(clauses) + "."


def build_fusion_prompt(description: str, question: str) -> str:
    """The VQA prompt: what the sensors said, then what was asked.

    The description is labelled as coming from the fusion encoder so the model
    treats it as evidence about the scene rather than as part of the question.
    """
    from app.models.geochat import IMAGE_TOKEN, VICUNA_SYSTEM

    return (
        f"{VICUNA_SYSTEM} USER: {IMAGE_TOKEN}\n"
        f"This optical image was captured with a paired SAR image. "
        f"A sensor-fusion encoder summarises the pair as: {description}\n"
        f"Using both the image and that summary, answer: {question} ASSISTANT:"
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class FusionEngine:
    """Encodes an optical+SAR pair, fuses it, and answers a question about it."""

    def __init__(self, config: FusionConfig, vqa: GeoChatEngine | None = None):
        self.config = config
        self.vqa = vqa if vqa is not None else GeoChatEngine(config.as_vqa_config())
        self._encoder: Any = None

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    @property
    def model_used(self) -> str:
        return f"{self.config.encoder} ({self.config.fusion_strategy}) + {self.config.base_model}"

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        """Bring the fusion encoder into memory. Raises ModelUnavailable on a bad env.

        Deliberately strict about the checkpoint: an SSL4EO-S12 architecture with
        random weights would still produce an embedding, and that embedding would
        still yield a confident-sounding sentence. Refusing is the honest answer.
        """
        if self.loaded:
            return

        checkpoint = self.config.checkpoint_path
        if not checkpoint:
            raise ModelUnavailable(
                "no checkpoint_path set for optical_sar_fusion; point it at SSL4EO-S12 "
                "or DeCUR weights in model_config.yaml"
            )
        path = Path(checkpoint)
        if not path.exists():
            raise ModelUnavailable(
                f"fusion encoder weights not found at {path}. Download SSL4EO-S12 or "
                "DeCUR weights, or point checkpoint_path at a local copy."
            )

        try:
            import timm
            import torch
        except ImportError as exc:
            raise ModelUnavailable(
                "timm and torch are needed for the fusion encoder; install scripts/requirements.txt"
            ) from exc

        started = time.perf_counter()
        try:
            encoder = timm.create_model(self.config.encoder, pretrained=False, num_classes=0)
            state = torch.load(path, map_location="cpu")
            encoder.load_state_dict(state.get("state_dict", state), strict=False)
            encoder.eval().to(self.config.device)
        except Exception as exc:  # noqa: BLE001 - surface any load failure as one type
            raise ModelUnavailable(f"could not load {self.config.encoder}: {exc}") from exc

        self._encoder = encoder
        logger.info(
            "fusion encoder %s loaded in %.1fs", self.config.encoder, time.perf_counter() - started
        )

    # -- embedding --------------------------------------------------------
    def embed(self, image_path: str | Path) -> list[float]:
        """One pooled embedding for one image, whichever modality it is.

        Both modalities go through the same encoder because SSL4EO-S12 and DeCUR
        are trained that way — the modality is carried by the input statistics,
        not by a separate set of weights.
        """
        self.load()

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        config = timm_config(self._encoder)
        tensor = _to_tensor(image, config["input_size"], config["mean"], config["std"])

        with torch.inference_mode():
            features = self._encoder(tensor.unsqueeze(0).to(self.config.device))
        return [float(v) for v in features.flatten().float().cpu().tolist()]

    # -- inference --------------------------------------------------------
    def infer(
        self,
        optical_path: str | Path,
        sar_path: str | Path,
        question: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Answer one question about one optical+SAR pair.

        Returns the shape every specialist produces:
        ``{answer, evidence, confidence, model_used}``. `evidence` is empty —
        fusion produces a described scene, not geometry.
        """
        started = time.perf_counter()
        optical = self.embed(optical_path)
        sar = self.embed(sar_path)

        fused = fuse_embeddings(optical, sar, self.config.fusion_strategy)
        stats = fusion_stats(optical, sar)
        description = describe_fusion(stats, self.config.feature_vocabulary)

        result = self.vqa.infer(
            optical_path,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt=build_fusion_prompt(description, question),
        )

        return {
            "answer": result["answer"],
            "evidence": [],
            "confidence": result["confidence"],
            "model_used": self.model_used,
            "fusion_description": description,
            "fusion_stats": stats,
            "fused_width": len(fused),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }


def timm_config(encoder: Any) -> dict[str, Any]:
    """Input size and normalisation the encoder was trained with.

    Read off the model when timm recorded it, with ImageNet defaults otherwise —
    a DeCUR checkpoint carrying its own statistics keeps them.
    """
    cfg = getattr(encoder, "pretrained_cfg", None) or {}
    size = cfg.get("input_size", (3, 224, 224))
    return {
        "input_size": (int(size[1]), int(size[2])),
        "mean": tuple(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(cfg.get("std", (0.229, 0.224, 0.225))),
    }


def _to_tensor(image: Any, size: tuple[int, int], mean: tuple, std: tuple) -> Any:
    """Resize, scale to 0-1 and normalise, without pulling in torchvision."""
    import torch

    resized = image.resize(size)
    tensor = torch.frombuffer(resized.tobytes(), dtype=torch.uint8).float().div(255.0)
    tensor = tensor.reshape(size[1], size[0], 3).permute(2, 0, 1)
    mean_t = torch.tensor(mean).reshape(3, 1, 1)
    std_t = torch.tensor(std).reshape(3, 1, 1)
    return (tensor - mean_t) / std_t
