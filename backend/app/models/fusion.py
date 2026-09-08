"""Optical-SAR fusion.

Four steps, and only the first needs weights of its own:

1. **Encode.** One SSL4EO-S12 (or DeCUR) encoder per modality turns the optical
   scene and the SAR capture into embeddings. Those pretraining schemes align
   the two modalities in a shared space, which is what makes step 3 meaningful.
   Both publish plain ViT-Base/16 weights, so `encoder` names the timm
   architecture and `checkpoint_path` names the pretraining run that fills it.
2. **Fuse.** Concatenate them — the strategy is `fusion_strategy` in
   model_config.yaml, and `concat` is the default because it loses nothing and
   needs no learned head.
3. **Verbalise.** Reduce the fused vector to a handful of interpretable
   statistics — which sensor dominates, whether the two agree — and render them
   as one sentence using the phrases in `feature_vocabulary`.
4. **Answer.** Hand that sentence to the VQA model along with the optical image
   and the user's question. The VQA model sees a true-colour RGB rendering of
   that image rather than its bands: it reads pictures, not multispectral
   rasters, and `rgb_preview` is what makes one out of the other.

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

#: The architecture timm builds, not the pretraining run. SSL4EO-S12 and DeCUR
#: both publish ViT-Base/16 checkpoints that load into this; which weights are
#: used is `checkpoint_path`, not this name.
DEFAULT_ENCODER = "vit_base_patch16_224"
DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

FUSION_STRATEGIES = ("concat", "cross_attention", "gated")

#: The two halves, in the order they are concatenated. Optical first, so the
#: fused vector's first half is always the modality the VQA model also sees.
MODALITIES = ("optical", "sar")

#: Sentinel-2 true colour, 1-indexed as the mission numbers its bands: B4 red,
#: B3 green, B2 blue. Only meaningful for a raster in that band order.
TRUE_COLOR_BANDS = (4, 3, 2)

#: Band counts that carry the Sentinel-2 band order — L2A drops B10, L1C keeps
#: it. Any other count is not assumed to be colour-mappable.
SENTINEL2_BAND_COUNTS = (12, 13)

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

    #: One architecture for both modalities; the weights differ, not the shape.
    encoder: str = DEFAULT_ENCODER
    #: Sentinel-2 L1C is 13 bands, L2A is 12.
    optical_checkpoint_path: str | None = None
    optical_in_chans: int = 13
    #: Sentinel-1 is 2 bands, VV and VH.
    sar_checkpoint_path: str | None = None
    sar_in_chans: int = 2
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
        # The single-checkpoint schema cannot be honoured: whichever modality it
        # was trained on, the other one would be encoded by the wrong weights.
        # Failing here beats fusing a real embedding with a meaningless one.
        if "checkpoint_path" in raw:
            raise ValueError(
                "optical_sar_fusion.checkpoint_path is no longer read — the two "
                "modalities need separate encoders. Replace it with "
                "optical_checkpoint_path and sar_checkpoint_path."
            )

        return cls(
            encoder=raw.get("encoder") or DEFAULT_ENCODER,
            optical_checkpoint_path=raw.get("optical_checkpoint_path"),
            optical_in_chans=int(raw.get("optical_in_chans", 13)),
            sar_checkpoint_path=raw.get("sar_checkpoint_path"),
            sar_in_chans=int(raw.get("sar_in_chans", 2)),
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

    def checkpoint_for(self, modality: str) -> tuple[str | None, int]:
        """The weights and band count for one modality."""
        if modality == "optical":
            return self.optical_checkpoint_path, self.optical_in_chans
        if modality == "sar":
            return self.sar_checkpoint_path, self.sar_in_chans
        raise ValueError(f"unknown modality {modality!r}; expected 'optical' or 'sar'")


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
    """Encodes an optical+SAR pair, fuses it, and answers a question about it.

    Two encoders, not one. SSL4EO-S12 pretrains per modality and the results are
    not interchangeable: the patch embedding is built for a specific band count,
    so a Sentinel-1 encoder cannot read Sentinel-2 imagery at all. Both are
    ViT-B/16 and pool to the same width, which is what makes the halves
    concatenable — and `load` checks that rather than trusting it.
    """

    def __init__(self, config: FusionConfig, vqa: GeoChatEngine | None = None):
        self.config = config
        self.vqa = vqa if vqa is not None else GeoChatEngine(config.as_vqa_config())
        self._encoders: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return set(self._encoders) >= set(MODALITIES)

    @property
    def model_used(self) -> str:
        return (
            f"{self.config.encoder} (optical {self.config.optical_in_chans}b + "
            f"SAR {self.config.sar_in_chans}b, {self.config.fusion_strategy}) + "
            f"{self.config.base_model}"
        )

    # -- loading ----------------------------------------------------------
    def _load_one(self, modality: str) -> Any:
        """Build the architecture for one modality and fill it from its checkpoint.

        Deliberately strict: a ViT-B/16 with random weights would still produce
        an embedding, and that embedding would still yield a confident-sounding
        sentence. Refusing is the honest answer.
        """
        import timm
        import torch

        checkpoint, in_chans = self.config.checkpoint_for(modality)
        if not checkpoint:
            raise ModelUnavailable(
                f"no {modality}_checkpoint_path set for optical_sar_fusion; point it at "
                "SSL4EO-S12 or DeCUR weights in model_config.yaml"
            )
        path = Path(checkpoint)
        if not path.exists():
            raise ModelUnavailable(
                f"{modality} encoder weights not found at {path}. Download the SSL4EO-S12 "
                f"{modality} checkpoint, or point {modality}_checkpoint_path at a local copy."
            )

        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(raw)

        # The band count is the thing a remote-sensing checkpoint is most likely
        # to differ on, and a mismatch silently leaves the patch embedding
        # random. Check it before building rather than after.
        bands = _checkpoint_in_chans(state)
        if bands is not None and bands != in_chans:
            raise ModelUnavailable(
                f"{path.name} was trained on {bands} input bands but {modality}_in_chans "
                f"is {in_chans}. Set {modality}_in_chans: {bands} in model_config.yaml, "
                f"or point {modality}_checkpoint_path at weights for this many bands."
            )

        encoder = timm.create_model(
            self.config.encoder, pretrained=False, num_classes=0, in_chans=in_chans
        )
        # MAE checkpoints carry a decoder the encoder-only model has no slot for;
        # strict=False drops it. `tests/verify_fusion_checkpoint.py` reports
        # exactly what each checkpoint left behind.
        encoder.load_state_dict(state, strict=False)
        return encoder.eval().to(self.config.device)

    def load(self) -> None:
        """Bring both encoders into memory. Raises ModelUnavailable on a bad env."""
        if self.loaded:
            return

        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise ModelUnavailable(
                "timm and torch are needed for the fusion encoders; install "
                "scripts/requirements.txt"
            ) from exc

        started = time.perf_counter()
        encoders: dict[str, Any] = {}
        for modality in MODALITIES:
            try:
                encoders[modality] = self._load_one(modality)
            except ModelUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - one failure type out of here
                raise ModelUnavailable(
                    f"could not load the {modality} encoder ({self.config.encoder}): {exc}"
                ) from exc

        # Concatenating two embeddings only means anything if they are the same
        # width, and cosine agreement needs them elementwise comparable. Both
        # hold for two ViT-B/16 towers, but a swapped checkpoint could break it
        # silently, so it is checked once here instead of assumed at every call.
        widths = {m: int(getattr(e, "num_features", 0)) for m, e in encoders.items()}
        if len(set(widths.values())) != 1 or not all(widths.values()):
            raise ModelUnavailable(
                f"the two encoders pool to different widths ({widths}); fusion needs "
                "one shared embedding space. Use checkpoints of the same architecture."
            )

        self._encoders = encoders
        self.embedding_width = next(iter(widths.values()))
        logger.info(
            "fusion encoders loaded in %.1fs: %s",
            time.perf_counter() - started,
            ", ".join(f"{m} {self.config.checkpoint_for(m)[1]}b -> {w}d" for m, w in widths.items()),
        )

    # -- embedding --------------------------------------------------------
    def embed(self, image_path: str | Path, modality: str) -> list[float]:
        """One pooled embedding for one image, through that modality's encoder.

        The band count has to match what the encoder was pretrained on, so an
        image carrying the wrong number is refused rather than padded or
        truncated into shape — either would put made-up values in front of a
        model whose output is then described in words.
        """
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
        self.load()
        image, bands = _read_bands(image_path)
        return self._embed_read(image, bands, modality)

    def _embed_read(self, image: Any, bands: int, modality: str) -> list[float]:
        """`embed` for an image that has already been read off disk.

        `infer` needs the optical bands twice — once for the encoder, once to
        render the preview the VQA model looks at — and reading a multispectral
        scene twice to get them is pure waste.
        """
        import torch

        encoder = self._encoders[modality]
        _, expected = self.config.checkpoint_for(modality)
        if bands != expected:
            raise ModelUnavailable(
                f"the {modality} image has {bands} band(s) but its encoder was pretrained "
                f"on {expected}. Bind a {expected}-band capture "
                f"({'Sentinel-2 multispectral' if modality == 'optical' else 'Sentinel-1 VV+VH'}), "
                f"or point {modality}_checkpoint_path at weights for {bands} bands."
            )

        config = timm_config(encoder)
        tensor = _to_tensor(image, bands, config["input_size"], config["mean"], config["std"])
        with torch.inference_mode():
            features = encoder(tensor.unsqueeze(0).to(self.config.device))
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

        The encoder and the VQA model see different renderings of the same
        optical capture: all of its bands, and the true-colour preview
        `rgb_preview` derives from them.
        """
        started = time.perf_counter()
        self.load()
        # Read once: the encoder needs every band, the VQA model needs a picture.
        optical_image, optical_bands = _read_bands(optical_path)
        optical = self._embed_read(optical_image, optical_bands, "optical")
        sar = self.embed(sar_path, "sar")

        fused = fuse_embeddings(optical, sar, self.config.fusion_strategy)
        stats = fusion_stats(optical, sar)
        description = describe_fusion(stats, self.config.feature_vocabulary)

        # Not `optical_path`: a 13-band GeoTIFF is not an image PIL can open,
        # and LLaVA has no way to look at raw multispectral data anyway.
        result = self.vqa.infer(
            rgb_preview(optical_image, optical_bands),
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


def _unwrap_state_dict(raw: Any) -> dict:
    """Pull the weights out of a checkpoint. Publishers nest them differently."""
    if not isinstance(raw, dict):
        return raw
    for key in ("state_dict", "model", "model_state_dict"):
        inner = raw.get(key)
        if isinstance(inner, dict):
            return inner
    return raw


def _checkpoint_in_chans(state: dict) -> int | None:
    """Input bands the checkpoint was trained on, from its patch embedding."""
    weight = state.get("patch_embed.proj.weight")
    return int(weight.shape[1]) if weight is not None and getattr(weight, "ndim", 0) == 4 else None


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


def _read_bands(image_path: str | Path) -> tuple[Any, int]:
    """Open an image and report how many bands it actually carries.

    Tries a real multispectral reader first. Sentinel-2 imagery is 12-13 bands
    and Sentinel-1 is 2, and PIL has no mode for either — it can only describe
    1, 3 or 4 band rasters — so without rasterio or tifffile a multispectral
    GeoTIFF cannot be read at all. Saying that plainly is better than handing
    back a silently truncated three-band version of a thirteen-band scene.
    """
    path = Path(image_path)

    try:  # pragma: no cover - depends on install
        import rasterio

        with rasterio.open(path) as src:
            return src.read(), int(src.count)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - not a raster rasterio understands
        logger.debug("rasterio could not read %s: %s", path, exc)

    try:  # pragma: no cover - depends on install
        import tifffile

        array = tifffile.imread(path)
        bands = int(array.shape[0] if array.ndim == 3 and array.shape[0] <= 32 else 1)
        return array, bands
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - not a TIFF tifffile understands
        logger.debug("tifffile could not read %s: %s", path, exc)

    from PIL import Image

    image = Image.open(path)
    bands = len(image.getbands())
    if path.suffix.lower() in {".tif", ".tiff", ".gtiff"} and bands <= 4:
        raise ModelUnavailable(
            f"{path.name} was read through PIL, which reports {bands} band(s) and cannot "
            "describe more. Install rasterio to read multispectral GeoTIFFs; without it "
            "a Sentinel-2 scene cannot be encoded."
        )
    return image, bands


def rgb_preview(image: Any, bands: int) -> Any:
    """Render something a VQA model can actually look at.

    The encoder wants all 13 bands; LLaVA-family models want a picture. They
    only ever see three channels of 8-bit colour, and PIL — which is what opens
    the image on the VQA side — cannot even decode more than four bands, so a
    Sentinel-2 scene has to be reduced before the question is asked.

    Sentinel-2 becomes its true-colour composite: B4/B3/B2, the mission's own
    1-indexed red, green and blue. Anything without a known colour mapping —
    Sentinel-1's two polarisations, a single-band raster — becomes its first
    band in grey, which is honest about showing one channel rather than
    inventing a palette for bands that have no colour.

    A PIL image is already viewable, so it is handed back untouched: a normal
    3-band PNG is the same object for both the encoder and the VQA model.
    """
    from PIL import Image
    from PIL.Image import Image as PILImage

    if isinstance(image, PILImage):
        return image if image.mode == "RGB" else image.convert("RGB")

    import numpy as np

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a (bands, h, w) array for the preview, got {array.shape}")

    if bands in SENTINEL2_BAND_COUNTS and array.shape[0] >= max(TRUE_COLOR_BANDS):
        channels = [array[i - 1] for i in TRUE_COLOR_BANDS]
    elif array.shape[0] >= 3:
        # Already in view order — an RGB or RGBA raster, any extra band dropped.
        channels = [array[i] for i in range(3)]
    else:
        channels = [array[0]] * 3

    return Image.fromarray(np.stack([_stretch(c) for c in channels], axis=-1), mode="RGB")


def _stretch(band: Any) -> Any:
    """One band scaled to 0-255 over its 2nd-98th percentile.

    Reflectance occupies a narrow slice of a 16-bit range, and a plain min-max
    would let one bright pixel push the rest of the scene to black. Percentiles
    are what every true-colour renderer uses for the same reason; the min-max
    fallback covers a band flat enough that the two percentiles coincide.
    """
    import numpy as np

    values = np.asarray(band, dtype="float32")
    low, high = (float(v) for v in np.percentile(values, (2, 98)))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype="uint8")
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype("uint8")


def _to_tensor(image: Any, bands: int, size: tuple[int, int], mean: tuple, std: tuple) -> Any:
    """Resize, scale to 0-1 and normalise, without pulling in torchvision.

    Takes either a PIL image or a (bands, h, w) array, so the multispectral
    readers and the PIL fallback both land here.

    Normalisation note: `mean`/`std` come from the architecture's ImageNet
    defaults, which describe RGB and nothing else. For any other band count this
    falls back to a plain 0.5/0.5, which centres the input without pretending to
    know the sensor's statistics. SSL4EO-S12 publishes per-band values, and they
    belong in model_config.yaml before these embeddings are trusted.
    """
    import torch

    from PIL.Image import Image as PILImage

    # Not a duck-typed check: numpy arrays also carry .resize and .tobytes, and
    # they mean something completely different there.
    if isinstance(image, PILImage):
        resized = image.resize(size)
        tensor = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        tensor = tensor.float().div(255.0).reshape(size[1], size[0], bands).permute(2, 0, 1)
    else:  # (bands, h, w) from rasterio or tifffile
        tensor = torch.as_tensor(image).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        peak = float(tensor.max()) or 1.0
        tensor = tensor.div(peak)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(size[1], size[0]), mode="bilinear", align_corners=False
        ).squeeze(0)

    if len(mean) != bands or len(std) != bands:
        mean, std = (0.5,) * bands, (0.5,) * bands
    mean_t = torch.tensor(mean).reshape(bands, 1, 1)
    std_t = torch.tensor(std).reshape(bands, 1, 1)
    return (tensor - mean_t) / std_t
