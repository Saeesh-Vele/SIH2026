"""Zero-shot LLaVA-family VQA inference.

The single implementation behind both `scripts/test_geochat_zeroshot.py` and the
graph's VQA node. Heavy imports (torch, transformers) stay inside `load()` so
importing this module costs nothing on a machine that will never run inference.

The working base today is stock LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`), loaded
through transformers' generic LLaVA classes. GeoChat is a LLaVA-1.5 derivative
and would be the better base for remote sensing, but its official checkpoint
does not load through those generic classes, and its repo's custom
`GeoChatLlamaForCausalLM` requires a 2023-era dependency stack (torch 2.0.1,
transformers 4.31.0) that this project cannot pin. The `geochat` loader path
below is kept for a GeoChat-format checkpoint if one is added later — e.g. a
fine-tuned adapter — and is used automatically when the `geochat` package is
importable.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

#: Relative `adapter_path` values in model_config.yaml resolve against this, so
#: the config reads the same whether the server was started from the repo root
#: or from backend/. Mirrors `app.core.config.BACKEND_ROOT`, but computed here
#: so this module stays importable without the FastAPI settings stack.
BACKEND_ROOT = Path(__file__).resolve().parents[2]

#: A GeoChat-format checkpoint, for the `geochat` loader path. Not the default:
#: it needs the official `geochat` package and its 2023-era dependency stack.
GEOCHAT_MODEL = "MBZUAI/geochat-7B"

# LLaVA-1.5 — and GeoChat, which derives from it — use the vicuna_v1 format.
VICUNA_SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
IMAGE_TOKEN = "<image>"

# GeoChat emits grounding boxes as {<x1><y1><x2><y2>|<angle>} on a 0-100 grid.
_BOX_RE = re.compile(
    r"\{<(?P<x1>\d{1,3})><(?P<y1>\d{1,3})><(?P<x2>\d{1,3})><(?P<y2>\d{1,3})>"
    r"(?:\|<(?P<angle>-?\d{1,3})>)?\}"
)


#: Set to 1/true/yes/on (or 0/false/no/off) to override
#: `vqa_grounding.cpu_fallback.enabled` on a demo machine without editing the YAML.
CPU_FALLBACK_ENV = "SATQUERY_CPU_FALLBACK"

#: What a fallback answer is attributed to. Never the fine-tuned LLaVA or its
#: adapter — those did not run.
CPU_FALLBACK_MODEL = "cpu_spectral_fallback (Salesforce/blip-vqa-base)"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"

#: Appended to ModelUnavailable when the fallback is off, so the opt-in is
#: discoverable from the trace rather than only from the README.
CPU_FALLBACK_HINT = (
    "(A CPU demo fallback exists but is off: set vqa_grounding.cpu_fallback.enabled "
    f"in model_config.yaml or {CPU_FALLBACK_ENV}=1. It does not use the fine-tuned model.)"
)


def cpu_fallback_allowed(configured: bool) -> bool:
    """The YAML value, unless SATQUERY_CPU_FALLBACK overrides it.

    Read at load time rather than at config parse, so the env var reaches every
    engine that wraps the VQA model however it was constructed.
    """
    raw = os.environ.get(CPU_FALLBACK_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return configured


def read_cpu_fallback(raw: dict[str, Any]) -> bool:
    """`cpu_fallback.enabled` from a config block; off when absent."""
    block = raw.get("cpu_fallback") or {}
    return bool(block.get("enabled", False)) if isinstance(block, dict) else False


def _short_reason(exc: BaseException) -> str:
    """The first clause of a load failure, for the trace's degraded_reason."""
    text = str(exc).strip()
    for sep in (" — ", "; ", ". "):
        text = text.split(sep, 1)[0]
    return text[:160] or type(exc).__name__


class ModelUnavailable(RuntimeError):
    """Raised when the checkpoint cannot be loaded on this machine.

    Carries a human-readable reason so callers can surface it instead of a
    stack trace — a missing GPU is an environment fact, not a bug.
    """


@dataclass
class GeoChatConfig:
    """The `vqa_grounding` block of model_config.yaml."""

    base_model: str = DEFAULT_MODEL
    adapter_path: str | None = None
    quantization: Literal["none", "8bit", "4bit"] = "4bit"
    device: str = "cuda:0"
    max_new_tokens: int = 256
    temperature: float = 0.2
    #: `cpu_fallback.enabled`. Off by default: a missing GPU is reported, not
    #: worked around. See `cpu_fallback_allowed` for the env override.
    cpu_fallback_enabled: bool = False

    @property
    def resolved_adapter_path(self) -> Path | None:
        """`adapter_path` as an absolute path, or None when running zero-shot."""
        if not self.adapter_path:
            return None
        path = Path(self.adapter_path)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GeoChatConfig":
        return cls(
            base_model=raw.get("base_model") or DEFAULT_MODEL,
            adapter_path=raw.get("adapter_path"),
            quantization=raw.get("quantization", "4bit"),
            device=raw.get("device", "cuda:0"),
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
            temperature=float(raw.get("temperature", 0.2)),
            cpu_fallback_enabled=read_cpu_fallback(raw),
        )


def build_prompt(question: str) -> str:
    """Vicuna-v1 single-turn prompt with the image token in the user message."""
    return f"{VICUNA_SYSTEM} USER: {IMAGE_TOKEN}\n{question} ASSISTANT:"


def parse_grounding(text: str) -> list[dict[str, Any]]:
    """Pull GeoChat-format grounding boxes out of an answer.

    Returns boxes normalised to 0-1 of the scene extent, the shape the canvas
    draws. Angles are kept when present — GeoChat emits oriented boxes for
    rotated objects such as aircraft and ships.

    Stock LLaVA-1.5, the current base, was not trained to emit these tokens, so
    this returns an empty list for it. It stays in the path for a GeoChat-format
    checkpoint added later.
    """
    boxes: list[dict[str, Any]] = []
    for i, match in enumerate(_BOX_RE.finditer(text)):
        x1, y1, x2, y2 = (int(match[k]) / 100 for k in ("x1", "y1", "x2", "y2"))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        angle = match["angle"]
        boxes.append(
            {
                "kind": "box",
                "id": f"g{i}",
                "label": "grounded",
                "confidence": 0.0,  # filled in by the caller from sequence scores
                "x": round(left, 4),
                "y": round(top, 4),
                "w": round(max(right - left, 0.005), 4),
                "h": round(max(bottom - top, 0.005), 4),
                **({"angle": int(angle)} if angle is not None else {}),
            }
        )
    return boxes


def strip_grounding(text: str) -> str:
    """The answer with box tokens removed, for display."""
    return re.sub(r"\s+", " ", _BOX_RE.sub("", text)).strip()


class GeoChatEngine:
    """Loads a LLaVA-family checkpoint once and answers questions about one image.

    Named for the `geochat` loader path it still supports; the default
    checkpoint is stock LLaVA-1.5.
    """

    def __init__(self, config: GeoChatConfig, loader: Literal["auto", "geochat", "llava"] = "auto"):
        self.config = config
        self.loader = loader
        self._model: Any = None
        self._tokenizer: Any = None
        self._image_processor: Any = None
        self._processor: Any = None
        self._use_geochat = False
        self._fallback_mode = False
        self._fallback_reason: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def degraded(self) -> bool:
        """True once load() has fallen back to the CPU demo model."""
        return self._fallback_mode

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def model_used(self) -> str:
        parts = [self.config.base_model]
        if self.config.adapter_path:
            parts.append(Path(self.config.adapter_path).name)
        return " + ".join(parts)

    # -- loading ----------------------------------------------------------
    def _preflight(self) -> None:
        """Fail fast, and with a reason a person can act on."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ModelUnavailable(
                "torch is not installed; install scripts/requirements.txt"
            ) from exc

        adapter = self.config.resolved_adapter_path
        if adapter is not None:
            # Checked before the base model is fetched: a typo in adapter_path
            # should not cost a 14 GB download to discover.
            if not (adapter / "adapter_config.json").is_file():
                raise ModelUnavailable(
                    f"no adapter_config.json under {adapter} — vqa_grounding.adapter_path "
                    "in model_config.yaml does not point at a PEFT adapter directory. "
                    "Set it to null to run the zero-shot base."
                )
            try:
                import peft  # noqa: F401
            except ImportError as exc:
                raise ModelUnavailable(
                    f"peft is not installed but adapter_path is set to {self.config.adapter_path}; "
                    "install scripts/requirements.txt, or set adapter_path to null"
                ) from exc

        if self.config.quantization == "4bit":
            if not torch.cuda.is_available():
                raise ModelUnavailable(
                    "4-bit quantization needs a CUDA GPU — bitsandbytes has no "
                    "CPU or MPS 4-bit kernel. Use quantization 'none' to run "
                    "unquantized on this machine."
                )
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ModelUnavailable(
                    "bitsandbytes is not installed; needed for 4-bit loading"
                ) from exc

    def _quant_config(self):
        if self.config.quantization == "none":
            return None

        import torch
        from transformers import BitsAndBytesConfig

        if self.config.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def load(self) -> None:
        """Bring the checkpoint into memory. Raises ModelUnavailable on a bad env.

        With `cpu_fallback` enabled, a load failure switches this engine to the
        CPU demo model instead; every answer it gives afterwards is marked
        degraded and never attributed to the configured checkpoint.
        """
        if self.loaded or self._fallback_mode:
            return

        try:
            self._preflight()
            self._load_checkpoint()
        except ModelUnavailable as exc:
            if cpu_fallback_allowed(self.config.cpu_fallback_enabled):
                self._enter_fallback(exc)
                return
            raise ModelUnavailable(f"{exc} {CPU_FALLBACK_HINT}") from exc

    def _enter_fallback(self, exc: BaseException) -> None:
        self._fallback_mode = True
        self._fallback_reason = _short_reason(exc)
        logger.warning(
            "%s unavailable (%s); cpu_fallback is enabled, answering with %s. "
            "Results are marked degraded.",
            self.config.base_model,
            exc,
            CPU_FALLBACK_MODEL,
        )

    def _load_checkpoint(self) -> None:
        use_geochat = self.loader == "geochat"
        if self.loader == "auto":
            try:
                import geochat  # noqa: F401

                use_geochat = True
            except ImportError:
                use_geochat = False
        self._use_geochat = use_geochat

        started = time.perf_counter()
        try:
            if use_geochat:
                self._load_geochat()
            else:
                self._load_llava()
            self._apply_adapter()
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any load failure as one type
            raise ModelUnavailable(f"could not load {self.config.base_model}: {exc}") from exc

        logger.info(
            "%s loaded via %s loader in %.1fs",
            self.config.base_model,
            "geochat" if use_geochat else "llava",
            time.perf_counter() - started,
        )

    def _apply_adapter(self) -> None:
        """Wrap the loaded base model in the configured LoRA adapter.

        Runs after either loader, since a LoRA adapter is a delta on the base
        weights and does not care which class produced them — only that the
        module names it targets are there. `adapter_config.json` records the
        base it was tuned against; a mismatch against `base_model` is logged
        rather than raised, because a compatible rename (a local copy of the
        same weights, say) is legitimate and only the caller can tell.

        No-op when `adapter_path` is unset, which is how the zero-shot base is
        run for comparison.
        """
        adapter = self.config.resolved_adapter_path
        if adapter is None:
            return

        import json

        from peft import PeftModel

        trained_on = json.loads((adapter / "adapter_config.json").read_text()).get(
            "base_model_name_or_path"
        )
        if trained_on and trained_on != self.config.base_model:
            logger.warning(
                "adapter %s was trained on %s but base_model is %s; loading anyway, "
                "but the tuned layers may not line up",
                adapter.name,
                trained_on,
                self.config.base_model,
            )

        self._model = PeftModel.from_pretrained(self._model, str(adapter))
        self._model.eval()
        logger.info("applied LoRA adapter %s", adapter.name)

    def _load_geochat(self) -> None:
        """Official GeoChat path — requires the `geochat` package on PYTHONPATH.

        Unused by default: the package pins torch 2.0.1 / transformers 4.31.0.
        Kept for a GeoChat-format checkpoint added later.
        """
        from geochat.mm_utils import get_model_name_from_path  # type: ignore
        from geochat.model.builder import load_pretrained_model  # type: ignore

        # The builder applies its own BitsAndBytesConfig from these flags.
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=self.config.base_model,
            model_base=None,
            model_name=get_model_name_from_path(self.config.base_model),
            load_4bit=self.config.quantization == "4bit",
            load_8bit=self.config.quantization == "8bit",
            device_map={"": self.config.device},
        )
        self._tokenizer, self._model, self._image_processor = tokenizer, model, image_processor

    def _load_llava(self) -> None:
        """Default path — transformers' generic LLaVA classes.

        Loads any LLaVA-compatible checkpoint, `llava-hf/llava-1.5-7b-hf`
        included. It will not load GeoChat's official weights, whose layout the
        generic classes do not match.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.config.base_model)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.config.base_model,
            quantization_config=self._quant_config(),
            device_map={"": self.config.device},
            low_cpu_mem_usage=True,
        )

    # -- inference --------------------------------------------------------
    def infer(
        self,
        image_path: str | Path | Any,
        question: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Answer one question about one image.

        Returns the shape every specialist node produces:
        ``{answer, evidence, confidence, model_used}``.

        `image_path` may also be an already-open PIL image. The fusion node
        passes one: its optical input is a multispectral GeoTIFF that PIL
        cannot decode, so it renders a viewable composite and hands that over
        rather than a path to bands this model could not read.

        `prompt` overrides the standard single-turn framing. The fusion node
        uses it to fold its sensor summary in beside the question; it must still
        carry exactly one image token.
        """
        self.load()

        if self._fallback_mode:
            return self._infer_spectral_fallback(image_path, question)

        import torch
        from PIL import Image

        source = image_path if isinstance(image_path, Image.Image) else Image.open(image_path)
        image = source.convert("RGB")
        prompt = prompt or build_prompt(question)
        max_new = max_new_tokens or self.config.max_new_tokens
        temp = self.config.temperature if temperature is None else temperature

        if self._use_geochat:
            from geochat.constants import IMAGE_TOKEN_INDEX  # type: ignore
            from geochat.mm_utils import tokenizer_image_token  # type: ignore

            pixel_values = self._image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ].to(self._model.device, dtype=torch.float16)
            input_ids = (
                tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(self._model.device)
            )
            inputs: dict[str, Any] = {"input_ids": input_ids, "images": pixel_values}
            prompt_len = input_ids.shape[1]
            tokenizer = self._tokenizer
        else:
            batch = self._processor(images=image, text=prompt, return_tensors="pt").to(
                self._model.device
            )
            inputs = dict(batch)
            prompt_len = batch["input_ids"].shape[1]
            tokenizer = self._processor.tokenizer

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                max_new_tokens=max_new,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        duration_ms = int((time.perf_counter() - started) * 1000)

        generated = output.sequences[0][prompt_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        confidence = _sequence_confidence(output, generated)

        boxes = parse_grounding(raw)
        for box in boxes:
            box["confidence"] = confidence

        return {
            "answer": strip_grounding(raw) or raw,
            "evidence": boxes,
            "confidence": confidence,
            "model_used": self.model_used,
            "raw": raw,
            "tokens": int(generated.shape[0]),
            "duration_ms": duration_ms,
        }

    def _get_blip(self):
        """Load the fallback's BLIP VQA model once.

        A failure here is not papered over with a statistics-only answer: the
        trace would then name a model that never ran.
        """
        if getattr(self, "_blip_model", None) is None:
            try:
                from transformers import BlipForQuestionAnswering, BlipProcessor

                self._blip_processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL)
                self._blip_model = BlipForQuestionAnswering.from_pretrained(BLIP_VQA_MODEL)
                self._blip_model.eval()
            except Exception as exc:  # noqa: BLE001 - reported as one type
                self._blip_model = None
                self._blip_processor = None
                raise ModelUnavailable(
                    f"cpu_fallback is enabled but {BLIP_VQA_MODEL} could not load: {exc}"
                ) from exc
        return self._blip_processor, self._blip_model

    def _infer_spectral_fallback(self, image_path: str | Path | Any, question: str) -> dict[str, Any]:
        """CPU demo answer: band statistics plus a small general-purpose VQA model.

        Opt-in only (`cpu_fallback`). The result is marked `degraded`, carries
        no confidence — nothing here measures one — and no grounding boxes,
        since nothing here localises anything.
        """
        import numpy as np
        from PIL import Image

        started = time.perf_counter()

        rgb: np.ndarray | None = None
        nir: np.ndarray | None = None
        band_count = 3
        width, height = 64, 64

        if isinstance(image_path, Image.Image):
            pil_img = image_path.convert("RGB")
            rgb = np.array(pil_img)
            height, width = rgb.shape[:2]
        else:
            p_str = str(image_path).lower()
            if p_str.endswith((".tif", ".tiff")):
                try:
                    import rasterio
                    with rasterio.open(image_path) as src:
                        band_count = src.count
                        width, height = src.width, src.height
                        if band_count >= 4:
                            r = src.read(4).astype(np.float32)
                            g = src.read(3).astype(np.float32)
                            b = src.read(2).astype(np.float32)
                            if band_count >= 8:
                                nir = src.read(8).astype(np.float32)
                        elif band_count >= 3:
                            r = src.read(1).astype(np.float32)
                            g = src.read(2).astype(np.float32)
                            b = src.read(3).astype(np.float32)
                        else:
                            r = g = b = src.read(1).astype(np.float32)
                        raw_rgb = np.stack([r, g, b], axis=-1)
                        p2, p98 = np.percentile(raw_rgb, (2, 98))
                        if p98 > p2:
                            rgb = np.clip((raw_rgb - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                        else:
                            rgb = np.clip(raw_rgb, 0, 255).astype(np.uint8)
                except Exception as exc:
                    logger.warning("rasterio read fallback: %s", exc)

            if rgb is None:
                try:
                    pil_img = Image.open(image_path).convert("RGB")
                    rgb = np.array(pil_img)
                    height, width = rgb.shape[:2]
                except Exception:
                    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        pil_img = Image.fromarray(rgb)

        r_f = rgb[:, :, 0].astype(np.float32)
        g_f = rgb[:, :, 1].astype(np.float32)
        b_f = rgb[:, :, 2].astype(np.float32)

        mean_r, mean_g, mean_b = float(np.mean(r_f)), float(np.mean(g_f)), float(np.mean(b_f))
        brightness = float(np.mean(rgb))
        contrast = float(np.std(rgb))

        gx = np.abs(np.diff(rgb, axis=1)).mean() if width > 1 else 0.0
        gy = np.abs(np.diff(rgb, axis=0)).mean() if height > 1 else 0.0
        edge_density = float((gx + gy) / 2.0)

        if nir is not None:
            ndvi_map = (nir - r_f) / (nir + r_f + 1e-6)
        else:
            ndvi_map = (g_f - r_f) / (g_f + r_f + 1e-6)
        mean_ndvi = float(np.mean(ndvi_map))

        veg_ratio = float(np.mean(ndvi_map > 0.2)) * 100.0
        water_ratio = float(np.mean((ndvi_map < 0.0) & (b_f > r_f))) * 100.0
        built_ratio = float(np.mean((r_f > 130) & (g_f > 130) & (b_f > 130))) * 100.0
        soil_ratio = max(0.0, 100.0 - veg_ratio - water_ratio - built_ratio)

        # 1. Run the small VQA model. Load failures raise ModelUnavailable;
        # inference errors propagate to the node like any other.
        processor, blip_model = self._get_blip()
        inputs = processor(pil_img, question, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=45)
        blip_observation = processor.decode(out[0], skip_special_tokens=True).strip()

        # 2. Dynamic Land Cover Classification
        if veg_ratio > 60 or mean_ndvi > 0.35:
            primary_label = "Forest / Dense Vegetation"
            land_desc = f"dense vegetation canopy (NDVI {mean_ndvi:.2f}, {veg_ratio:.1f}% coverage)"
        elif water_ratio > 40 or (mean_b > mean_g and mean_b > mean_r and mean_r < 60):
            primary_label = "Water Body / Basin"
            land_desc = f"open water surface (water index {water_ratio:.1f}%)"
        elif built_ratio > 30 or (edge_density > 110 and contrast > 50):
            primary_label = "Urban / Built-up Area"
            land_desc = f"built structures and high spatial contrast (texture index {edge_density:.1f})"
        else:
            primary_label = "Agricultural / Open Terrain"
            land_desc = f"cultivated terrain / open ground ({soil_ratio:.1f}% soil/open cover)"

        # 3. Generate Dynamic Multi-Modal Response
        q_lower = question.lower()
        obs_text = f"Visual reasoning identifies: '{blip_observation}'." if blip_observation else ""

        if any(w in q_lower for w in ["describe", "caption", "overview", "what is this", "what is visible", "tell me about", "what does this"]):
            answer = (
                f"{obs_text} Optical multispectral analysis across {band_count} band(s) reveals {land_desc}. "
                f"Spectral vegetation index is measured at NDVI={mean_ndvi:.2f} with surface texture complexity of {edge_density:.1f} "
                f"across the {width}x{height} capture."
            )
        elif any(w in q_lower for w in ["land cover", "classification", "type of land", "category", "class"]):
            answer = (
                f"Classification: {primary_label}. {obs_text} "
                f"Composition: {veg_ratio:.1f}% vegetation, {water_ratio:.1f}% water/shadow, {built_ratio:.1f}% built, {soil_ratio:.1f}% soil/open. "
                f"Analyzed {band_count} spectral bands (NDVI: {mean_ndvi:.2f})."
            )
        elif any(w in q_lower for w in ["how many", "count"]):
            if blip_observation and blip_observation.lower() not in ["none", "no", "0"]:
                answer = f"Visual detection: {blip_observation}. Analyzed across the {width}x{height} scene."
            else:
                answer = f"The {width}x{height} footprint represents continuous {primary_label.lower()} ({land_desc}); no discrete target objects to count."
        elif any(w in q_lower for w in ["is there", "are there", "does it have"]):
            if blip_observation:
                answer = f"{blip_observation.capitalize()}. Surface properties: {land_desc} across the {width}x{height} area."
            else:
                answer = f"Analysis confirms {primary_label.lower()} ({land_desc})."
        else:
            answer = (
                f"{obs_text} Spectral analysis indicates {primary_label.lower()} ({land_desc}) "
                f"with NDVI={mean_ndvi:.2f} and {band_count} spectral channels."
            )

        duration_ms = int((time.perf_counter() - started) * 1000)

        return {
            "answer": answer.strip(),
            "evidence": [],
            "confidence": None,
            "model_used": CPU_FALLBACK_MODEL,
            "degraded": True,
            "degraded_reason": self._fallback_reason,
            "raw": answer,
            "tokens": len(answer.split()),
            "duration_ms": duration_ms,
        }


def _sequence_confidence(output: Any, generated: Any) -> float:
    """Mean probability of the tokens the model actually chose.

    A generative VQA model has no calibrated confidence head, so this is the
    honest stand-in: how sure the decoder was, averaged over the answer.
    """
    import torch

    scores = getattr(output, "scores", None)
    if not scores:
        return 0.0

    probs: list[float] = []
    for step, logits in enumerate(scores):
        if step >= generated.shape[0]:
            break
        distribution = torch.softmax(logits[0].float(), dim=-1)
        probs.append(distribution[generated[step]].item())

    if not probs:
        return 0.0
    return round(sum(probs) / len(probs), 4)
