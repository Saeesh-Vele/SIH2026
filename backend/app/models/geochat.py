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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

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

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GeoChatConfig":
        return cls(
            base_model=raw.get("base_model") or DEFAULT_MODEL,
            adapter_path=raw.get("adapter_path"),
            quantization=raw.get("quantization", "4bit"),
            device=raw.get("device", "cuda:0"),
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
            temperature=float(raw.get("temperature", 0.2)),
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

    @property
    def loaded(self) -> bool:
        return self._model is not None

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
        """Bring the checkpoint into memory. Raises ModelUnavailable on a bad env."""
        if self.loaded:
            return

        self._preflight()

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
