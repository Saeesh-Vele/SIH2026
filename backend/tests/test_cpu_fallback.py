#!/usr/bin/env python3
"""The opt-in CPU fallback: off by default, and unmistakable when on.

`vqa_grounding.cpu_fallback.enabled` (or SATQUERY_CPU_FALLBACK) lets a demo
machine without a GPU answer from a small CPU model instead of reporting the
task unavailable. These checks pin the contract around it:

- Off (the shipped default), a missing GPU is still ModelUnavailable — for
  VQA, change detection and fusion alike.
- On, every answer is marked `degraded`, carries confidence None, and is never
  attributed to LLaVA or the LoRA adapter.
- `degraded` and a null confidence always travel together; a real-model result
  without a float confidence is refused.

Same rules as the suites beside it: no GPU, no MongoDB, no network, no weights.
BLIP is stubbed, CUDA is forced absent, and every socket connect raises, so a
stray Hugging Face download fails the test instead of quietly succeeding.

    python backend/tests/test_cpu_fallback.py
    pytest backend/tests
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.geochat import (  # noqa: E402
    CPU_FALLBACK_ENV,
    CPU_FALLBACK_MODEL,
    GeoChatConfig,
    GeoChatEngine,
    ModelUnavailable,
)
from app.models.schemas import Intent, QueryStatus, TaskType, UploadMode  # noqa: E402

#: Captured before any test stubs it.
_REAL_GET_BLIP = GeoChatEngine._get_blip


# --------------------------------------------------------------------------
# Isolation: no network, no GPU, no real BLIP, no leaked env var
# --------------------------------------------------------------------------
class _NetworkBlocked(RuntimeError):
    pass


@contextlib.contextmanager
def _offline() -> Iterator[None]:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise _NetworkBlocked(f"network access attempted: {args!r}")

    saved = socket.socket.connect, socket.create_connection
    socket.socket.connect = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = saved  # type: ignore[assignment]


@contextlib.contextmanager
def _no_gpu() -> Iterator[None]:
    try:
        import torch
    except ImportError:  # preflight then reports torch missing, which is also no GPU
        yield
        return
    saved = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cuda.is_available = saved  # type: ignore[assignment]


@contextlib.contextmanager
def _env(value: str | None) -> Iterator[None]:
    saved = os.environ.get(CPU_FALLBACK_ENV)
    if value is None:
        os.environ.pop(CPU_FALLBACK_ENV, None)
    else:
        os.environ[CPU_FALLBACK_ENV] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(CPU_FALLBACK_ENV, None)
        else:
            os.environ[CPU_FALLBACK_ENV] = saved


class _FakeBlipProcessor:
    def __call__(self, image: Any, question: str, return_tensors: str = "pt") -> dict[str, Any]:
        return {"pixel_values": image, "question": question}

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        return "a field"


class _FakeBlipModel:
    def generate(self, **kwargs: Any) -> list[list[int]]:
        return [[0]]


@contextlib.contextmanager
def _fake_blip() -> Iterator[list[int]]:
    """Stand in for Salesforce/blip-vqa-base. Yields a call counter."""
    calls: list[int] = []
    saved = GeoChatEngine._get_blip

    def fake(self: GeoChatEngine) -> tuple[Any, Any]:
        calls.append(1)
        return _FakeBlipProcessor(), _FakeBlipModel()

    GeoChatEngine._get_blip = fake  # type: ignore[assignment]
    try:
        yield calls
    finally:
        GeoChatEngine._get_blip = saved  # type: ignore[assignment]


def isolated(fn):
    """Offline, GPU-less, fake BLIP, and SATQUERY_CPU_FALLBACK unset — per test."""

    @functools.wraps(fn)
    def wrapper() -> None:
        with _offline(), _no_gpu(), _fake_blip(), _env(None):
            fn()

    return wrapper


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _png(directory: Path, name: str, colour: tuple[int, int, int], patch: bool = False) -> str:
    from PIL import Image

    image = Image.new("RGB", (32, 32), colour)
    if patch:
        for x in range(8, 24):
            for y in range(8, 24):
                image.putpixel((x, y), (250, 250, 250))
    path = directory / name
    image.save(path)
    return str(path)


def _engine(enabled: bool) -> GeoChatEngine:
    # 4-bit on a GPU-less machine: the preflight refuses, the same way it does
    # for the shipped YAML.
    return GeoChatEngine(
        GeoChatConfig(quantization="4bit", cpu_fallback_enabled=enabled), loader="llava"
    )


def _change_engine(enabled: bool):
    from app.models.change import ChangeConfig, ChangeDetectorEngine

    return ChangeDetectorEngine(ChangeConfig(cpu_fallback_enabled=enabled))


def _run_node(task: str, node, state: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Run `node` with `engine` registered for `task`, then restore."""
    from app.core.model_registry import DEFAULT_LOADERS, get_registry

    registry = get_registry()
    registry.register_loader(task, lambda name, config: engine)
    try:
        return asyncio.run(node(state))
    finally:
        registry.register_loader(task, DEFAULT_LOADERS[task])


def _single_state(path: str) -> dict[str, Any]:
    return {
        "status": QueryStatus.OK,
        "intent": Intent.SINGLE_IMAGE_VQA,
        "query": "What land cover is this?",
        "assets": [{"role": "primary", "stored_path": path}],
        "parameters": {},
    }


def _pair_state(t0: str, t1: str) -> dict[str, Any]:
    return {
        "status": QueryStatus.OK,
        "intent": Intent.CHANGE_VQA,
        "task": TaskType.CHANGE_DETECTION,
        "query": "What changed?",
        "upload_mode": UploadMode.BI_TEMPORAL,
        "assets": [
            {"role": "t0", "stored_path": t0},
            {"role": "t1", "stored_path": t1},
        ],
        "parameters": {},
    }


def _assert_degraded(result: dict[str, Any]) -> None:
    """The pairing every fallback result must carry, and no LLaVA credit."""
    assert result["degraded"] is True, result
    assert result["confidence"] is None, result["confidence"]
    assert result.get("degraded_reason"), result
    model = result["model_used"].lower()
    assert "llava" not in model and "eurosat" not in model, result["model_used"]


# --------------------------------------------------------------------------
# Default: off
# --------------------------------------------------------------------------
@isolated
def test_the_shipped_yaml_keeps_the_fallback_off():
    from app.core.model_registry import ModelRegistry

    registry = ModelRegistry(BACKEND / "model_config.yaml")
    vqa = GeoChatConfig.from_mapping(registry.config_for("vqa_grounding"))
    assert vqa.cpu_fallback_enabled is False
    assert GeoChatConfig().cpu_fallback_enabled is False
    assert GeoChatConfig.from_mapping({}).cpu_fallback_enabled is False


@isolated
def test_flag_off_no_gpu_raises_model_unavailable_and_names_the_flag():
    engine = _engine(enabled=False)
    try:
        engine.load()
    except ModelUnavailable as exc:
        assert "CUDA" in str(exc) or "torch" in str(exc), exc
        assert "cpu_fallback" in str(exc) and CPU_FALLBACK_ENV in str(exc), exc
    else:
        raise AssertionError("expected ModelUnavailable with the fallback off")
    assert not engine.degraded


@isolated
def test_flag_off_vqa_node_reports_unavailable_not_an_answer():
    from app.agent.nodes.vqa import vqa_grounding_node

    with tempfile.TemporaryDirectory() as tmp:
        path = _png(Path(tmp), "scene.png", (40, 120, 40))
        out = _run_node("vqa_grounding", vqa_grounding_node, _single_state(path), _engine(False))

    assert out["status"] is QueryStatus.UNAVAILABLE, out
    assert "answer" not in out and "degraded" not in out, out


@isolated
def test_missing_peft_names_scripts_requirements():
    """The message is how someone finds the inference install step."""
    import builtins

    real_import = builtins.__import__

    def no_peft(name: str, *args: Any, **kwargs: Any):
        if name == "peft":
            raise ImportError("No module named 'peft'")
        return real_import(name, *args, **kwargs)

    engine = GeoChatEngine(
        GeoChatConfig(adapter_path="./checkpoints/eurosat_lora_v1_final", quantization="none"),
        loader="llava",
    )
    if engine.config.resolved_adapter_path is None or not (
        engine.config.resolved_adapter_path / "adapter_config.json"
    ).is_file():
        return  # adapter not checked out here; the missing-adapter path covers it

    builtins.__import__ = no_peft
    try:
        engine.load()
    except ModelUnavailable as exc:
        assert "scripts/requirements.txt" in str(exc), exc
    else:
        raise AssertionError("expected ModelUnavailable without peft")
    finally:
        builtins.__import__ = real_import


# --------------------------------------------------------------------------
# On: degraded and honest
# --------------------------------------------------------------------------
@isolated
def test_flag_on_no_gpu_answers_degraded_with_the_fallback_model():
    engine = _engine(enabled=True)
    with tempfile.TemporaryDirectory() as tmp:
        result = engine.infer(_png(Path(tmp), "scene.png", (40, 120, 40)), "Describe the scene.")

    assert engine.degraded
    _assert_degraded(result)
    assert result["model_used"] == "cpu_spectral_fallback (Salesforce/blip-vqa-base)"
    assert result["model_used"] == CPU_FALLBACK_MODEL
    assert "CUDA" in result["degraded_reason"], result["degraded_reason"]
    assert result["evidence"] == [], "the fallback localises nothing, so it draws nothing"
    assert result["answer"]


@isolated
def test_flag_on_vqa_trace_and_response_say_degraded():
    from app.agent.nodes.combiner import output_combiner
    from app.agent.nodes.vqa import vqa_grounding_node
    from app.models.schemas import ExecutionTraceCreate
    from app.routes.query import _to_response

    with tempfile.TemporaryDirectory() as tmp:
        state = _single_state(_png(Path(tmp), "scene.png", (40, 120, 40)))
        out = _run_node("vqa_grounding", vqa_grounding_node, state, _engine(True))

    assert "status" not in out, out  # an answer, not UNAVAILABLE
    _assert_degraded(out)
    assert out["steps"][0]["detail"].startswith("DEGRADED — CPU fallback"), out["steps"]

    merged = {**state, **out, "query_id": "q1", "trace_id": "t1"}
    merged.update(asyncio.run(output_combiner(merged)))
    assert merged["confidence"] is None and merged["degraded"] is True
    assert "DEGRADED" in merged["steps"][-1]["detail"], merged["steps"][-1]

    response = _to_response(merged)
    assert response.degraded is True and response.confidence is None
    assert response.degraded_reason
    assert response.models_used == [CPU_FALLBACK_MODEL], response.models_used

    trace = ExecutionTraceCreate(
        task_selected=TaskType.VQA_GROUNDING,
        models_used=response.models_used,
        confidence=merged["confidence"],
        degraded=merged["degraded"],
        degraded_reason=merged["degraded_reason"],
    )
    assert trace.model_dump(mode="json")["confidence"] is None


@isolated
def test_env_var_enables_the_fallback():
    with _env("1"):
        engine = _engine(enabled=False)
        engine.load()
    assert engine.degraded, "SATQUERY_CPU_FALLBACK=1 should enable the fallback"


@isolated
def test_env_var_zero_overrides_a_yaml_true():
    with _env("0"):
        try:
            _engine(enabled=True).load()
        except ModelUnavailable:
            pass
        else:
            raise AssertionError("SATQUERY_CPU_FALLBACK=0 should disable the fallback")


@isolated
def test_a_blip_load_failure_is_reported_not_papered_over():
    """No statistics-only answer credited to a model that never ran."""
    engine = _engine(enabled=True)

    def broken(self: GeoChatEngine):
        raise ModelUnavailable("cpu_fallback is enabled but Salesforce/blip-vqa-base could not load")

    saved = GeoChatEngine._get_blip
    GeoChatEngine._get_blip = broken  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            engine.infer(_png(Path(tmp), "scene.png", (40, 120, 40)), "What is here?")
    except ModelUnavailable as exc:
        assert "blip" in str(exc).lower(), exc
    else:
        raise AssertionError("expected ModelUnavailable when BLIP cannot load")
    finally:
        GeoChatEngine._get_blip = saved  # type: ignore[assignment]


@isolated
def test_the_real_blip_loader_fails_closed_when_the_download_fails():
    """The un-stubbed loader turns a failed fetch into ModelUnavailable."""
    try:
        from transformers import BlipProcessor
    except ImportError:  # pragma: no cover - depends on install
        return

    def refuse(*args: Any, **kwargs: Any):
        raise OSError("offline and not in the local cache")

    saved = BlipProcessor.from_pretrained
    BlipProcessor.from_pretrained = refuse  # type: ignore[assignment]
    try:
        _REAL_GET_BLIP(_engine(enabled=True))
    except ModelUnavailable as exc:
        assert "blip" in str(exc).lower(), exc
    else:
        raise AssertionError("expected ModelUnavailable when BLIP cannot be fetched")
    finally:
        BlipProcessor.from_pretrained = saved  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Change detection
# --------------------------------------------------------------------------
@isolated
def test_change_flag_off_no_gpu_raises_not_a_template_answer():
    engine = _change_engine(enabled=False)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = _png(Path(tmp), "t0.png", (40, 120, 40))
        t1 = _png(Path(tmp), "t1.png", (40, 120, 40), patch=True)
        try:
            engine.infer(t0, t1, "What changed?")
        except ModelUnavailable as exc:
            assert "cpu_fallback" in str(exc), exc
        else:
            raise AssertionError("expected ModelUnavailable, not a template answer")


@isolated
def test_change_node_flag_off_reports_unavailable():
    from app.agent.nodes.change import change_node

    with tempfile.TemporaryDirectory() as tmp:
        t0 = _png(Path(tmp), "t0.png", (40, 120, 40))
        t1 = _png(Path(tmp), "t1.png", (40, 120, 40), patch=True)
        out = _run_node("change_detection", change_node, _pair_state(t0, t1), _change_engine(False))

    assert out["status"] is QueryStatus.UNAVAILABLE, out
    assert "answer" not in out and "degraded" not in out, out


@isolated
def test_change_flag_on_is_degraded_and_does_not_credit_llava():
    engine = _change_engine(enabled=True)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = _png(Path(tmp), "t0.png", (40, 120, 40))
        t1 = _png(Path(tmp), "t1.png", (40, 120, 40), patch=True)
        result = engine.infer(t0, t1, "What changed?")

    _assert_degraded(result)
    assert result["evidence"], "the painted patch should show up as a changed region"
    assert 0.0 < result["changed_fraction"] < 1.0, result["changed_fraction"]


@isolated
def test_change_node_flag_on_marks_the_trace():
    from app.agent.nodes.change import change_node

    with tempfile.TemporaryDirectory() as tmp:
        t0 = _png(Path(tmp), "t0.png", (40, 120, 40))
        t1 = _png(Path(tmp), "t1.png", (40, 120, 40), patch=True)
        out = _run_node("change_detection", change_node, _pair_state(t0, t1), _change_engine(True))

    assert "status" not in out, out
    _assert_degraded(out)
    assert out["steps"][0]["detail"].startswith("DEGRADED — CPU fallback"), out["steps"]


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
class _DegradedVQA:
    """What a fallen-back GeoChatEngine returns, minus the image work."""

    def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "A field.",
            "evidence": [],
            "confidence": None,
            "model_used": CPU_FALLBACK_MODEL,
            "degraded": True,
            "degraded_reason": "4-bit quantization needs a CUDA GPU",
        }


@isolated
def test_fusion_passes_degraded_through_and_does_not_credit_llava():
    import numpy as np

    from app.models import fusion as fusion_module
    from app.models.fusion import FusionConfig, FusionEngine

    class _Engine(FusionEngine):
        def load(self) -> None:
            self._encoders = {"optical": object(), "sar": object()}

        def _embed_read(self, image, bands, modality):
            return [1.0, 0.0] if modality == "optical" else [0.6, 0.8]

    scenes = {
        "optical.tif": (np.zeros((13, 8, 8), dtype="uint16"), 13),
        "sar.tif": (np.zeros((2, 8, 8), dtype="uint16"), 2),
    }
    engine = _Engine(FusionConfig(), vqa=_DegradedVQA())

    original = fusion_module._read_bands
    fusion_module._read_bands = lambda path: scenes[Path(path).name]
    try:
        result = engine.infer("/scenes/optical.tif", "/scenes/sar.tif", "Any vessels?")
    finally:
        fusion_module._read_bands = original

    _assert_degraded(result)
    assert CPU_FALLBACK_MODEL in result["model_used"], result["model_used"]
    assert engine.config.base_model not in result["model_used"]


@isolated
def test_fusion_node_passes_degraded_through():
    from app.agent.nodes.fusion import fusion_node

    class _Stub:
        model_used = "stub-fusion"

        def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "answer": "A field.",
                "evidence": [],
                "confidence": None,
                "model_used": f"vit_base_patch16_224 (...) + {CPU_FALLBACK_MODEL}",
                "degraded": True,
                "degraded_reason": "4-bit quantization needs a CUDA GPU",
            }

    state = {
        "status": QueryStatus.OK,
        "intent": Intent.OPTICAL_SAR_FUSION,
        "task": TaskType.OPTICAL_SAR_FUSION,
        "query": "Any vessels?",
        "upload_mode": UploadMode.CROSS_MODAL,
        "assets": [
            {"role": "optical", "stored_path": "/tmp/opt.tif"},
            {"role": "sar", "stored_path": "/tmp/sar.tif"},
        ],
        "parameters": {},
    }
    out = _run_node("optical_sar_fusion", fusion_node, state, _Stub())
    _assert_degraded(out)
    assert out["steps"][0]["detail"].startswith("DEGRADED — CPU fallback"), out["steps"]


@isolated
def test_the_registry_shares_one_flag_with_change_and_fusion():
    import yaml

    from app.core.model_registry import ModelRegistry

    raw = yaml.safe_load((BACKEND / "model_config.yaml").read_text())
    raw["vqa_grounding"]["cpu_fallback"] = {"enabled": True}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model_config.yaml"
        path.write_text(yaml.safe_dump(raw))
        registry = ModelRegistry(path)
        from app.core.model_registry import DEFAULT_LOADERS

        for task, loader in DEFAULT_LOADERS.items():
            registry.register_loader(task, loader)
        change = registry.get_model("change_detection")
        fusion = registry.get_model("optical_sar_fusion")

    assert change.vqa.config.cpu_fallback_enabled is True
    assert fusion.vqa.config.cpu_fallback_enabled is True


# --------------------------------------------------------------------------
# Null confidence and degraded always travel together
# --------------------------------------------------------------------------
@isolated
def test_fallback_results_have_null_confidence_and_degraded_together():
    """VQA and change: both halves of the pairing, on the engines themselves."""
    with tempfile.TemporaryDirectory() as tmp:
        scene = _png(Path(tmp), "scene.png", (40, 120, 40))
        t1 = _png(Path(tmp), "t1.png", (40, 120, 40), patch=True)
        results = [
            _engine(True).infer(scene, "What is here?"),
            _change_engine(True).infer(scene, t1, "What changed?"),
        ]
    for result in results:
        assert result["confidence"] is None and result["degraded"] is True, result


def test_a_real_model_result_always_carries_a_float_confidence():
    from app.agent.state import honesty_fields

    assert honesty_fields({"confidence": 0.7}) == {
        "confidence": 0.7,
        "degraded": False,
        "degraded_reason": None,
    }
    assert isinstance(honesty_fields({"confidence": 1})["confidence"], float)

    for bad in ({"confidence": None}, {}, {"degraded": True, "confidence": 0.9}):
        try:
            honesty_fields(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"honesty_fields accepted {bad!r}")


@isolated
def test_a_node_refuses_a_real_result_with_no_confidence():
    """A missing number is a failed run, never a null that reads as 'fallback'."""
    from app.agent.nodes.vqa import vqa_grounding_node

    class _Broken:
        model_used = "llava-hf/llava-1.5-7b-hf"

        def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"answer": "x", "evidence": [], "confidence": None, "model_used": self.model_used}

    out = _run_node(
        "vqa_grounding", vqa_grounding_node, _single_state("/nonexistent.png"), _Broken()
    )
    assert out["status"] is QueryStatus.FAILED, out


def test_the_schema_refuses_null_confidence_without_degraded():
    from pydantic import ValidationError

    from app.models.schemas import ExecutionTraceCreate, QueryResponse

    base = {"query_id": "q", "trace_id": "t", "task_selected": TaskType.VQA_GROUNDING, "answer": "a"}
    for kwargs in ({"confidence": None}, {"confidence": 0.5, "degraded": True}):
        for model, extra in ((QueryResponse, base), (ExecutionTraceCreate, {"task_selected": TaskType.VQA_GROUNDING})):
            try:
                model(**extra, **kwargs)
            except ValidationError:
                pass
            else:
                raise AssertionError(f"{model.__name__} accepted {kwargs!r}")

    assert QueryResponse(**base, confidence=0.5).degraded is False
    assert QueryResponse(**base, confidence=None, degraded=True).confidence is None


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a runner reports, it does not raise
            failures.append(name)
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
