#!/usr/bin/env python3
"""CPU-only checks for the controller: graph paths, upload validation, routing.

Everything here runs without a GPU, a MongoDB or an OpenRouter key — no weights
are loaded and no network call is made. Run it either way:

    python backend/tests/test_cpu_smoke.py     # no dependencies beyond the app's
    pytest backend/tests                       # if pytest is installed
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.schemas import (  # noqa: E402
    INTENT_TO_MODES,
    INTENT_TO_TASK,
    Intent,
    QueryStatus,
    TaskType,
    UploadMode,
)


# --------------------------------------------------------------------------
# model registry / checkpoint
# --------------------------------------------------------------------------
def test_default_model_is_llava():
    from app.models.geochat import DEFAULT_MODEL, GEOCHAT_MODEL

    assert DEFAULT_MODEL == "llava-hf/llava-1.5-7b-hf", DEFAULT_MODEL
    assert GEOCHAT_MODEL == "MBZUAI/geochat-7B"


def test_yaml_matches_default():
    import yaml

    cfg = yaml.safe_load((BACKEND / "model_config.yaml").read_text())
    assert cfg["vqa_grounding"]["base_model"] == "llava-hf/llava-1.5-7b-hf"
    assert set(cfg) == {t.value for t in TaskType}

    # The fine-tuned adapter ships in the repo, so the configured path must
    # actually be there — a dangling adapter_path only surfaces on the GPU box.
    # `python backend/tests/verify_lora_adapter.py` checks its contents.
    adapter = BACKEND / cfg["vqa_grounding"]["adapter_path"]
    assert (adapter / "adapter_config.json").is_file(), adapter


def test_registry_resolves_without_loading_weights():
    """Every task now has a real loader; see test_specialists for the other two."""
    from app.core.model_registry import get_registry
    from app.models.geochat import GeoChatEngine

    registry = get_registry()
    assert registry.tasks == sorted(t.value for t in TaskType), registry.tasks

    engine = registry.get_model("vqa_grounding")
    assert isinstance(engine, GeoChatEngine)
    assert engine.loaded is False, "constructing the engine must not touch weights"
    # The adapter is named in model_used because that string is what the
    # execution trace records as having answered the question.
    assert engine.model_used == (
        "llava-hf/llava-1.5-7b-hf + eurosat_lora_v1_final"
    ), engine.model_used


def test_adapter_path_resolves_against_the_backend_root():
    """Not the working directory — uvicorn gets started from either one."""
    from app.models.geochat import BACKEND_ROOT, GeoChatConfig

    relative = GeoChatConfig.from_mapping({"adapter_path": "./checkpoints/some-lora"})
    assert relative.resolved_adapter_path == BACKEND_ROOT / "checkpoints" / "some-lora"

    absolute = GeoChatConfig.from_mapping({"adapter_path": "/opt/loras/some-lora"})
    assert absolute.resolved_adapter_path == Path("/opt/loras/some-lora")

    assert GeoChatConfig.from_mapping({}).resolved_adapter_path is None


def test_a_missing_adapter_is_reported_before_the_base_model_is_fetched():
    """A typo in adapter_path should not cost a 14 GB download to discover."""
    from app.models.geochat import GeoChatConfig, GeoChatEngine, ModelUnavailable

    engine = GeoChatEngine(
        GeoChatConfig(adapter_path="./checkpoints/does-not-exist", quantization="none"),
        loader="llava",
    )
    try:
        engine.load()
    except ModelUnavailable as exc:
        assert "adapter_config.json" in str(exc), exc
    else:
        raise AssertionError("expected ModelUnavailable for a missing adapter")


def test_config_falls_back_to_default_model():
    from app.models.geochat import DEFAULT_MODEL, GeoChatConfig

    assert GeoChatConfig.from_mapping({}).base_model == DEFAULT_MODEL
    assert GeoChatConfig.from_mapping({"base_model": None}).base_model == DEFAULT_MODEL


def test_missing_gpu_is_reported_not_raised():
    from app.models.geochat import GeoChatConfig, GeoChatEngine, ModelUnavailable

    engine = GeoChatEngine(GeoChatConfig(base_model="llava-hf/llava-1.5-7b-hf"), loader="llava")
    try:
        engine.load()
    except ModelUnavailable as exc:
        assert "CUDA" in str(exc) or "torch" in str(exc), exc
    else:
        raise AssertionError("expected ModelUnavailable on a machine with no CUDA GPU")


def test_grounding_parser_still_reads_box_tokens():
    """Kept for a grounding-trained checkpoint; LLaVA-1.5 never emits these."""
    from app.models.geochat import build_prompt, parse_grounding, strip_grounding

    boxes = parse_grounding("Two planes {<10><20><30><40>|<45>} and {<50><60><70><80>}")
    assert len(boxes) == 2 and boxes[0]["angle"] == 45
    assert boxes[0]["x"] == 0.1 and round(boxes[0]["w"], 3) == 0.2
    assert strip_grounding("A jet {<10><20><30><40>} here") == "A jet here"
    assert parse_grounding("plain llava prose, no tokens") == []
    assert "<image>" in build_prompt("q")


# --------------------------------------------------------------------------
# upload validation
# --------------------------------------------------------------------------
def test_upload_extension_rules():
    from fastapi import HTTPException

    from app.routes.upload import MODE_ROLES, _validate_extension

    _validate_extension("scene.tif", False)
    _validate_extension("scene.TIFF", False)
    _validate_extension("scene.png", True)

    for filename in ("scene.png", "scene.jpg"):
        try:
            _validate_extension(filename, False)
        except HTTPException as exc:
            assert exc.status_code == 415
        else:
            raise AssertionError(f"{filename} must be rejected outside benchmark mode")

    try:
        _validate_extension("scene.pdf", True)
    except HTTPException as exc:
        assert exc.status_code == 415
    else:
        raise AssertionError(".pdf must be rejected in every mode")

    assert MODE_ROLES[UploadMode.CROSS_MODAL] == ("optical", "sar")
    assert len(MODE_ROLES) == len(list(UploadMode))


# --------------------------------------------------------------------------
# intent classification
# --------------------------------------------------------------------------
def test_intent_taxonomy_is_captioning_not_grounding():
    labels = {i.value for i in Intent}
    assert "single_image_captioning" in labels
    assert "single_image_grounding" not in labels
    assert set(INTENT_TO_TASK) == set(Intent)
    assert set(INTENT_TO_MODES) == set(Intent)


def test_captioning_queries_classify_as_captioning():
    from app.agent.llm import classify_heuristic

    captioning = [
        "Describe this image.",
        "Describe what this scene contains.",
        "What's visible in this image?",
        "What is visible here?",
        "Give me a summary of this scene.",
        "Write a caption for this image.",
        "Tell me about this area.",
        "What can you see in this capture?",
        "Provide an overview of the site.",
    ]
    for query in captioning:
        decision = classify_heuristic(query)
        assert decision.intent is Intent.SINGLE_IMAGE_CAPTIONING, (query, decision.intent)


def test_pointed_questions_stay_single_image_vqa():
    """The caption phrases must not swallow ordinary VQA — note "visible"."""
    from app.agent.llm import classify_heuristic

    vqa = [
        "How many aircraft are parked on the apron?",
        "How many aircraft are visible on the tarmac?",
        "Is there a runway in this scene?",
        "What colour is the largest building?",
    ]
    for query in vqa:
        decision = classify_heuristic(query)
        assert decision.intent is Intent.SINGLE_IMAGE_VQA, (query, decision.intent)


def test_more_specific_intents_beat_captioning():
    from app.agent.llm import classify_heuristic

    assert classify_heuristic("Describe what changed between the two captures.").intent is (
        Intent.CHANGE_VQA
    )
    assert classify_heuristic("Describe the scene using SAR backscatter.").intent is (
        Intent.OPTICAL_SAR_FUSION
    )


def test_classifier_prompt_lists_the_current_labels():
    from app.agent.llm import SYSTEM_PROMPT

    assert "single_image_captioning" in SYSTEM_PROMPT
    assert "single_image_grounding" not in SYSTEM_PROMPT
    for intent in Intent:
        assert intent.value in SYSTEM_PROMPT, intent


def test_intent_classifier_node_falls_back_without_a_key():
    from app.agent.nodes.intent import intent_classifier

    out = asyncio.run(
        intent_classifier({"status": QueryStatus.OK, "query": "Describe this image."})
    )
    assert out["intent"] is Intent.SINGLE_IMAGE_CAPTIONING, out
    assert out["intent_source"] == "heuristic", out
    assert out["steps"][0]["label"] == "Classify intent"


# --------------------------------------------------------------------------
# query routing
# --------------------------------------------------------------------------
def test_both_single_image_intents_share_one_task():
    assert INTENT_TO_TASK[Intent.SINGLE_IMAGE_VQA] is TaskType.VQA_GROUNDING
    assert INTENT_TO_TASK[Intent.SINGLE_IMAGE_CAPTIONING] is TaskType.VQA_GROUNDING
    assert INTENT_TO_MODES[Intent.SINGLE_IMAGE_CAPTIONING] == (UploadMode.SINGLE,)


def test_router_sends_captioning_to_the_vqa_node():
    from app.agent.nodes.router import route_to_specialist, task_router

    out = asyncio.run(
        task_router({"intent": Intent.SINGLE_IMAGE_CAPTIONING, "status": QueryStatus.OK})
    )
    assert out["task"] is TaskType.VQA_GROUNDING, out
    assert out["model_config_used"]["base_model"] == "llava-hf/llava-1.5-7b-hf"
    assert "llava-hf/llava-1.5-7b-hf" in out["steps"][0]["detail"], out["steps"]
    assert (
        route_to_specialist({"status": QueryStatus.OK, "task": out["task"]}) == "vqa_grounding_node"
    )


def test_router_edges_short_circuit_on_failure():
    from app.agent.nodes.router import route_after_validation, route_to_specialist

    assert (
        route_to_specialist({"status": QueryStatus.OK, "task": TaskType.CHANGE_DETECTION})
        == "change_node"
    )
    assert route_to_specialist({"status": QueryStatus.REJECTED}) == "output_combiner"
    assert route_after_validation({"status": QueryStatus.REJECTED}) == "output_combiner"
    assert route_after_validation({"status": QueryStatus.OK}) == "task_router"


def test_validator_accepts_captioning_on_a_single_image():
    from app.agent.nodes.validator import input_validator

    out = asyncio.run(
        input_validator(
            {
                "status": QueryStatus.OK,
                "intent": Intent.SINGLE_IMAGE_CAPTIONING,
                "upload_mode": UploadMode.SINGLE,
                "assets": [{"role": "primary", "stored_path": "/tmp/scene.tif"}],
            }
        )
    )
    assert "status" not in out, out
    assert out["steps"][0]["detail"].endswith("single_image_captioning"), out["steps"]


def test_validator_rejects_captioning_on_a_bi_temporal_upload():
    from app.agent.nodes.validator import input_validator

    out = asyncio.run(
        input_validator(
            {
                "status": QueryStatus.OK,
                "intent": Intent.SINGLE_IMAGE_CAPTIONING,
                "upload_mode": UploadMode.BI_TEMPORAL,
                "assets": [
                    {"role": "t0", "stored_path": "/tmp/a.tif"},
                    {"role": "t1", "stored_path": "/tmp/b.tif"},
                ],
            }
        )
    )
    assert out["status"] is QueryStatus.REJECTED, out
    assert "Describing one scene" in out["error"], out["error"]


# --------------------------------------------------------------------------
# vqa node — prompt framing and degradation
# --------------------------------------------------------------------------
class _RecordingEngine:
    """Stands in for GeoChatEngine so the prompt can be inspected without weights."""

    model_used = "recording-engine"

    def __init__(self):
        self.question: str | None = None

    def infer(self, image_path, question, **kwargs):
        self.question = question
        return {"answer": "a scene", "evidence": [], "confidence": 0.5, "tokens": 3}


def _run_vqa_with_recorder(intent: Intent) -> _RecordingEngine:
    from app.agent.nodes.vqa import vqa_grounding_node
    from app.core.model_registry import _load_geochat, get_registry

    recorder = _RecordingEngine()
    registry = get_registry()
    registry.register_loader("vqa_grounding", lambda task, config: recorder)
    try:
        asyncio.run(
            vqa_grounding_node(
                {
                    "status": QueryStatus.OK,
                    "intent": intent,
                    "query": "What is here?",
                    "assets": [{"role": "primary", "stored_path": "/tmp/scene.tif"}],
                    "parameters": {},
                }
            )
        )
    finally:
        registry.register_loader("vqa_grounding", _load_geochat)
    return recorder


def test_captioning_intent_widens_the_prompt():
    from app.agent.nodes.vqa import CAPTION_SUFFIX

    recorder = _run_vqa_with_recorder(Intent.SINGLE_IMAGE_CAPTIONING)
    assert recorder.question == f"What is here?{CAPTION_SUFFIX}", recorder.question


def test_vqa_intent_leaves_the_question_alone():
    recorder = _run_vqa_with_recorder(Intent.SINGLE_IMAGE_VQA)
    assert recorder.question == "What is here?", recorder.question


def test_vqa_node_degrades_when_the_model_cannot_load():
    from app.agent.nodes.vqa import vqa_grounding_node

    out = asyncio.run(
        vqa_grounding_node(
            {
                "status": QueryStatus.OK,
                "intent": Intent.SINGLE_IMAGE_VQA,
                "query": "How many aircraft?",
                "assets": [{"role": "primary", "stored_path": "/nonexistent/scene.tif"}],
                "parameters": {},
            }
        )
    )
    assert out["status"] is QueryStatus.UNAVAILABLE, out
    assert out["model_used"] == (
        "llava-hf/llava-1.5-7b-hf + eurosat_lora_v1_final"
    ), out["model_used"]
    assert out["steps"][0]["label"] == "Run VQA model", out["steps"]


# --------------------------------------------------------------------------
# graph / app wiring
# --------------------------------------------------------------------------
def test_graph_compiles():
    from app.agent.graph import build_graph

    assert build_graph() is not None


def test_app_exposes_the_documented_routes():
    from app.main import app

    paths = set(app.openapi()["paths"])
    for route in ("/api/upload", "/api/query", "/api/query/stream", "/api/health"):
        assert route in paths, (route, sorted(paths))


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
