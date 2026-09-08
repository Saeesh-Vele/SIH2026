#!/usr/bin/env python3
"""CPU-only checks for the two specialist nodes and their routing.

Same rules as test_cpu_smoke.py: no GPU, no MongoDB, no OpenRouter key, no
network, and no weights. The engines are exercised through their pure parts —
the difference map, the fusion arithmetic, the verbalisation — and the nodes are
exercised against recording stubs registered on the model registry, which is the
same seam real SSL4EO-S12 or DeCUR weights arrive through.

    python backend/tests/test_specialists.py
    pytest backend/tests
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.geochat import ModelUnavailable  # noqa: E402
from app.models.schemas import (  # noqa: E402
    Intent,
    QueryStatus,
    TaskType,
    UploadMode,
)

#: The shape every specialist node must return on a successful run. The point of
#: the contract is that the combiner, the API and the canvas never need to know
#: which node produced a result.
CONTRACT_KEYS = {"answer", "evidence", "confidence", "model_used"}


def assert_contract(out: dict[str, Any]) -> None:
    missing = CONTRACT_KEYS - set(out)
    assert not missing, f"missing contract keys: {sorted(missing)}"
    assert isinstance(out["answer"], str)
    assert isinstance(out["evidence"], list)
    assert isinstance(out["confidence"], float)
    assert isinstance(out["model_used"], str) and out["model_used"]
    for item in out["evidence"]:
        assert item.get("kind") in {"box", "mask"}, item
        assert isinstance(item.get("id"), str) and item["id"], item
        assert 0.0 <= item.get("confidence", -1) <= 1.0, item


# --------------------------------------------------------------------------
# Registry stubbing — the seam real checkpoints arrive through
# --------------------------------------------------------------------------
class _StubEngine:
    """Records the call and returns whatever the test asked it to."""

    def __init__(self, result: Any = None, raises: BaseException | None = None):
        self.model_used = "stub-engine"
        self._result = result
        self._raises = raises
        self.calls: list[tuple[tuple, dict]] = []

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._result


def _run_with_stub(task: str, node, state: dict[str, Any], stub: _StubEngine) -> dict[str, Any]:
    """Run `node` with `stub` standing in for the task's engine, then restore."""
    from app.core.model_registry import DEFAULT_LOADERS, get_registry

    registry = get_registry()
    registry.register_loader(task, lambda name, config: stub)
    try:
        return asyncio.run(node(state))
    finally:
        registry.register_loader(task, DEFAULT_LOADERS[task])


def _bi_temporal_state(query: str = "What changed?") -> dict[str, Any]:
    return {
        "status": QueryStatus.OK,
        "intent": Intent.CHANGE_VQA,
        "task": TaskType.CHANGE_DETECTION,
        "query": query,
        "upload_mode": UploadMode.BI_TEMPORAL,
        "assets": [
            {"role": "t0", "stored_path": "/tmp/t0.tif"},
            {"role": "t1", "stored_path": "/tmp/t1.tif"},
        ],
        "parameters": {},
    }


def _cross_modal_state(query: str = "Confirm the vessels.") -> dict[str, Any]:
    return {
        "status": QueryStatus.OK,
        "intent": Intent.OPTICAL_SAR_FUSION,
        "task": TaskType.OPTICAL_SAR_FUSION,
        "query": query,
        "upload_mode": UploadMode.CROSS_MODAL,
        "assets": [
            {"role": "optical", "stored_path": "/tmp/opt.tif"},
            {"role": "sar", "stored_path": "/tmp/sar.tif"},
        ],
        "parameters": {},
    }


CHANGE_RESULT = {
    "answer": "A new pier extends into the harbour.",
    "evidence": [
        {
            "kind": "mask",
            "id": "c0",
            "label": "changed region",
            "confidence": 0.71,
            "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]],
            "area": 0.12,
        }
    ],
    "confidence": 0.71,
    "model_used": "llava-hf/llava-1.5-7b-hf (prompted_diff)",
    "changed_fraction": 0.12,
    "grid": "24x24",
    "tokens": 18,
    "duration_ms": 900,
}

FUSION_RESULT = {
    "answer": "Three vessels are berthed along the quay.",
    "evidence": [],
    "confidence": 0.66,
    "model_used": "vit_base_patch16_224 (optical 13b + SAR 2b, concat) + llava-hf/llava-1.5-7b-hf",
    "fusion_description": "strong radar backscatter; flat optical response; the sensors disagree.",
    "fusion_stats": {"sar_share": 0.72, "agreement": 0.11, "width": 768.0},
    "fused_width": 1536,
    "duration_ms": 1400,
}


# --------------------------------------------------------------------------
# change detection — the difference map
# --------------------------------------------------------------------------
def test_threshold_map_marks_cells_at_or_above():
    from app.models.change import threshold_map

    assert threshold_map([[0.2, 0.5], [0.8, 0.1]], 0.5) == [[False, True], [True, False]]
    assert threshold_map([], 0.5) == []


def test_mask_evidence_finds_one_region():
    from app.models.change import mask_evidence

    scores = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.9, 0.8, 0.0],
        [0.0, 0.7, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    regions = mask_evidence(scores, threshold=0.5, min_region_area=0.0)
    assert len(regions) == 1, regions

    region = regions[0]
    assert region["kind"] == "mask" and region["id"] == "c0"
    assert region["label"] == "changed region"
    assert region["area"] == 0.25, region
    assert abs(region["confidence"] - 0.825) < 1e-6, region
    # Rows 1-2 of 4 and columns 1-2 of 4 -> the middle quarter of the frame.
    assert region["polygon"] == [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def test_mask_evidence_separates_disjoint_regions_and_ranks_them():
    from app.models.change import mask_evidence

    scores = [
        [0.9, 0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    regions = mask_evidence(scores, threshold=0.5, min_region_area=0.0)
    assert len(regions) == 2, regions
    # Strongest first, and ids follow the ranking rather than scan order.
    assert [r["id"] for r in regions] == ["c0", "c1"]
    assert regions[0]["confidence"] > regions[1]["confidence"]
    assert regions[0]["area"] < regions[1]["area"], "the stronger region is the smaller one here"


def test_mask_evidence_drops_noise_and_caps_the_count():
    from app.models.change import mask_evidence

    # One changed cell in a 4x4 grid is 6.25% of the scene.
    speckle = [[0.0] * 4 for _ in range(4)]
    speckle[0][0] = 0.9
    assert mask_evidence(speckle, threshold=0.5, min_region_area=0.1) == []
    assert len(mask_evidence(speckle, threshold=0.5, min_region_area=0.01)) == 1

    # Eight isolated cells, only three of them wanted.
    scattered = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        scattered[i][i] = 0.5 + i / 100
    regions = mask_evidence(scattered, threshold=0.4, min_region_area=0.0, max_regions=3)
    assert len(regions) == 3
    assert [r["id"] for r in regions] == ["c0", "c1", "c2"]


def test_mask_evidence_handles_no_change_and_empty_input():
    from app.models.change import mask_evidence

    assert mask_evidence([[0.1, 0.2], [0.05, 0.0]], threshold=0.5) == []
    assert mask_evidence([], threshold=0.5) == []
    assert mask_evidence([[]], threshold=0.5) == []


def test_changed_fraction():
    from app.models.change import changed_fraction

    assert changed_fraction([[0.9, 0.1], [0.9, 0.9]], 0.5) == 0.75
    assert changed_fraction([], 0.5) == 0.0


def test_change_prompt_carries_both_frames():
    from app.models.change import CHANGE_INSTRUCTION, build_change_prompt

    plain = build_change_prompt()
    assert plain.count("<image>") == 2, plain
    assert CHANGE_INSTRUCTION in plain
    assert plain.rstrip().endswith("ASSISTANT:")

    asked = build_change_prompt("Did the pier get longer?")
    assert asked.count("<image>") == 2
    assert "Did the pier get longer?" in asked
    assert build_change_prompt("   ") == plain, "a blank question must not add a clause"


def test_change_config_reads_the_real_yaml():
    import yaml

    from app.models.change import ChangeConfig

    raw = yaml.safe_load((BACKEND / "model_config.yaml").read_text())["change_detection"]
    config = ChangeConfig.from_mapping(raw)
    assert config.method == "prompted_diff"
    assert config.base_model == "llava-hf/llava-1.5-7b-hf"
    assert 0.0 < config.diff_threshold < 1.0
    assert config.as_vqa_config().base_model == config.base_model
    assert "prompted_diff" in ChangeConfig.from_mapping({}).method


def test_change_engine_reports_its_checkpoint_without_loading_it():
    from app.models.change import ChangeConfig, ChangeDetectorEngine

    engine = ChangeDetectorEngine(ChangeConfig())
    assert engine.loaded is False
    assert engine.model_used == "llava-hf/llava-1.5-7b-hf (prompted_diff)"


# --------------------------------------------------------------------------
# change detection — the node
# --------------------------------------------------------------------------
def test_change_node_returns_the_contract():
    from app.agent.nodes.change import change_node

    stub = _StubEngine(result=CHANGE_RESULT)
    out = _run_with_stub("change_detection", change_node, _bi_temporal_state(), stub)

    assert_contract(out)
    assert out["answer"] == CHANGE_RESULT["answer"]
    assert out["evidence"] == CHANGE_RESULT["evidence"]
    assert out["model_used"] == CHANGE_RESULT["model_used"]
    assert "status" not in out, "a successful run leaves the status alone"

    # Both frames reach the engine, earlier first, with the query.
    (args, kwargs) = stub.calls[0]
    assert args[:3] == ("/tmp/t0.tif", "/tmp/t1.tif", "What changed?")
    assert set(kwargs) == {"max_new_tokens", "temperature"}

    labels = {m["label"]: m["value"] for m in out["metrics"]}
    assert labels["changed regions"] == "1"
    assert labels["scene changed"] == "12.0%"
    assert labels["diff grid"] == "24x24"
    assert out["steps"][0]["label"] == "Run change detection"


def test_change_node_degrades_when_the_model_is_unavailable():
    from app.agent.nodes.change import change_node

    stub = _StubEngine(raises=ModelUnavailable("no CUDA GPU here"))
    out = _run_with_stub("change_detection", change_node, _bi_temporal_state(), stub)

    assert out["status"] is QueryStatus.UNAVAILABLE, out
    assert out["error"] == "no CUDA GPU here"
    assert out["steps"][0]["status"] == "failed"


def test_change_node_survives_a_raising_engine():
    from app.agent.nodes.change import change_node

    stub = _StubEngine(raises=RuntimeError("cuda oom"))
    out = _run_with_stub("change_detection", change_node, _bi_temporal_state(), stub)

    assert out["status"] is QueryStatus.FAILED, out
    assert "cuda oom" in out["error"]


def test_change_node_needs_both_frames():
    from app.agent.nodes.change import change_node

    state = _bi_temporal_state()
    state["assets"] = [{"role": "t0", "stored_path": "/tmp/t0.tif"}]
    stub = _StubEngine(result=CHANGE_RESULT)
    out = _run_with_stub("change_detection", change_node, state, stub)

    assert out["status"] is QueryStatus.FAILED, out
    assert stub.calls == [], "the engine must not run without both frames"


def test_change_node_skips_an_already_failed_run():
    from app.agent.nodes.change import change_node

    state = _bi_temporal_state()
    state["status"] = QueryStatus.REJECTED
    stub = _StubEngine(result=CHANGE_RESULT)
    assert _run_with_stub("change_detection", change_node, state, stub) == {}
    assert stub.calls == []


# --------------------------------------------------------------------------
# fusion — the arithmetic and the verbalisation
# --------------------------------------------------------------------------
def test_fuse_embeddings_concat_and_gated():
    from app.models.fusion import fuse_embeddings

    assert fuse_embeddings([1.0, 2.0], [3.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]

    gated = fuse_embeddings([1.0, 0.0], [0.0, 1.0], "gated")
    assert len(gated) == 2
    assert all(abs(v - 0.5) < 1e-9 for v in gated), gated

    assert fuse_embeddings([0.0, 0.0], [0.0, 0.0], "gated") == [0.0, 0.0]


def test_fuse_embeddings_rejects_what_it_cannot_do():
    from app.models.fusion import fuse_embeddings

    for optical, sar, strategy, needle in (
        ([1.0], [1.0, 2.0], "concat", "match in width"),
        ([], [], "concat", "empty"),
        ([1.0], [1.0], "cross_attention", "trained head"),
    ):
        try:
            fuse_embeddings(optical, sar, strategy)
        except ValueError as exc:
            assert needle in str(exc), (strategy, exc)
        else:
            raise AssertionError(f"{strategy} with {optical}/{sar} should have raised")


def test_fusion_stats_measure_dominance_and_agreement():
    from app.models.fusion import fusion_stats

    # Identical embeddings: balanced, perfectly agreed.
    same = fusion_stats([1.0, 1.0], [1.0, 1.0])
    assert same["sar_share"] == 0.5
    assert same["agreement"] == 1.0

    # Orthogonal: no agreement.
    assert fusion_stats([1.0, 0.0], [0.0, 1.0])["agreement"] == 0.0

    # A loud SAR response takes the larger share.
    loud = fusion_stats([0.1, 0.1], [5.0, 5.0])
    assert loud["sar_share"] > 0.9, loud

    # A zero embedding must not divide by zero.
    assert fusion_stats([0.0, 0.0], [0.0, 0.0])["agreement"] == 0.0


def test_describe_fusion_uses_the_configured_vocabulary():
    from app.models.fusion import describe_fusion

    vocabulary = {
        "sar_high": "SAR-HIGH",
        "sar_low": "SAR-LOW",
        "optical_high": "OPT-HIGH",
        "optical_low": "OPT-LOW",
        "agree": "AGREE",
        "disagree": "DISAGREE",
    }
    sar_led = describe_fusion({"sar_share": 0.8, "agreement": 0.9}, vocabulary)
    assert "SAR-HIGH" in sar_led and "OPT-LOW" in sar_led and "AGREE" in sar_led
    assert sar_led.endswith(".")

    optical_led = describe_fusion({"sar_share": 0.2, "agreement": -0.4}, vocabulary)
    assert "SAR-LOW" in optical_led and "OPT-HIGH" in optical_led and "DISAGREE" in optical_led

    # A partial vocabulary falls back rather than raising a KeyError.
    assert "SAR-HIGH" in describe_fusion({"sar_share": 0.9, "agreement": 0.9}, {"sar_high": "SAR-HIGH"})


def test_fusion_prompt_holds_one_image_and_both_halves():
    from app.models.fusion import build_fusion_prompt

    prompt = build_fusion_prompt("strong radar backscatter.", "How many vessels?")
    assert prompt.count("<image>") == 1, "the VQA path takes exactly one image token"
    assert "strong radar backscatter." in prompt
    assert "How many vessels?" in prompt
    assert prompt.rstrip().endswith("ASSISTANT:")


def test_fusion_config_reads_the_real_yaml():
    import yaml

    from app.models.fusion import FusionConfig

    raw = yaml.safe_load((BACKEND / "model_config.yaml").read_text())["optical_sar_fusion"]
    config = FusionConfig.from_mapping(raw)
    assert config.fusion_strategy == "concat"
    # `encoder` is the timm architecture, shared; the weights are per modality.
    assert config.encoder == "vit_base_patch16_224", config.encoder
    assert "optical" in (config.optical_checkpoint_path or ""), config.optical_checkpoint_path
    assert "sar" in (config.sar_checkpoint_path or ""), config.sar_checkpoint_path
    assert config.optical_checkpoint_path != config.sar_checkpoint_path
    # Sentinel-2 L1C is 13 bands; Sentinel-1 is VV + VH.
    assert config.optical_in_chans == 13, config.optical_in_chans
    assert config.sar_in_chans == 2, config.sar_in_chans
    # The YAML wording wins, and any band it omits still has a default.
    assert config.feature_vocabulary["sar_high"] == raw["feature_vocabulary"]["sar_high"]
    assert set(config.feature_vocabulary) >= {
        "sar_high",
        "sar_low",
        "optical_high",
        "optical_low",
        "agree",
        "disagree",
    }


def test_fusion_config_rejects_an_unknown_strategy():
    from app.models.fusion import FusionConfig

    try:
        FusionConfig.from_mapping({"fusion_strategy": "telepathy"})
    except ValueError as exc:
        assert "telepathy" in str(exc)
    else:
        raise AssertionError("an unknown fusion_strategy must not be accepted silently")


def test_fusion_config_rejects_the_single_checkpoint_schema():
    """One checkpoint cannot serve both modalities, so the old key must not pass."""
    from app.models.fusion import FusionConfig

    try:
        FusionConfig.from_mapping({"checkpoint_path": "./weights/fusion/ssl4eo.pth"})
    except ValueError as exc:
        assert "optical_checkpoint_path" in str(exc) and "sar_checkpoint_path" in str(exc), exc
    else:
        raise AssertionError("a legacy single-checkpoint config must fail loudly, not silently")


def test_config_routes_each_modality_to_its_own_weights():
    from app.models.fusion import MODALITIES, FusionConfig

    config = FusionConfig(
        optical_checkpoint_path="/w/optical.pth",
        optical_in_chans=13,
        sar_checkpoint_path="/w/sar.pth",
        sar_in_chans=2,
    )
    assert config.checkpoint_for("optical") == ("/w/optical.pth", 13)
    assert config.checkpoint_for("sar") == ("/w/sar.pth", 2)
    assert MODALITIES == ("optical", "sar"), "optical is concatenated first"

    try:
        config.checkpoint_for("lidar")
    except ValueError as exc:
        assert "lidar" in str(exc)
    else:
        raise AssertionError("an unknown modality must not resolve to either encoder")


def test_fusion_engine_refuses_to_run_without_weights():
    """Random weights would still produce a confident sentence. They must not."""
    from app.models.fusion import FusionConfig, FusionEngine

    complete = {
        "optical_checkpoint_path": "/nonexistent/optical.pth",
        "optical_in_chans": 13,
        "sar_checkpoint_path": "/nonexistent/sar.pth",
        "sar_in_chans": 2,
    }
    # `load` walks MODALITIES in order, so optical is the one that reports.
    # Whichever modality fails, the message has to name it — a bare "weights not
    # found" would leave you guessing which of the two paths to fix.
    cases = [
        (FusionConfig(**{**complete, "optical_checkpoint_path": None}), "no optical_checkpoint_path"),
        (FusionConfig(**complete), "optical encoder weights not found"),
    ]
    for config, needle in cases:
        engine = FusionEngine(config)
        assert engine.loaded is False
        try:
            engine.load()
        except ModelUnavailable as exc:
            assert needle in str(exc), (needle, exc)
        else:
            raise AssertionError(f"expected {needle!r} to be refused")

    # The SAR half is refused the same way; it is only reached once optical is
    # satisfiable, so its message is checked on the string the config builds.
    sar_only = FusionConfig(**{**complete, "sar_checkpoint_path": None})
    assert sar_only.checkpoint_for("sar") == (None, 2)
    assert sar_only.checkpoint_for("optical")[0] == "/nonexistent/optical.pth"


def test_embed_rejects_a_modality_it_has_no_encoder_for():
    """Routing is checked before loading, so this needs no weights."""
    from app.models.fusion import FusionConfig, FusionEngine

    engine = FusionEngine(FusionConfig())
    try:
        engine.embed("/tmp/scene.tif", "lidar")
    except ValueError as exc:
        assert "lidar" in str(exc), exc
    else:
        raise AssertionError("embed must not silently pick an encoder for an unknown modality")


class _FakeWeight:
    """Stands in for a patch-embed tensor without importing torch."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)


def test_checkpoint_band_count_is_read_not_assumed():
    """The band count is what a remote-sensing checkpoint most often differs on."""
    from app.models.fusion import _checkpoint_in_chans, _unwrap_state_dict

    # Sentinel-1 SAR: VV + VH.
    assert _checkpoint_in_chans({"patch_embed.proj.weight": _FakeWeight((768, 2, 16, 16))}) == 2
    # Sentinel-2 L1C: 13 bands.
    assert _checkpoint_in_chans({"patch_embed.proj.weight": _FakeWeight((768, 13, 16, 16))}) == 13
    # Nothing to read from.
    assert _checkpoint_in_chans({}) is None
    assert _checkpoint_in_chans({"patch_embed.proj.weight": _FakeWeight((768, 16))}) is None

    # Publishers nest the weights under different keys; all of them unwrap.
    inner = {"patch_embed.proj.weight": _FakeWeight((768, 2, 16, 16))}
    for key in ("model", "state_dict", "model_state_dict"):
        assert _unwrap_state_dict({key: inner, "optimizer": {}}) is inner, key
    assert _unwrap_state_dict(inner) is inner, "a bare state dict passes through"


def test_to_tensor_keeps_arrays_and_pil_images_apart():
    """A numpy array also has .resize and .tobytes, and means something else by them.

    Skipped where torch is absent — it is an inference dependency, not a service
    one, and this is the only check here that needs it.
    """
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover - depends on install
        return

    from app.models.fusion import _to_tensor

    # The array branch: (bands, h, w), as rasterio hands it over.
    bands = 13
    array = [[[float(b + 1)] * 8 for _ in range(8)] for b in range(bands)]
    tensor = _to_tensor(array, bands, (4, 4), (0.5,) * bands, (0.5,) * bands)
    assert tuple(tensor.shape) == (bands, 4, 4), tuple(tensor.shape)

    # RGB statistics do not describe 13 bands, so they are replaced rather than
    # broadcast — a wrong normalisation would be silent and would skew every
    # embedding downstream.
    rgb_stats = _to_tensor(array, bands, (4, 4), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert tuple(rgb_stats.shape) == (bands, 4, 4)

    # A single-band array is promoted rather than rejected.
    assert tuple(_to_tensor([[1.0] * 8] * 8, 1, (4, 4), (0.5,), (0.5,)).shape) == (1, 4, 4)


def test_fusion_engine_names_both_halves():
    from app.models.fusion import FusionConfig, FusionEngine

    used = FusionEngine(FusionConfig()).model_used
    # Both band counts appear: they are what distinguishes the two encoders.
    assert used == (
        "vit_base_patch16_224 (optical 13b + SAR 2b, concat) + llava-hf/llava-1.5-7b-hf"
    ), used


# --------------------------------------------------------------------------
# fusion — the node
# --------------------------------------------------------------------------
def test_fusion_node_returns_the_contract():
    from app.agent.nodes.fusion import fusion_node

    stub = _StubEngine(result=FUSION_RESULT)
    out = _run_with_stub("optical_sar_fusion", fusion_node, _cross_modal_state(), stub)

    assert_contract(out)
    assert out["answer"] == FUSION_RESULT["answer"]
    assert out["evidence"] == [], "fusion describes a scene; it produces no geometry"
    assert out["model_used"] == FUSION_RESULT["model_used"]
    assert "status" not in out

    # Optical first, then SAR, then the question.
    (args, _kwargs) = stub.calls[0]
    assert args[:3] == ("/tmp/opt.tif", "/tmp/sar.tif", "Confirm the vessels.")

    labels = {m["label"]: m["value"] for m in out["metrics"]}
    assert labels["sensor summary"] == FUSION_RESULT["fusion_description"]
    assert labels["SAR share"] == "0.72"
    assert labels["sensor agreement"] == "0.11"
    assert labels["fused width"] == "1536"
    assert out["steps"][0]["label"] == "Run optical-SAR fusion"


def test_fusion_node_degrades_when_weights_are_missing():
    from app.agent.nodes.fusion import fusion_node

    stub = _StubEngine(raises=ModelUnavailable("fusion encoder weights not found"))
    out = _run_with_stub("optical_sar_fusion", fusion_node, _cross_modal_state(), stub)

    assert out["status"] is QueryStatus.UNAVAILABLE, out
    assert "weights not found" in out["error"]


def test_fusion_node_survives_a_raising_engine():
    from app.agent.nodes.fusion import fusion_node

    stub = _StubEngine(raises=RuntimeError("encoder shape mismatch"))
    out = _run_with_stub("optical_sar_fusion", fusion_node, _cross_modal_state(), stub)

    assert out["status"] is QueryStatus.FAILED, out
    assert "encoder shape mismatch" in out["error"]


def test_fusion_node_needs_both_modalities():
    from app.agent.nodes.fusion import fusion_node

    state = _cross_modal_state()
    state["assets"] = [{"role": "optical", "stored_path": "/tmp/opt.tif"}]
    stub = _StubEngine(result=FUSION_RESULT)
    out = _run_with_stub("optical_sar_fusion", fusion_node, state, stub)

    assert out["status"] is QueryStatus.FAILED, out
    assert stub.calls == []


def test_fusion_node_skips_an_already_failed_run():
    from app.agent.nodes.fusion import fusion_node

    state = _cross_modal_state()
    state["status"] = QueryStatus.UNAVAILABLE
    stub = _StubEngine(result=FUSION_RESULT)
    assert _run_with_stub("optical_sar_fusion", fusion_node, state, stub) == {}
    assert stub.calls == []


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def test_every_task_has_its_own_node():
    from app.agent.graph import SPECIALIST_NODES
    from app.agent.nodes.router import TASK_NODES

    assert set(TASK_NODES) == set(TaskType), "a task with no node would fall to the stub"
    assert len(set(TASK_NODES.values())) == len(TaskType), "each task needs a distinct node"
    for node in TASK_NODES.values():
        assert node in SPECIALIST_NODES, node


def test_route_to_specialist_dispatches_per_task():
    from app.agent.nodes.router import route_to_specialist

    expected = {
        TaskType.VQA_GROUNDING: "vqa_grounding_node",
        TaskType.CHANGE_DETECTION: "change_node",
        TaskType.OPTICAL_SAR_FUSION: "fusion_node",
    }
    for task, node in expected.items():
        assert route_to_specialist({"status": QueryStatus.OK, "task": task}) == node

    assert route_to_specialist({"status": QueryStatus.OK, "task": None}) == "specialist_stub"
    assert route_to_specialist({"status": QueryStatus.FAILED, "task": TaskType.VQA_GROUNDING}) == (
        "output_combiner"
    )


def test_task_router_resolves_the_new_specialists():
    from app.agent.nodes.router import task_router

    change = asyncio.run(task_router({"intent": Intent.CHANGE_VQA, "status": QueryStatus.OK}))
    assert change["task"] is TaskType.CHANGE_DETECTION
    assert change["model_config_used"]["method"] == "prompted_diff"

    fusion = asyncio.run(
        task_router({"intent": Intent.OPTICAL_SAR_FUSION, "status": QueryStatus.OK})
    )
    assert fusion["task"] is TaskType.OPTICAL_SAR_FUSION
    assert fusion["model_config_used"]["fusion_strategy"] == "concat"
    # The encoder is the half that distinguishes fusion, so it is what the trace
    # names — not the VQA base_model the entry also carries.
    detail = fusion["steps"][0]["detail"]
    assert "vit_base_patch16_224" in detail, detail
    assert "llava" not in detail.lower(), detail


def test_registry_builds_real_engines_for_every_task():
    from app.core.model_registry import get_registry
    from app.models.change import ChangeDetectorEngine
    from app.models.fusion import FusionEngine
    from app.models.geochat import GeoChatEngine

    registry = get_registry()
    expected = {
        "vqa_grounding": GeoChatEngine,
        "change_detection": ChangeDetectorEngine,
        "optical_sar_fusion": FusionEngine,
    }
    for task, kind in expected.items():
        engine = registry.get_model(task)
        assert isinstance(engine, kind), (task, type(engine))
        assert engine.loaded is False, f"{task} must not load weights at construction"


def test_node_sequence_names_the_specialist_that_will_run():
    from app.agent.graph import NODE_SEQUENCE, node_sequence

    assert node_sequence(TaskType.CHANGE_DETECTION)[4] == "change_node"
    assert node_sequence(TaskType.OPTICAL_SAR_FUSION)[4] == "fusion_node"
    assert node_sequence(TaskType.VQA_GROUNDING)[4] == "vqa_grounding_node"
    assert node_sequence()[4] == "vqa_grounding_node", "the default plan is the common path"
    assert NODE_SEQUENCE == node_sequence()
    for task in (None, *TaskType):
        assert node_sequence(task)[0] == "intake"
        assert node_sequence(task)[-1] == "trace_logger"


def test_graph_registers_both_new_nodes():
    from app.agent.graph import build_graph

    nodes = set(build_graph().get_graph().nodes)
    assert {"change_node", "fusion_node"} <= nodes, sorted(nodes)


# --------------------------------------------------------------------------
# intent — dispatch on query *and* input type
# --------------------------------------------------------------------------
def test_bound_imagery_corrects_an_impossible_label():
    from app.agent.llm import IntentDecision
    from app.agent.nodes.intent import reconcile_with_upload

    weak = IntentDecision(Intent.SINGLE_IMAGE_CAPTIONING, 0.4, "no keyword", "heuristic")

    fixed, note = reconcile_with_upload(weak, UploadMode.BI_TEMPORAL)
    assert fixed.intent is Intent.CHANGE_VQA, fixed
    assert note and "bi_temporal" in note
    assert "corrected from single_image_captioning" in fixed.reason

    fixed, note = reconcile_with_upload(weak, UploadMode.CROSS_MODAL)
    assert fixed.intent is Intent.OPTICAL_SAR_FUSION
    assert note and "cross_modal" in note


def test_reconciliation_leaves_everything_else_alone():
    from app.agent.llm import IntentDecision
    from app.agent.nodes.intent import reconcile_with_upload

    vqa = IntentDecision(Intent.SINGLE_IMAGE_VQA, 0.9, "counting question", "llm")

    # Single-image mode admits two intents, so it cannot pick between them.
    assert reconcile_with_upload(vqa, UploadMode.SINGLE) == (vqa, None)
    # No upload bound yet.
    assert reconcile_with_upload(vqa, None) == (vqa, None)
    # Already right for the mode.
    change = IntentDecision(Intent.CHANGE_VQA, 0.8, "temporal", "llm")
    assert reconcile_with_upload(change, UploadMode.BI_TEMPORAL) == (change, None)


def test_intent_node_dispatches_on_query_and_input_together():
    from app.agent.nodes.intent import intent_classifier

    # The words alone read as captioning; the bound pair makes it a comparison.
    out = asyncio.run(
        intent_classifier(
            {
                "status": QueryStatus.OK,
                "query": "Describe this scene.",
                "upload_mode": UploadMode.BI_TEMPORAL,
            }
        )
    )
    assert out["intent"] is Intent.CHANGE_VQA, out
    assert "corrected" in out["steps"][0]["detail"], out["steps"]

    # A forced intent is the caller's decision and is never overridden.
    forced = asyncio.run(
        intent_classifier(
            {
                "status": QueryStatus.OK,
                "query": "Describe this scene.",
                "upload_mode": UploadMode.BI_TEMPORAL,
                "forced_intent": Intent.SINGLE_IMAGE_CAPTIONING,
            }
        )
    )
    assert forced["intent"] is Intent.SINGLE_IMAGE_CAPTIONING, forced


# --------------------------------------------------------------------------
# validator — the per-task input rules
# --------------------------------------------------------------------------
def _validate(state: dict[str, Any]) -> dict[str, Any]:
    from app.agent.nodes.validator import input_validator

    return asyncio.run(input_validator({"status": QueryStatus.OK, **state}))


def test_validator_accepts_a_well_formed_pair_for_each_task():
    for state in (_bi_temporal_state(), _cross_modal_state()):
        out = _validate({k: state[k] for k in ("intent", "upload_mode", "assets")})
        assert "status" not in out, out


def test_validator_rejects_a_single_file_bound_to_both_slots():
    for state, needle in (
        (_bi_temporal_state(), "nothing to compare"),
        (_cross_modal_state(), "only one modality"),
    ):
        assets = [dict(a, stored_path="/tmp/same.tif") for a in state["assets"]]
        out = _validate({**{k: state[k] for k in ("intent", "upload_mode")}, "assets": assets})
        assert out["status"] is QueryStatus.REJECTED, out
        assert needle in out["error"], out["error"]


def test_validator_rejects_the_wrong_number_of_images_per_task():
    for state, needle in (
        (_bi_temporal_state(), "exactly 2 captures"),
        (_cross_modal_state(), "exactly 2 images"),
    ):
        assets = [*state["assets"], dict(state["assets"][0], stored_path="/tmp/extra.tif")]
        out = _validate({**{k: state[k] for k in ("intent", "upload_mode")}, "assets": assets})
        assert out["status"] is QueryStatus.REJECTED, out
        assert needle in out["error"], out["error"]


def test_validator_rejects_images_that_are_not_co_registered():
    """Different pixel dimensions mean the two frames are not the same footprint.

    The check is skipped where Pillow cannot open the files — that is an
    inference dependency, and on an API-only machine "cannot check" must never
    become "invalid".
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on install
        state = _bi_temporal_state()
        out = _validate({k: state[k] for k in ("intent", "upload_mode", "assets")})
        assert "status" not in out, "unreadable sizes must not reject the run"
        return

    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "small.png"
        large = Path(tmp) / "large.png"
        Image.new("RGB", (16, 16)).save(small)
        Image.new("RGB", (32, 32)).save(large)

        for state, roles in ((_bi_temporal_state(), ("t0", "t1")), (_cross_modal_state(), ("optical", "sar"))):
            mismatched = [
                {"role": roles[0], "stored_path": str(small)},
                {"role": roles[1], "stored_path": str(large)},
            ]
            out = _validate(
                {**{k: state[k] for k in ("intent", "upload_mode")}, "assets": mismatched}
            )
            assert out["status"] is QueryStatus.REJECTED, out
            assert "co-registered" in out["error"], out["error"]

            matched = [
                {"role": roles[0], "stored_path": str(small)},
                {"role": roles[1], "stored_path": str(Path(tmp) / "small2.png")},
            ]
            Image.new("RGB", (16, 16)).save(matched[1]["stored_path"])
            out = _validate(
                {**{k: state[k] for k in ("intent", "upload_mode")}, "assets": matched}
            )
            assert "status" not in out, out


# --------------------------------------------------------------------------
# fusion — the RGB preview the VQA model is actually shown
# --------------------------------------------------------------------------
def _marked_scene(bands: int, size: int = 8):
    """A (bands, h, w) raster where band `b` is the only one lit at column `b`.

    Per-band contrast stretching erases any relationship between the bands'
    values, so a marker is the only way to say which band came out where.
    """
    import numpy as np

    scene = np.zeros((bands, size, size), dtype="uint16")
    for band in range(bands):
        scene[band, 0, band % size] = 1000
    return scene


def test_rgb_preview_composes_sentinel2_true_colour():
    """13 bands in, three channels out, and B4/B3/B2 in the right order."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on install
        return

    from app.models.fusion import rgb_preview

    preview = rgb_preview(_marked_scene(13), 13)
    assert preview.mode == "RGB", preview.mode
    assert preview.size == (8, 8), preview.size

    pixels = np.asarray(preview)
    assert pixels.shape == (8, 8, 3), pixels.shape
    assert pixels.dtype == np.uint8, pixels.dtype

    # 1-indexed B4/B3/B2 are 0-indexed 3/2/1, and each marker lands in exactly
    # one channel. B1 — aerosol, colourless — must not appear at all.
    for channel, band in enumerate((3, 2, 1)):
        lit = pixels[0, band, channel]
        assert lit == 255, (channel, band, lit)
        assert list(pixels[0, band]).count(255) == 1, pixels[0, band]
    assert pixels[0, 0].tolist() == [0, 0, 0], "B1 is not part of true colour"


def test_rgb_preview_falls_back_to_grey_for_sar():
    """Two polarisations have no colour mapping; inventing one would be a lie."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on install
        return

    from app.models.fusion import rgb_preview

    preview = rgb_preview(_marked_scene(2), 2)
    pixels = np.asarray(preview)
    assert preview.mode == "RGB" and pixels.shape == (8, 8, 3), pixels.shape
    # VV replicated across all three channels: grey, not a false colour.
    assert (pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all()
    assert pixels[0, 0].tolist() == [255, 255, 255], "VV is the band shown"
    assert pixels[0, 1].tolist() == [0, 0, 0], "VH's marker must not leak in"

    # A single-band raster arrives as (h, w) from tifffile, not (1, h, w).
    flat = rgb_preview(np.asarray(_marked_scene(1))[0], 1)
    assert np.asarray(flat).shape == (8, 8, 3)


def test_rgb_preview_leaves_an_ordinary_image_alone():
    """A 3-band PNG is already what the VQA model wants; nothing to render."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on install
        return

    from app.models.fusion import rgb_preview

    png = Image.new("RGB", (16, 16), (10, 20, 30))
    assert rgb_preview(png, 3) is png, "an RGB image must pass through untouched"

    # A greyscale PNG is viewable but not 3-channel; LLaVA needs the conversion.
    grey = rgb_preview(Image.new("L", (16, 16), 128), 1)
    assert grey.mode == "RGB" and grey.getpixel((0, 0)) == (128, 128, 128)


def test_fusion_infer_shows_the_vqa_a_picture_not_thirteen_bands():
    """The regression: PIL cannot open a 13-band GeoTIFF, so it must never see one."""
    try:
        import numpy as np
        from PIL.Image import Image as PILImage
    except ImportError:  # pragma: no cover - depends on install
        return

    from app.models import fusion as fusion_module
    from app.models.fusion import FusionConfig, FusionEngine

    scenes = {"optical.tif": (_marked_scene(13), 13), "sar.tif": (_marked_scene(2), 2)}

    class _Engine(FusionEngine):
        """Real reading and real verbalisation; only the weights are stubbed."""

        def load(self) -> None:
            self._encoders = {"optical": object(), "sar": object()}

        def _embed_read(self, image, bands, modality):
            assert isinstance(image, np.ndarray), (modality, type(image))
            return [1.0, 0.0] if modality == "optical" else [0.6, 0.8]

    vqa = _StubEngine({"answer": "Two vessels.", "confidence": 0.71, "evidence": []})
    engine = _Engine(FusionConfig(), vqa=vqa)

    original = fusion_module._read_bands
    fusion_module._read_bands = lambda path: scenes[Path(path).name]
    try:
        out = engine.infer("/scenes/optical.tif", "/scenes/sar.tif", "How many vessels?")
    finally:
        fusion_module._read_bands = original

    assert_contract(out)
    assert out["answer"] == "Two vessels."
    assert out["fused_width"] == 4, out["fused_width"]

    (shown, question), _ = vqa.calls[0]
    assert isinstance(shown, PILImage), f"the VQA model was handed a {type(shown)}"
    assert shown.mode == "RGB" and shown.size == (8, 8), (shown.mode, shown.size)
    assert question == "How many vessels?"


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
