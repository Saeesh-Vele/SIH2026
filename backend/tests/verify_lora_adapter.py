#!/usr/bin/env python3
"""Check that the configured LoRA adapter is wired up and fits its base model.

`vqa_grounding.adapter_path` in model_config.yaml points at a PEFT adapter
directory that is applied on top of `base_model` at load time. This confirms
the wiring end to end — registry, config, path resolution, adapter metadata —
without loading a single weight:

    python backend/tests/verify_lora_adapter.py

What it checks, and why each one has bitten before:

- The path resolves. `adapter_path` is relative to the backend root, not the
  working directory, so a path that looks right in the YAML can still miss
  depending on where uvicorn was started from. This goes through
  `GeoChatConfig.resolved_adapter_path`, the same property the engine uses.
- `adapter_config.json` parses and is a LoRA adapter. A directory of the right
  name holding the wrong thing fails at first inference, on a GPU box, in
  front of whoever is demoing.
- Its `base_model_name_or_path` matches `base_model`. A LoRA delta targets
  named modules in a specific architecture; applied to a different base it
  either errors or, worse, silently attaches to a subset.
- The weights file exists and its size is sane. Config without weights is the
  usual half-committed-adapter failure.
- The registry hands back an engine that reports the adapter in `model_used`,
  which is what lands in the execution trace.

CPU only: nothing is loaded, so this runs anywhere. Deliberately NOT named
test_* — it reads real files from disk, and the suites beside it are
dependency-free by design.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.geochat import GeoChatConfig  # noqa: E402

#: PEFT writes one or the other depending on its version and `safe_serialization`.
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")

#: An r=16 adapter on q_proj/v_proj of a 7B model lands near 40 MB. Two orders of
#: magnitude either side means something other than a LoRA adapter is here — a
#: git-lfs pointer stub below, a merged full checkpoint above.
MIN_WEIGHT_MB, MAX_WEIGHT_MB = 1.0, 2000.0


def check_config(config: GeoChatConfig) -> tuple[bool, Path | None]:
    """Resolve the adapter path the way the engine does."""
    print("\n=== CONFIG " + "=" * 53)
    print(f"  [config] base_model   = {config.base_model}")
    print(f"  [config] adapter_path = {config.adapter_path}")

    if not config.adapter_path:
        print(
            "  FAIL: vqa_grounding.adapter_path is unset, so the engine runs the "
            "zero-shot base. Point it at a PEFT adapter directory.",
            file=sys.stderr,
        )
        return False, None

    adapter = config.resolved_adapter_path
    assert adapter is not None  # guarded above; narrows the type
    print(f"  [resolve] -> {adapter}")
    print(f"  [resolve] relative to backend root {BACKEND} (not the working directory)")

    if not adapter.is_dir():
        print(f"  FAIL: no directory at {adapter}", file=sys.stderr)
        return False, adapter

    contents = sorted(p.name for p in adapter.iterdir())
    print(f"  [resolve] contents = {', '.join(contents)}")
    return True, adapter


def check_adapter(adapter: Path, base_model: str) -> bool:
    """Read the adapter's own metadata and hold it against the configured base."""
    print("\n=== ADAPTER " + "=" * 52)
    ok = True

    config_file = adapter / "adapter_config.json"
    if not config_file.is_file():
        print(f"  FAIL: no adapter_config.json in {adapter}", file=sys.stderr)
        return False

    try:
        meta = json.loads(config_file.read_text())
    except json.JSONDecodeError as exc:
        print(f"  FAIL: adapter_config.json does not parse: {exc}", file=sys.stderr)
        return False
    print(f"  [meta]   adapter_config.json parsed, {len(meta)} keys")

    peft_type = meta.get("peft_type")
    print(f"  [meta]   peft_type     = {peft_type}")
    if peft_type != "LORA":
        print(f"  FAIL: expected a LORA adapter, got peft_type={peft_type!r}", file=sys.stderr)
        ok = False

    trained_on = meta.get("base_model_name_or_path")
    print(f"  [meta]   trained on    = {trained_on}", end="")
    if trained_on == base_model:
        print("  (matches model_config.yaml)")
    else:
        print(f"  <-- model_config.yaml says {base_model}")
        print(
            f"  FAIL: adapter was tuned against {trained_on!r} but vqa_grounding.base_model "
            f"is {base_model!r}. A LoRA delta targets named modules in one architecture.",
            file=sys.stderr,
        )
        ok = False

    print(f"  [meta]   r = {meta.get('r')}, lora_alpha = {meta.get('lora_alpha')}, "
          f"dropout = {meta.get('lora_dropout')}")
    targets = meta.get("target_modules")
    print(f"  [meta]   target_modules = {sorted(targets) if targets else targets}")
    if not targets:
        print("  FAIL: no target_modules; this adapter would apply to nothing", file=sys.stderr)
        ok = False
    print(f"  [meta]   task_type = {meta.get('task_type')}, "
          f"peft_version = {meta.get('peft_version')}")

    weights = [adapter / name for name in WEIGHT_FILES if (adapter / name).is_file()]
    if not weights:
        print(
            f"  FAIL: no {' or '.join(WEIGHT_FILES)} beside the config — the adapter "
            "config is here but its weights are not.",
            file=sys.stderr,
        )
        return False

    for path in weights:
        size_mb = path.stat().st_size / 1e6
        print(f"  [weights] {path.name} = {size_mb:.1f} MB", end="")
        if MIN_WEIGHT_MB <= size_mb <= MAX_WEIGHT_MB:
            print("  (plausible for a LoRA adapter)")
        else:
            print("")
            print(
                f"  FAIL: {size_mb:.1f} MB is outside the {MIN_WEIGHT_MB}-{MAX_WEIGHT_MB} MB "
                "range a LoRA adapter should fall in. A git-lfs pointer stub, or a full "
                "merged checkpoint?",
                file=sys.stderr,
            )
            ok = False

    return ok


def check_registry(adapter: Path) -> bool:
    """Confirm the same adapter arrives through ModelRegistry, not just the YAML.

    Constructing the engine touches no weights, so this is CPU-safe: it proves
    the registry builds a real `GeoChatEngine` (not a `MockModel`) and that the
    adapter name reaches `model_used`, which is the field the execution trace
    records as having answered the question.
    """
    print("\n=== REGISTRY " + "=" * 51)
    from app.core.model_registry import DEFAULT_LOADERS, ModelRegistry
    from app.models.geochat import GeoChatEngine

    registry = ModelRegistry(BACKEND / "model_config.yaml")
    for task, loader in DEFAULT_LOADERS.items():
        if task in registry.tasks:
            registry.register_loader(task, loader)

    engine = registry.get_model("vqa_grounding")
    print(f"  [registry] vqa_grounding -> {type(engine).__name__}")
    if not isinstance(engine, GeoChatEngine):
        print(
            f"  FAIL: expected a GeoChatEngine, got {type(engine).__name__} — no loader "
            "registered for vqa_grounding.",
            file=sys.stderr,
        )
        return False

    print(f"  [registry] loaded = {engine.loaded} (constructing touches no weights)")
    resolved = engine.config.resolved_adapter_path
    print(f"  [registry] resolved adapter = {resolved}")
    if resolved != adapter:
        print(
            f"  FAIL: registry resolved {resolved}, this script resolved {adapter}",
            file=sys.stderr,
        )
        return False

    print(f"  [registry] model_used = {engine.model_used!r}")
    if adapter.name not in engine.model_used:
        print(
            f"  FAIL: {adapter.name!r} is missing from model_used, so traces would not "
            "record that the fine-tune answered.",
            file=sys.stderr,
        )
        return False

    print("  OK: the registry hands back an engine wired to this adapter.")
    return True


def main() -> int:
    import yaml

    raw = yaml.safe_load((BACKEND / "model_config.yaml").read_text())["vqa_grounding"]
    config = GeoChatConfig.from_mapping(raw)

    ok, adapter = check_config(config)
    if not ok or adapter is None:
        print("\nFAIL: config", file=sys.stderr)
        return 1

    results = {
        "adapter": check_adapter(adapter, config.base_model),
        "registry": check_registry(adapter),
    }

    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(f"\nFAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        f"\nOK: {adapter.name} is wired to {config.base_model} and reaches the engine "
        "through the registry. Weights themselves are untested here — that needs a GPU."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
