#!/usr/bin/env python3
"""Check that both fusion checkpoints fit the configured architecture.

Optical-SAR fusion runs two encoders — SSL4EO-S12 pretrains per modality and the
results are not interchangeable, because each patch embedding is built for a
specific band count. This loads each checkpoint into its own instance of the
configured architecture and reports what could not be matched, per modality.
CPU only: nothing is run through the models, only loaded into them.

    python backend/tests/verify_fusion_checkpoint.py

Reading the result. Three kinds of leftover, and they mean different things:

- **unexpected** — in the checkpoint, no slot in the model. A large block of
  these is normal for an MAE checkpoint: the decoder and mask token exist only
  during pretraining, and an encoder-only model has nowhere to put them.
- **missing** — in the model, absent from the checkpoint. A handful is normal
  (timm's classifier head has no counterpart in weights trained without labels).
  Dozens means the checkpoint does not describe this architecture.
- **shape mismatch** — the key exists on both sides at different shapes. Never
  benign: it means the model was built with the wrong dimensions, and every one
  of those tensors would stay randomly initialised. `load_state_dict` raises on
  these even with `strict=False`, so they are filtered out and counted here
  rather than being allowed to abort the report.

`in_chans` is read off each checkpoint's own `patch_embed.proj.weight` rather
than assumed — that is exactly what the two modalities differ on. The last check
is that both pool to the same width, since concatenating them otherwise means
nothing.

Deliberately NOT named test_*: it needs timm and real checkpoints on disk, and
the suites beside it are dependency-free by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.fusion import MODALITIES  # noqa: E402

#: More missing keys than this and the checkpoint is not this architecture.
#: Unexpected keys are not capped — an MAE decoder is legitimately ~100 tensors.
MAX_MISSING = 12


def _state_dict(raw: object) -> dict:
    """Unwrap the checkpoint. Publishers nest the weights under varying keys."""
    if not isinstance(raw, dict):
        raise SystemExit(f"checkpoint is a {type(raw).__name__}, not a dict")
    for key in ("state_dict", "model", "model_state_dict"):
        inner = raw.get(key)
        if isinstance(inner, dict):
            print(f"  [unwrap] weights found under {key!r}")
            return inner
    return raw


def _in_chans(state: dict) -> int | None:
    """Input bands the checkpoint was trained on, from its patch embedding."""
    weight = state.get("patch_embed.proj.weight")
    return int(weight.shape[1]) if weight is not None and weight.ndim == 4 else None


def check(modality: str, encoder_name: str, checkpoint: Path, configured: int) -> tuple[bool, int]:
    """Report one modality. Returns (ok, pooled width)."""
    import timm
    import torch

    print(f"\n=== {modality.upper()} " + "=" * (58 - len(modality)))
    print(f"  [config] checkpoint = {checkpoint}")
    if not checkpoint.exists():
        print(f"  error: checkpoint not found at {checkpoint}", file=sys.stderr)
        return False, 0
    print(f"  [config] size       = {checkpoint.stat().st_size / 1e6:.1f} MB")

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _state_dict(raw)
    print(f"  [load]   checkpoint tensors = {len(state)}")

    args = raw.get("args") if isinstance(raw, dict) else None
    if args is not None:
        for field in ("model", "in_channels", "input_size"):
            if hasattr(args, field):
                print(f"  [load]   trained with {field} = {getattr(args, field)}")
        # The pretraining corpus is what makes these two checkpoints different.
        for field in ("root_s1", "root_s2"):
            if hasattr(args, field):
                print(f"  [load]   pretrained on {Path(str(getattr(args, field))).name}")

    bands = _in_chans(state)
    if bands is None:
        print("  warning: no patch_embed.proj.weight; falling back to the configured value")
        bands = configured
    print(f"  [build]  in_chans = {bands} (from the checkpoint)", end="")
    if bands != configured:
        print(f"  <-- model_config.yaml says {configured}")
    else:
        print("  (matches model_config.yaml)")

    model = timm.create_model(encoder_name, pretrained=False, num_classes=0, in_chans=bands)
    width = int(model.num_features)
    print(f"  [build]  params = {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, "
          f"pooled width = {width}")

    # Shape mismatches abort load_state_dict even with strict=False, so they are
    # separated out first. Counting them beats losing the whole report to one.
    reference = model.state_dict()
    fitting, mismatched = {}, []
    for key, tensor in state.items():
        target = reference.get(key)
        if target is not None and tuple(target.shape) != tuple(tensor.shape):
            mismatched.append((key, tuple(tensor.shape), tuple(target.shape)))
        else:
            fitting[key] = tensor

    result = model.load_state_dict(fitting, strict=False)
    missing = [k for k in result.missing_keys if k not in {m[0] for m in mismatched}]
    unexpected = list(result.unexpected_keys)
    decoder = [k for k in unexpected if k.startswith("decoder") or k == "mask_token"]

    print(f"  [result] missing         = {len(missing)}")
    for key in missing:
        print(f"             - {key}")
    print(f"  [result] unexpected      = {len(unexpected)}"
          + (f"  ({len(decoder)} of them the MAE decoder and mask token)" if decoder else ""))
    print(f"  [result] shape mismatch  = {len(mismatched)}")
    for key, have, want in mismatched:
        print(f"             - {key}: checkpoint {have}, model {want}")
    print(f"  [result] matched         = {len(state) - len(unexpected) - len(mismatched)} of {len(state)}")

    if bands != configured:
        print(
            f"  FAIL: {modality}_in_chans is {configured} but the checkpoint is {bands}. "
            f"Set {modality}_in_chans: {bands} in model_config.yaml.",
            file=sys.stderr,
        )
        return False, width
    if mismatched:
        print(
            f"  FAIL: {len(mismatched)} tensor(s) differ in shape and would stay randomly "
            "initialised.",
            file=sys.stderr,
        )
        return False, width
    if len(missing) > MAX_MISSING:
        print(
            f"  FAIL: {len(missing)} missing keys, over the {MAX_MISSING} expected for a "
            f"head-only gap. {encoder_name!r} probably does not describe these weights.",
            file=sys.stderr,
        )
        return False, width

    print(f"  OK: {encoder_name!r} at in_chans={bands} fits this checkpoint.")
    return True, width


def main() -> int:
    import yaml

    from app.models.fusion import FusionConfig

    raw = yaml.safe_load((BACKEND / "model_config.yaml").read_text())["optical_sar_fusion"]
    config = FusionConfig.from_mapping(raw)
    print(f"[config] encoder = {config.encoder} (one architecture, two sets of weights)")

    results, widths = {}, {}
    for modality in MODALITIES:
        checkpoint, configured = config.checkpoint_for(modality)
        if not checkpoint:
            print(f"\nerror: no {modality}_checkpoint_path in model_config.yaml", file=sys.stderr)
            results[modality] = False
            continue
        ok, width = check(modality, config.encoder, (BACKEND / checkpoint).resolve(), configured)
        results[modality] = ok
        widths[modality] = width

    print("\n=== FUSION " + "=" * 53)
    distinct = set(widths.values())
    if len(distinct) == 1 and widths:
        width = distinct.pop()
        print(f"  both encoders pool to {width}d; concat gives {width * 2}d")
    else:
        print(f"  FAIL: encoders pool to different widths {widths}; concat is meaningless",
              file=sys.stderr)
        results["fusion"] = False

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\nFAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nOK: both checkpoints fit, and their outputs are fusable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
