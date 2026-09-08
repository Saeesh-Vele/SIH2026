#!/usr/bin/env python3
"""Zero-shot VQA smoke test for the LLaVA-family checkpoint.

Loads any LLaVA-compatible checkpoint and runs a single VQA inference on one
image, printing the answer. Run this before trusting the controller's VQA node
— it exercises the same code, `backend/app/models/geochat.py`, so a pass here
means the graph will run too.

    python scripts/test_geochat_zeroshot.py --image samples/airport.png \
        --question "How many aircraft are visible on the tarmac?"

The default checkpoint is `llava-hf/llava-1.5-7b-hf`; `--model` takes any other
LLaVA-compatible repo id or local path. GeoChat is not required — `--loader
geochat` exists for GeoChat-format checkpoints and needs the official `geochat`
package, which pins a 2023-era dependency stack.

4-bit is the default and needs a CUDA GPU; bitsandbytes has no CPU or MPS 4-bit
kernel. On a machine without one, pass `--quantization none --device cpu`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The engine lives in the backend package so the script and the controller
# cannot drift apart.
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.geochat import (  # noqa: E402
    DEFAULT_MODEL,
    GeoChatConfig,
    GeoChatEngine,
    ModelUnavailable,
)

DEFAULT_QUESTION = "Describe the scene in this satellite image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLaVA-compatible HF repo id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to the sample image")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="VQA question to ask")
    parser.add_argument(
        "--adapter", default=None, help="Optional LoRA adapter path (reported in model_used)"
    )
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="4bit")
    parser.add_argument("--device", default="cuda:0", help="Device map target (default: cuda:0)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2, help="0 for greedy decoding")
    parser.add_argument(
        "--loader",
        choices=("auto", "geochat", "llava"),
        default="auto",
        help=(
            "llava: transformers' generic LLaVA classes (the working path). "
            "geochat: the official geochat package, for GeoChat-format checkpoints. "
            "auto (default): geochat if importable, else llava"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the raw result object")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    config = GeoChatConfig(
        base_model=args.model,
        adapter_path=args.adapter,
        quantization=args.quantization,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    engine = GeoChatEngine(config, loader=args.loader)

    print(f"[load] model={config.base_model} quant={config.quantization} device={config.device}")
    try:
        result = engine.infer(args.image, args.question)
    except ModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"[infer] q: {args.question}")
    print(f"[infer] {result['tokens']} tokens in {result['duration_ms']} ms")
    print("\n--- ANSWER ---")
    print(result["answer"] or "(empty)")
    print(f"\nconfidence (mean token probability): {result['confidence']}")
    if result["evidence"]:
        print(f"grounding boxes: {len(result['evidence'])}")
        for box in result["evidence"]:
            print(f"  {box['label']}: x={box['x']} y={box['y']} w={box['w']} h={box['h']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
