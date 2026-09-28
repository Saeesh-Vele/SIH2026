#!/usr/bin/env python3
"""Record real answers for the sample gallery and the landing page, on a GPU.

Runs every "ready" sample in frontend/public/samples/manifest.json through the
real controller graph — same code path as /api/query, minus HTTP and sign-in —
and writes what came back. Nothing is edited by hand, so the landing page's
"Ask it anything" section only ever shows a genuine output.

In Colab (Runtime -> Change runtime type -> GPU), one cell:

    !git clone -b feat/auth-ui-redesign https://github.com/Saeesh-Vele/SIH2026.git
    %cd SIH2026
    !pip install -q -r backend/requirements.txt -r scripts/requirements.txt
    !python scripts/record_examples.py --landing eurosat-river

Then download recorded/ and frontend/src/content/landing-example.json, or
commit them from Colab.

Writes:
  recorded/<sample-id>.json                  one per sample, landing format
  frontend/src/content/landing-example.json  the --landing sample, only if its
                                             run ended with status "ok"

A degraded (CPU-fallback) run is recorded as degraded; the landing page shows
its Degraded badge. Pass --require-gpu to refuse to record one at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
PUBLIC = ROOT / "frontend" / "public"
sys.path.insert(0, str(BACKEND))


def stage_upload(sample: dict, upload_dir: Path) -> str:
    """Copy the sample into the upload store exactly as /upload would lay it out."""
    from app.agent.nodes.intake import MANIFEST_NAME
    from app.core.imagery import raster_metadata

    upload_id = uuid.uuid4().hex
    dest = upload_dir / upload_id
    dest.mkdir(parents=True)
    assets = []
    for entry in sample["files"]:
        source = PUBLIC / entry["path"].lstrip("/")
        asset_id = uuid.uuid4().hex
        stored = dest / f"{asset_id}{source.suffix.lower()}"
        shutil.copy(source, stored)
        assets.append(
            {
                "asset_id": asset_id,
                "filename": source.name,
                "content_type": None,
                "size_bytes": stored.stat().st_size,
                "role": entry["role"],
                "stored_path": str(stored),
                "geo": raster_metadata(stored).model_dump(),
            }
        )
    benchmark = any(a["filename"].lower().endswith((".png", ".jpg", ".jpeg")) for a in assets)
    manifest = {
        "upload_id": upload_id,
        "mode": sample["mode"],
        "benchmark_mode": benchmark,
        "assets": assets,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Recorded outside sign-in; intake matches this against the run's uid.
        "uid": None,
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return upload_id


async def run(sample: dict, upload_dir: Path) -> dict:
    from app.agent.graph import get_graph, initial_state
    from app.routes.query import _to_response

    upload_id = stage_upload(sample, upload_dir)
    final = await get_graph().ainvoke(initial_state(query=sample["question"], upload_id=upload_id))
    return _to_response(final).model_dump(mode="json")


def to_landing(sample: dict, response: dict) -> dict:
    """The shape src/lib/landing-example.ts reads."""
    specialist = next(
        (s for s in response["steps"] if s["label"].startswith("Run ")), None
    )
    return {
        "status": "recorded",
        "sample_id": sample["id"],
        "question": sample["question"],
        "answer": response["answer"],
        "confidence": response["confidence"],
        "degraded": response["degraded"],
        "task_selected": response["task_selected"],
        "models_used": response["models_used"],
        "trace_line": (
            {
                "label": specialist["label"],
                "detail": specialist["detail"],
                "duration_ms": specialist["duration_ms"],
            }
            if specialist
            else None
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        # Kept for the record; the page ignores these.
        "run_status": response["status"],
        "error": response["error"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--landing", help="sample id to publish as the landing example")
    parser.add_argument("--only", nargs="*", help="record just these sample ids")
    parser.add_argument(
        "--require-gpu", action="store_true", help="exit instead of recording a degraded run"
    )
    args = parser.parse_args()

    import os

    upload_dir = ROOT / "recorded" / "_uploads"
    os.environ["SATQUERY_UPLOAD_DIR"] = str(upload_dir)
    # No database on Colab; the graph notes the trace as unsaved and carries on.
    os.environ.setdefault("SATQUERY_MONGO_TIMEOUT_MS", "200")

    manifest = json.loads((PUBLIC / "samples" / "manifest.json").read_text())
    samples = [s for s in manifest["samples"] if s["status"] == "ready"]
    if args.only:
        samples = [s for s in samples if s["id"] in args.only]

    out_dir = ROOT / "recorded"
    out_dir.mkdir(exist_ok=True)
    try:
        return record(samples, upload_dir, out_dir, args)
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


def record(samples: list[dict], upload_dir: Path, out_dir: Path, args: argparse.Namespace) -> int:
    recorded: dict[str, dict] = {}
    for sample in samples:
        print(f"-> {sample['id']}: {sample['question']}")
        response = asyncio.run(run(sample, upload_dir))
        if args.require_gpu and response["degraded"]:
            print("   degraded (CPU fallback) and --require-gpu is set; stopping.")
            return 1
        example = to_landing(sample, response)
        (out_dir / f"{sample['id']}.json").write_text(json.dumps(example, indent=2) + "\n")
        recorded[sample["id"]] = example
        flag = " [DEGRADED]" if response["degraded"] else ""
        print(f"   {response['status']}{flag}: {response['answer'][:120]}")

    if args.landing:
        example = recorded.get(args.landing)
        if example is None:
            print(f"--landing {args.landing}: no such ready sample was recorded.")
            return 1
        if example["run_status"] != "ok":
            print(f"--landing {args.landing}: run ended {example['run_status']}; not published.")
            return 1
        target = ROOT / "frontend" / "src" / "content" / "landing-example.json"
        public = {k: v for k, v in example.items() if k not in {"run_status", "error"}}
        target.write_text(json.dumps(public, indent=2) + "\n")
        print(f"Landing example written to {target.relative_to(ROOT)}")

    return 0




if __name__ == "__main__":
    raise SystemExit(main())
