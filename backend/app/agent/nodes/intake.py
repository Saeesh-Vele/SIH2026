"""intake — resolve the upload reference into concrete assets."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.core.config import get_settings
from app.db import mongo
from app.models.schemas import QueryStatus, TraceStepStatus, UploadMode

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def manifest_path(upload_id: str) -> Path:
    return Path(get_settings().upload_dir) / upload_id / MANIFEST_NAME


def _from_disk(upload_id: str) -> dict[str, Any] | None:
    """Read the manifest /upload writes next to the files.

    Mongo is the record of truth, but the graph must still answer questions
    when it is unreachable — the imagery is on disk either way.
    """
    path = manifest_path(upload_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable manifest at %s", path, exc_info=True)
        return None


async def _from_mongo(upload_id: str) -> dict[str, Any] | None:
    if not mongo.available():
        return None
    try:
        return await mongo.get_db()[mongo.UPLOADS].find_one({"_id": upload_id})
    except Exception:  # noqa: BLE001 - fall through to the on-disk manifest
        mongo.mark_unavailable()
        logger.warning("mongo unavailable for upload lookup", exc_info=True)
        return None


#: Upload ids are uuid4 hex, minted by /upload. Anything else — "../" above
#: all, since the id becomes a directory name — is not an upload.
_UPLOAD_ID = re.compile(r"[0-9a-f]{32}")


async def load_upload_record(upload_id: str) -> dict[str, Any] | None:
    """The stored upload, from Mongo when it is reachable, else the manifest."""
    if not _UPLOAD_ID.fullmatch(upload_id):
        return None
    return await _from_mongo(upload_id) or _from_disk(upload_id)


async def intake(state: GraphState) -> dict[str, Any]:
    started = time.perf_counter()
    upload_id = state.get("upload_id")

    if not upload_id:
        return {
            "assets": [],
            "upload_mode": None,
            "benchmark_mode": False,
            "steps": [
                step(
                    "Intake",
                    TraceStepStatus.COMPLETE,
                    "no upload referenced — the query carries no imagery",
                    elapsed_ms(started),
                )
            ],
        }

    record = await load_upload_record(upload_id)

    # Someone else's upload reads exactly like a missing one, so an upload id
    # alone tells a caller nothing about whether it exists.
    if record is not None and record.get("uid") != state.get("uid"):
        record = None

    if record is None:
        return {
            "assets": [],
            "upload_mode": None,
            "status": QueryStatus.REJECTED,
            "error": f"Upload {upload_id} was not found. Bind the scene again.",
            "steps": [
                step(
                    "Intake",
                    TraceStepStatus.FAILED,
                    f"unknown upload {upload_id}",
                    elapsed_ms(started),
                )
            ],
        }

    assets = list(record.get("assets", []))
    raw_mode = record.get("mode")
    mode = UploadMode(raw_mode) if raw_mode else None

    missing = [a["filename"] for a in assets if not Path(a.get("stored_path", "")).exists()]
    if missing:
        return {
            "assets": assets,
            "upload_mode": mode,
            "status": QueryStatus.REJECTED,
            "error": f"Stored file missing for {', '.join(missing)}. Upload the scene again.",
            "steps": [
                step(
                    "Intake",
                    TraceStepStatus.FAILED,
                    f"{len(missing)} file(s) missing on disk",
                    elapsed_ms(started),
                )
            ],
        }

    roles = ", ".join(a.get("role", "?") for a in assets)
    return {
        "assets": assets,
        "upload_mode": mode,
        "benchmark_mode": bool(record.get("benchmark_mode", False)),
        "steps": [
            step(
                "Intake",
                TraceStepStatus.COMPLETE,
                f"{len(assets)} asset(s) from {mode.value if mode else 'unknown'} upload: {roles}",
                elapsed_ms(started),
            )
        ],
    }
