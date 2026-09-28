"""Upload endpoint.

Mirrors the three client-side modes: a single image, an optical+SAR pair, or a
bi-temporal pair. GeoTIFF is the primary format; PNG/JPEG are accepted only when
the caller flags benchmark-dataset mode.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.agent.nodes.intake import MANIFEST_NAME
from app.core.auth import AuthUser, current_user
from app.core.config import get_settings
from app.db import mongo
from app.models.schemas import UploadedAsset, UploadMode, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])

GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".gtiff"}
BENCHMARK_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Which asset roles each mode expects, in order.
MODE_ROLES: dict[UploadMode, tuple[str, ...]] = {
    UploadMode.SINGLE: ("primary",),
    UploadMode.CROSS_MODAL: ("optical", "sar"),
    UploadMode.BI_TEMPORAL: ("t0", "t1"),
}


def _validate_extension(filename: str, benchmark_mode: bool) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix in GEOTIFF_EXTENSIONS:
        return
    if suffix in BENCHMARK_EXTENSIONS:
        if benchmark_mode:
            return
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{filename}: PNG/JPEG are accepted only in benchmark dataset mode",
        )
    raise HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        f"{filename}: expected GeoTIFF (.tif/.tiff)",
    )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload(
    mode: UploadMode = Form(UploadMode.SINGLE),
    benchmark_mode: bool = Form(False),
    files: list[UploadFile] = File(...),
    user: AuthUser = Depends(current_user),
) -> UploadResponse:
    roles = MODE_ROLES[mode]
    if len(files) != len(roles):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"mode {mode.value!r} expects {len(roles)} file(s), got {len(files)}",
        )

    settings = get_settings()
    upload_id = uuid.uuid4().hex
    dest_dir = Path(settings.upload_dir) / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    assets: list[UploadedAsset] = []
    for role, upload_file in zip(roles, files):
        filename = upload_file.filename or f"{role}.tif"
        _validate_extension(filename, benchmark_mode)

        payload = await upload_file.read()
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"{filename}: exceeds {settings.max_upload_bytes} bytes",
            )

        asset_id = uuid.uuid4().hex
        stored_path = dest_dir / f"{asset_id}{Path(filename).suffix.lower()}"
        stored_path.write_bytes(payload)

        assets.append(
            UploadedAsset(
                asset_id=asset_id,
                filename=filename,
                content_type=upload_file.content_type,
                size_bytes=len(payload),
                role=role,
                stored_path=str(stored_path),
            )
        )

    response = UploadResponse(
        upload_id=upload_id,
        mode=mode,
        benchmark_mode=benchmark_mode,
        assets=assets,
        created_at=datetime.now(timezone.utc),
    )

    # The owner is stored with the upload but not echoed back: intake checks it
    # so an upload id is useless to anyone but the person who made it.
    record = {**response.model_dump(mode="json"), "uid": user.uid}

    # The manifest is what the graph reads when Mongo is unreachable: the
    # imagery is on disk either way, so a missing database must not cost us the
    # ability to answer questions about it.
    (dest_dir / MANIFEST_NAME).write_text(json.dumps(record, indent=2))

    if mongo.available():
        try:
            await mongo.get_db()[mongo.UPLOADS].insert_one(
                {"_id": upload_id, **{k: v for k, v in record.items() if k != "upload_id"}}
            )
        except Exception:  # noqa: BLE001 - the manifest is enough to proceed
            mongo.mark_unavailable()
            logger.warning("upload %s not indexed in mongo; manifest written", upload_id)

    return response
