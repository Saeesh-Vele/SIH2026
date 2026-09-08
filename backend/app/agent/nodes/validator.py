"""input_validator — check the bound imagery can answer the classified intent."""

from __future__ import annotations

from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.models.schemas import (
    INTENT_TO_MODES,
    Intent,
    QueryStatus,
    TraceStepStatus,
    UploadMode,
)

#: What each mode is called in a sentence addressed to the person asking.
_MODE_PHRASE: dict[UploadMode, str] = {
    UploadMode.SINGLE: "a single image",
    UploadMode.CROSS_MODAL: "an optical image paired with a SAR image",
    UploadMode.BI_TEMPORAL: "two captures of the same place at different times",
}

_INTENT_PHRASE: dict[Intent, str] = {
    Intent.SINGLE_IMAGE_VQA: "Answering a question about one scene",
    Intent.SINGLE_IMAGE_CAPTIONING: "Describing one scene",
    Intent.CHANGE_VQA: "Comparing a place over time",
    Intent.OPTICAL_SAR_FUSION: "Combining optical and SAR data",
}


def _image_size(path: str | None) -> tuple[int, int] | None:
    """Pixel dimensions, or None when they cannot be read here.

    Pillow is an inference dependency, not a service one, and a GeoTIFF may not
    open at all. Both are ordinary on a machine running only the API, so an
    unreadable size means "cannot check", never "invalid".
    """
    if not path:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return (int(image.width), int(image.height))
    except Exception:  # noqa: BLE001 - any failure means the check is unavailable
        return None


def _distinct_files(assets: list[dict[str, Any]]) -> bool:
    """Whether the bound assets are actually different files."""
    paths = [a.get("stored_path") for a in assets]
    return len(set(paths)) == len(paths) and all(paths)


def _reject(message: str, detail: str, started: float) -> dict[str, Any]:
    return {
        "status": QueryStatus.REJECTED,
        "error": message,
        "steps": [step("Validate inputs", TraceStepStatus.FAILED, detail, elapsed_ms(started))],
    }


async def input_validator(state: GraphState) -> dict[str, Any]:
    started = time.perf_counter()
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return {}

    intent = state.get("intent")
    if intent is None:
        return _reject("The query could not be classified.", "no intent on state", started)

    assets: list[dict[str, Any]] = state.get("assets", [])
    mode = state.get("upload_mode")
    accepted = INTENT_TO_MODES[intent]

    if not assets:
        return _reject(
            f"{_INTENT_PHRASE[intent]} needs {_MODE_PHRASE[accepted[0]]}. Bind a scene first.",
            "no assets bound",
            started,
        )

    if mode not in accepted:
        wanted = " or ".join(_MODE_PHRASE[m] for m in accepted)
        have = _MODE_PHRASE.get(mode, "an unrecognised upload") if mode else "nothing"
        return _reject(
            f"{_INTENT_PHRASE[intent]} needs {wanted}, but the bound scene is {have}. "
            f"Re-upload in {accepted[0].value.replace('_', '-')} mode, or ask a different question.",
            f"intent {intent.value} rejects mode {mode.value if mode else 'none'}",
            started,
        )

    # Beyond the mode, check the assets themselves carry what the task reads.
    if intent is Intent.CHANGE_VQA:
        if len(assets) != 2:
            return _reject(
                f"Comparing a place over time needs exactly 2 captures; {len(assets)} were bound.",
                f"change_vqa got {len(assets)} assets",
                started,
            )
        stamps = [a.get("acquired") for a in assets]
        if all(stamps) and stamps[0] == stamps[1]:
            return _reject(
                "Both captures carry the same acquisition time, so there is no interval "
                "to compare. Bind captures from different dates.",
                f"identical timestamps: {stamps[0]}",
                started,
            )
        roles = {a.get("role") for a in assets}
        if roles != {"t0", "t1"}:
            return _reject(
                "The two captures are not marked as earlier and later. Re-upload in "
                "bi-temporal mode so the order is known.",
                f"expected roles t0/t1, got {sorted(roles)}",
                started,
            )
        if not _distinct_files(assets):
            return _reject(
                "Both slots point at the same file, so there is nothing to compare. "
                "Bind two different captures.",
                "t0 and t1 resolve to one stored path",
                started,
            )

        # Change detection differences the two frames patch by patch, so they
        # have to be the same size for a patch to mean the same place in both.
        sizes = [_image_size(a.get("stored_path")) for a in assets]
        if all(sizes) and sizes[0] != sizes[1]:
            return _reject(
                f"The two captures are different sizes ({sizes[0][0]}x{sizes[0][1]} and "
                f"{sizes[1][0]}x{sizes[1][1]}), so they are not co-registered. Bind two "
                "captures of the same footprint at the same resolution.",
                f"size mismatch {sizes[0]} vs {sizes[1]}",
                started,
            )

    elif intent is Intent.OPTICAL_SAR_FUSION:
        if len(assets) != 2:
            return _reject(
                f"Combining optical and SAR needs exactly 2 images; {len(assets)} were bound.",
                f"optical_sar_fusion got {len(assets)} assets",
                started,
            )
        roles = {a.get("role") for a in assets}
        if roles != {"optical", "sar"}:
            return _reject(
                "Combining optical and SAR needs one image of each. Re-upload in "
                "optical + SAR mode.",
                f"expected roles optical/sar, got {sorted(roles)}",
                started,
            )
        if not _distinct_files(assets):
            return _reject(
                "Both slots point at the same file, so there is only one modality here. "
                "Bind the optical scene and its SAR capture.",
                "optical and sar resolve to one stored path",
                started,
            )

        # Fusion compares the two embeddings position by position, which only
        # means anything if the captures cover the same ground.
        optical = next(a for a in assets if a.get("role") == "optical")
        sar = next(a for a in assets if a.get("role") == "sar")
        sizes = (_image_size(optical.get("stored_path")), _image_size(sar.get("stored_path")))
        if all(sizes) and sizes[0] != sizes[1]:
            return _reject(
                f"The optical and SAR images are different sizes ({sizes[0][0]}x{sizes[0][1]} "
                f"and {sizes[1][0]}x{sizes[1][1]}), so they are not co-registered. Bind a "
                "pair covering the same footprint at the same resolution.",
                f"size mismatch optical {sizes[0]} vs sar {sizes[1]}",
                started,
            )

    elif len(assets) != 1:
        return _reject(
            f"A question about one scene needs a single image; {len(assets)} were bound.",
            f"{intent.value} got {len(assets)} assets",
            started,
        )

    return {
        "steps": [
            step(
                "Validate inputs",
                TraceStepStatus.COMPLETE,
                f"{len(assets)} asset(s) match {intent.value}",
                elapsed_ms(started),
            )
        ]
    }
