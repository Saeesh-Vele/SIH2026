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
    Intent.SINGLE_IMAGE_GROUNDING: "Locating an object in one scene",
    Intent.CHANGE_VQA: "Comparing a place over time",
    Intent.OPTICAL_SAR_FUSION: "Combining optical and SAR data",
}


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

    elif intent is Intent.OPTICAL_SAR_FUSION:
        roles = {a.get("role") for a in assets}
        if roles != {"optical", "sar"}:
            return _reject(
                "Combining optical and SAR needs one image of each. Re-upload in "
                "optical + SAR mode.",
                f"expected roles optical/sar, got {sorted(roles)}",
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
