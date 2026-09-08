from fastapi import APIRouter

from app import __version__
from app.core.model_registry import get_registry
from app.db import mongo
from app.models.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mongo_ok = await mongo.ping()
    try:
        tasks = get_registry().tasks
    except Exception:  # noqa: BLE001 - a bad config must not hide the probe
        tasks = []
    return HealthResponse(
        status="ok" if mongo_ok and tasks else "degraded",
        version=__version__,
        mongo=mongo_ok,
        tasks=tasks,
    )
