"""Async MongoDB connection lifecycle (motor)."""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.core.config import get_settings

logger = logging.getLogger(__name__)

QUERY_HISTORY = "query_history"
EXECUTION_TRACES = "execution_traces"
UPLOADS = "uploads"

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

#: Last known reachability. Kept so callers can skip a doomed round trip
#: instead of paying the server-selection timeout on every operation. Set at
#: connect, cleared when an operation fails, and refreshed by ping().
_available: bool = False


def available() -> bool:
    """Whether the last Mongo operation succeeded. Cheap; never does I/O."""
    return _available


def mark_unavailable() -> None:
    """Called by writers when an operation fails, so the next one skips fast."""
    global _available
    _available = False


async def connect() -> None:
    """Open the client and ensure indexes.

    Never blocks startup on an unreachable server: the client is created either
    way and motor reconnects on its own once Mongo appears. Until then reads and
    writes fail per request and /health reports ``mongo: false``.
    """
    global _client, _db
    if _client is not None:
        return

    settings = get_settings()
    _client = AsyncIOMotorClient(
        settings.mongo_uri,
        uuidRepresentation="standard",
        serverSelectionTimeoutMS=settings.mongo_timeout_ms,
        connectTimeoutMS=settings.mongo_timeout_ms,
    )
    _db = _client[settings.mongo_db]

    global _available
    try:
        await _ensure_indexes(_db)
    except Exception:  # noqa: BLE001 - degraded start is better than no start
        _available = False
        logger.warning("mongo unreachable at %s; starting degraded", settings.mongo_uri)
    else:
        _available = True
        logger.info("mongo connected: db=%s", settings.mongo_db)


async def disconnect() -> None:
    global _client, _db, _available
    if _client is not None:
        _client.close()
    _client, _db, _available = None, None, False


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("mongo not connected; call connect() during startup")
    return _db


async def ping() -> bool:
    """Liveness probe used by /health. Also refreshes the cached flag, which is
    how a server that comes up after startup gets picked back up."""
    global _available
    if _client is None:
        _available = False
        return False
    try:
        await _client.admin.command("ping")
        if not _available:
            await _ensure_indexes(get_db())
        _available = True
    except Exception:  # noqa: BLE001 - health check must not raise
        logger.warning("mongo ping failed", exc_info=True)
        _available = False
    return _available


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    await db[QUERY_HISTORY].create_index([("timestamp", DESCENDING)])
    await db[QUERY_HISTORY].create_index([("task_type", ASCENDING)])
    await db[QUERY_HISTORY].create_index([("uid", ASCENDING), ("timestamp", DESCENDING)])
    await db[EXECUTION_TRACES].create_index([("uid", ASCENDING), ("query_id", ASCENDING)])
    await db[EXECUTION_TRACES].create_index([("timestamp", DESCENDING)])
    await db[EXECUTION_TRACES].create_index([("query_id", ASCENDING)])
    await db[EXECUTION_TRACES].create_index([("task_selected", ASCENDING)])
    await db[UPLOADS].create_index([("created_at", DESCENDING)])
