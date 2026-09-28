#!/usr/bin/env python3
"""Sign-in on the protected routes: fail closed, and every user sees only their own.

- No token, or a token that does not verify, is a 401 before the route runs.
- A server with no Firebase configuration refuses protected routes (503)
  rather than letting them through; /health stays public.
- A verified caller's uid is stamped on the history and trace records, the
  history endpoint returns only that uid's records, and another user's trace
  or upload reads exactly like a missing one.

Same rules as the suites beside it: no MongoDB (an in-memory stand-in), no
network, no GPU. Token verification is replaced through
`app.dependency_overrides[get_verifier]` — there is no bypass flag in the app.

    python backend/tests/test_auth.py
    pytest backend/tests
"""

from __future__ import annotations

import asyncio
import functools
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Keyword classification only, and uploads in a scratch directory.
os.environ["SATQUERY_OPENROUTER_API_KEY"] = ""
os.environ["SATQUERY_UPLOAD_DIR"] = tempfile.mkdtemp(prefix="satquery-auth-")
for name in (
    "SATQUERY_FIREBASE_PROJECT_ID",
    "SATQUERY_FIREBASE_SERVICE_ACCOUNT_PATH",
    "SATQUERY_FIREBASE_SERVICE_ACCOUNT_JSON",
):
    os.environ.pop(name, None)

from fastapi.testclient import TestClient  # noqa: E402
from firebase_admin import auth as fb_auth  # noqa: E402

from app.core.auth import get_verifier  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.db import mongo  # noqa: E402

get_settings.cache_clear()

from app.main import app  # noqa: E402

#: Tokens the fake verifier accepts, and the uid each one signs in as.
TOKENS = {"token-alice": "alice", "token-bob": "bob", "token-guest": "guest-123"}


def fake_verifier() -> Any:
    def verify(token: str) -> dict[str, Any]:
        if token not in TOKENS:
            raise fb_auth.InvalidIdTokenError("token does not verify")
        uid = TOKENS[token]
        provider = "anonymous" if uid.startswith("guest") else "password"
        return {"uid": uid, "firebase": {"sign_in_provider": provider}}

    return verify


# --------------------------------------------------------------------------
# In-memory Mongo, just wide enough for the code paths under test
# --------------------------------------------------------------------------
def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    return all(doc.get(k) == v for k, v in query.items())


class _Cursor:
    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def sort(self, key: str, direction: int) -> "_Cursor":
        self._docs.sort(key=lambda d: d.get(key) or "", reverse=direction < 0)
        return self

    def limit(self, n: int) -> "_Cursor":
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc

        return gen()


class _Collection:
    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    async def insert_one(self, doc: dict[str, Any]) -> None:
        self.docs.append(dict(doc))

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> None:
        for doc in self.docs:
            if _matches(doc, query):
                for key, value in update.get("$push", {}).items():
                    doc.setdefault(key, []).append(value)
                return

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next((dict(d) for d in self.docs if _matches(d, query)), None)

    def find(self, query: dict[str, Any] | None = None) -> _Cursor:
        return _Cursor([dict(d) for d in self.docs if _matches(d, query or {})])

    async def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Database(dict):
    def __missing__(self, name: str) -> _Collection:
        self[name] = _Collection()
        return self[name]


def isolated(fn):
    """Fresh fake database, fake verifier, and no leftover overrides."""

    @functools.wraps(fn)
    def wrapper() -> None:
        saved_db, saved_available = mongo._db, mongo._available
        mongo._db = _Database()  # type: ignore[assignment]
        mongo._available = True
        app.dependency_overrides[get_verifier] = fake_verifier
        try:
            fn()
        finally:
            app.dependency_overrides.clear()
            mongo._db, mongo._available = saved_db, saved_available

    return wrapper


def client() -> TestClient:
    # Not used as a context manager: that would run the lifespan, which
    # connects to a real MongoDB.
    return TestClient(app)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def ask(token: str, query: str = "Describe this scene.", upload_id: str | None = None):
    return client().post(
        "/api/query", json={"query": query, "upload_id": upload_id}, headers=bearer(token)
    )


PROTECTED = [
    ("post", "/api/query", {"json": {"query": "hello"}}),
    ("post", "/api/query/stream", {"json": {"query": "hello"}}),
    ("get", "/api/query/history", {}),
    ("get", "/api/query/abc/trace", {}),
    ("post", "/api/upload", {"data": {"mode": "single"}, "files": {"files": ("a.tif", b"x")}}),
]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@isolated
def test_missing_token_is_401_on_every_protected_route():
    for method, path, kwargs in PROTECTED:
        response = getattr(client(), method)(path, **kwargs)
        assert response.status_code == 401, (path, response.status_code, response.text)
        assert response.headers.get("www-authenticate") == "Bearer", path


@isolated
def test_invalid_token_is_401_on_every_protected_route():
    for method, path, kwargs in PROTECTED:
        response = getattr(client(), method)(path, headers=bearer("forged"), **kwargs)
        assert response.status_code == 401, (path, response.status_code, response.text)


@isolated
def test_a_non_bearer_scheme_is_401():
    response = client().get("/api/query/history", headers={"Authorization": "Basic abc"})
    assert response.status_code == 401, response.text


@isolated
def test_health_stays_public():
    response = client().get("/api/health")
    assert response.status_code == 200, response.text


def test_unconfigured_server_refuses_protected_routes():
    """Fail closed: no project id and no service account is a 503, never a pass."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        firebase_project_id=None,
        firebase_service_account_path=None,
        firebase_service_account_json=None,
    )
    try:
        for method, path, kwargs in PROTECTED:
            response = getattr(client(), method)(path, headers=bearer("token-alice"), **kwargs)
            assert response.status_code == 503, (path, response.status_code, response.text)
        assert client().get("/api/health").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_real_verifier_rejects_a_malformed_token_offline():
    """The firebase-admin path itself, project-id only: a garbage token is 401.

    A token that does not even parse fails before any certificate fetch, so
    this needs no network.
    """
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, firebase_project_id="satquery-test"
    )
    try:
        response = client().get("/api/query/history", headers=bearer("not.a.jwt"))
        assert response.status_code == 401, response.text
    finally:
        app.dependency_overrides.clear()


@isolated
def test_valid_token_runs_the_query_and_stores_the_uid():
    response = ask("token-alice")
    assert response.status_code == 200, response.text
    body = response.json()

    history = mongo.get_db()[mongo.QUERY_HISTORY].docs
    traces = mongo.get_db()[mongo.EXECUTION_TRACES].docs
    assert [h["uid"] for h in history] == ["alice"], history
    assert [t["uid"] for t in traces] == ["alice"], traces
    assert history[0]["_id"] == body["query_id"]
    # The stored trace carries the answer, so the run can be reopened later.
    assert traces[0]["answer"] == body["answer"]


@isolated
def test_streaming_stores_the_uid_too():
    response = client().post(
        "/api/query/stream", json={"query": "Describe this scene."}, headers=bearer("token-guest")
    )
    assert response.status_code == 200, response.text
    assert "event: result" in response.text
    history = mongo.get_db()[mongo.QUERY_HISTORY].docs
    assert [h["uid"] for h in history] == ["guest-123"], history


@isolated
def test_history_is_isolated_between_users():
    for _ in range(2):
        assert ask("token-alice", "What land cover is this?").status_code == 200
    assert ask("token-bob", "Is there a river?").status_code == 200
    # A record from before sign-in existed: no uid, so it belongs to nobody.
    asyncio.run(
        mongo.get_db()[mongo.QUERY_HISTORY].insert_one(
            {"_id": "legacy", "query": "old", "timestamp": "2026-01-01T00:00:00Z"}
        )
    )

    alice = client().get("/api/query/history", headers=bearer("token-alice")).json()
    bob = client().get("/api/query/history", headers=bearer("token-bob")).json()

    assert len(alice) == 2 and {h["uid"] for h in alice} == {"alice"}, alice
    assert len(bob) == 1 and bob[0]["query"] == "Is there a river?", bob
    assert "legacy" not in {h["_id"] for h in alice + bob}


@isolated
def test_trace_is_owner_only():
    query_id = ask("token-alice").json()["query_id"]

    mine = client().get(f"/api/query/{query_id}/trace", headers=bearer("token-alice"))
    theirs = client().get(f"/api/query/{query_id}/trace", headers=bearer("token-bob"))

    assert mine.status_code == 200, mine.text
    assert mine.json()["uid"] == "alice"
    assert theirs.status_code == 404, theirs.text


@isolated
def test_another_users_upload_reads_as_missing():
    png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    uploaded = client().post(
        "/api/upload",
        data={"mode": "single", "benchmark_mode": "true"},
        files={"files": ("scene.png", png, "image/png")},
        headers=bearer("token-alice"),
    )
    assert uploaded.status_code == 201, uploaded.text
    upload_id = uploaded.json()["upload_id"]
    assert "uid" not in uploaded.json(), "the owner is stored, not echoed"

    # Bob's run stops at intake with the same message a missing upload gets.
    bob = ask("token-bob", upload_id=upload_id).json()
    assert bob["status"] == "rejected", bob
    assert "was not found" in (bob["error"] or ""), bob

    # Alice's own intake resolves it. Checked at the node so the test never
    # reaches the VQA model.
    from app.agent.nodes.intake import intake

    result = asyncio.run(intake({"upload_id": upload_id, "uid": "alice"}))
    assert result.get("status") is None, result
    assert len(result["assets"]) == 1


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a runner reports, it does not raise
            failures.append(name)
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
