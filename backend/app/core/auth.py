"""Firebase ID-token verification for the protected routes.

`current_user` is the FastAPI dependency: it reads `Authorization: Bearer
<idToken>`, verifies it with firebase-admin, and hands the route an AuthUser.
It fails closed — a missing header, a bad token, or a server with no Firebase
configuration all end the request before the route body runs.

Verification is split out into `get_verifier` so tests can swap the one piece
that talks to Google while still exercising the header handling and the error
mapping for real. There is no bypass switch; tests use
`app.dependency_overrides[get_verifier]`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials
from firebase_admin.exceptions import FirebaseError
from google.auth.credentials import AnonymousCredentials

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Takes a raw ID token, returns its decoded claims, raises on anything invalid.
Verifier = Callable[[str], dict[str, Any]]

_bearer = HTTPBearer(auto_error=False)

_CHALLENGE = {"WWW-Authenticate": "Bearer"}


@dataclass(frozen=True)
class AuthUser:
    uid: str
    #: Signed in with "Continue as guest" (Firebase anonymous auth).
    anonymous: bool
    email: str | None


class AuthNotConfigured(RuntimeError):
    """No Firebase project id or service account is set on this server."""


@lru_cache
def _firebase_app(project_id: str | None, sa_path: str | None, sa_json: str | None):
    """One firebase-admin app per distinct configuration.

    Keyed on the settings values rather than cached once, so a test that swaps
    settings gets a matching app instead of whichever was built first.
    """
    if sa_json:
        credential: Any = credentials.Certificate(json.loads(sa_json))
    elif sa_path:
        credential = credentials.Certificate(sa_path)
    elif project_id:
        # Verifying an ID token needs only Google's public certificates. Left
        # to itself firebase-admin would go looking for Application Default
        # Credentials and fail on any machine without them.
        credential = _AnonymousCredential()
    else:
        raise AuthNotConfigured(
            "set SATQUERY_FIREBASE_PROJECT_ID (or a service account) to verify sign-ins"
        )

    project = project_id or getattr(credential, "project_id", None)
    if not project:
        raise AuthNotConfigured("the service account does not name a project")

    name = f"satquery-{project}-{'sa' if (sa_json or sa_path) else 'anon'}"
    return firebase_admin.initialize_app(credential, {"projectId": project}, name=name)


class _AnonymousCredential(credentials.Base):
    """A firebase-admin credential that carries no secret."""

    def get_credential(self):
        return AnonymousCredentials()


def get_verifier(settings: Settings = Depends(get_settings)) -> Verifier:
    """The token verifier for this server's Firebase project.

    Raises 503 rather than letting the request through when nothing is
    configured: a server that cannot check sign-ins must not act as if it had.
    """
    try:
        app = _firebase_app(
            settings.firebase_project_id,
            str(settings.firebase_service_account_path)
            if settings.firebase_service_account_path
            else None,
            settings.firebase_service_account_json,
        )
    except AuthNotConfigured as exc:
        logger.error("sign-in verification unavailable: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Sign-in verification is not configured on this server.",
        ) from exc
    except (OSError, ValueError) as exc:
        logger.error("could not load the Firebase service account: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Sign-in verification is misconfigured on this server.",
        ) from exc

    return lambda token: auth.verify_id_token(token, app=app)


async def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    verify: Verifier = Depends(get_verifier),
) -> AuthUser:
    """The signed-in caller, or 401."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue.", _CHALLENGE)

    try:
        # verify_id_token may fetch Google's certificates over the network.
        claims = await run_in_threadpool(verify, credentials.credentials)
    except auth.CertificateFetchError as exc:
        logger.warning("could not fetch Firebase certificates: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not reach Google to check your sign-in. Try again shortly.",
        ) from exc
    except auth.ExpiredIdTokenError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Your session expired. Sign in again.", _CHALLENGE
        ) from exc
    except (FirebaseError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Your sign-in could not be verified.", _CHALLENGE
        ) from exc

    uid = claims.get("uid") or claims.get("sub")
    if not uid:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Your sign-in could not be verified.", _CHALLENGE
        )

    provider = (claims.get("firebase") or {}).get("sign_in_provider")
    return AuthUser(uid=uid, anonymous=provider == "anonymous", email=claims.get("email"))
