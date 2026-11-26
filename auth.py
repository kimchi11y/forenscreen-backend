"""Verify Neon Managed Auth (Better Auth) JWTs.

Tokens are EdDSA (Ed25519) and live for about fifteen minutes. Verification runs
against the branch's JWKS, so the API trusts the auth service's signature rather
than anything the client claims - being "signed in" in a browser buys no access
on its own.

This is also where the chain of custody gets its `actor`. That matters: a
custody record whose "who" is supplied by the caller is not evidence, it is an
assertion. Until a token verifies, entries are recorded as `unauthenticated`.

Configuration:
    NEON_AUTH_BASE_URL   branch Auth URL, including its path
    NEON_AUTH_JWKS_URL   optional; derived from the base URL when absent

Note on issuer/audience: the token's `iss` and `aud` are the bare origin, with
no path - e.g. ``https://ep-xxx.neonauth.c-2.<region>.aws.neon.tech`` - even
though the Auth URL itself ends in ``/neondb/auth``. The defaults below follow
that, and both can be overridden explicitly if the claims ever differ.
"""

import logging
import os
from urllib.parse import urlsplit

import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient

logger = logging.getLogger("forenscreen.auth")

ALGORITHMS = ["EdDSA"]

_jwks_client = None
_jwks_url_cached = None


class AuthNotConfigured(RuntimeError):
    """Raised when a token is presented but no auth service is configured."""


def _base_url() -> str | None:
    return os.getenv("NEON_AUTH_BASE_URL") or None


def _jwks_url() -> str | None:
    explicit = os.getenv("NEON_AUTH_JWKS_URL")
    if explicit:
        return explicit
    base = _base_url()
    return base.rstrip("/") + "/.well-known/jwks.json" if base else None


def _origin() -> str | None:
    base = _base_url()
    if not base:
        return None
    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else None


def _client(jwks_url: str) -> PyJWKClient:
    """One cached client per JWKS URL; it handles key rotation internally."""
    global _jwks_client, _jwks_url_cached
    if _jwks_client is None or _jwks_url_cached != jwks_url:
        _jwks_client = PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=3600)
        _jwks_url_cached = jwks_url
    return _jwks_client


def is_configured() -> bool:
    return bool(_jwks_url())


def verify_token(token: str) -> dict:
    """Verify a bearer token and return its claims.

    Raises `AuthNotConfigured` when there is nothing to verify against, and
    `jwt.InvalidTokenError` (or HTTPException 401 at the dependency level) for a
    token that does not check out.
    """
    jwks_url = _jwks_url()
    if not jwks_url:
        raise AuthNotConfigured(
            "NEON_AUTH_BASE_URL (or NEON_AUTH_JWKS_URL) is not set."
        )

    signing_key = _client(jwks_url).get_signing_key_from_jwt(token)

    options = {"require": ["exp", "sub"], "verify_aud": False, "verify_iss": False}
    kwargs = {}

    origin = _origin()
    issuer = os.getenv("NEON_AUTH_ISSUER") or origin
    if issuer:
        options["verify_iss"] = True
        kwargs["issuer"] = issuer

    # The audience is the origin in the tokens observed from this branch. Only
    # asserted when we have something to compare against, since requiring an
    # audience we cannot name would reject valid tokens.
    audience = os.getenv("NEON_AUTH_AUDIENCE") or origin
    if audience:
        options["verify_aud"] = True
        kwargs["audience"] = audience

    return jwt.decode(
        token,
        signing_key.key,
        algorithms=ALGORITHMS,
        options=options,
        **kwargs,
    )


def actor_from(claims: dict | None) -> str:
    """The string recorded as the custody `actor`.

    Prefers a stable identifier over a display name: an email can change, an
    account id cannot.
    """
    if not claims:
        return "unauthenticated"
    identifier = claims.get("sub") or claims.get("id")
    return f"user:{identifier}" if identifier else "unauthenticated"


def optional_user(request: Request) -> dict | None:
    """Dependency: the verified claims, or None for an anonymous request.

    A request that presents a token and fails verification is rejected rather
    than silently downgraded to anonymous - otherwise a broken or expired token
    would look like a successful anonymous call, and every custody entry would
    quietly lose its actor.
    """
    header = request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        return None

    token = header.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        return verify_token(token)
    except AuthNotConfigured as exc:
        logger.warning("Token presented but auth is not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        ) from exc
    except Exception as exc:
        logger.warning("Rejected bearer token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_user(user: dict | None = Depends(optional_user)) -> dict:
    """Dependency: the verified claims, or 401 when there is no signed-in user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
