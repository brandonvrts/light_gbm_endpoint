"""v1 auth: simple X-API-Key header (internal LAN trust boundary). Swap for Azure AD
JWT later — the dependency shape stays the same."""
from __future__ import annotations
from fastapi import Header, HTTPException

from core import db

_API_KEY = db.ENV.get("GBME_API_KEY")


def require_key(x_api_key: str | None = Header(default=None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return True
