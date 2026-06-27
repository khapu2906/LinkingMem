"""Plugin auth — Bearer token validation (same contract as text plugin)."""

import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

PLUGIN_AUTH_TOKEN: str | None = os.getenv("PLUGIN_AUTH_TOKEN")

_bearer = HTTPBearer(auto_error=False)


def verify_auth(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    if not PLUGIN_AUTH_TOKEN:
        return
    if credentials is None or credentials.credentials != PLUGIN_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")
