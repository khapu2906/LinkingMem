"""Plugin auth — Bearer token validation.

Behaviour:
  - PLUGIN_AUTH_TOKEN not set  → auth disabled (all requests pass through).
    Safe for unix socket deployments where the socket permission is the boundary.
  - PLUGIN_AUTH_TOKEN set      → every request must carry
    "Authorization: Bearer <token>", or the server returns 401.
"""

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
