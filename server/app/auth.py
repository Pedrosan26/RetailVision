"""
auth.py

Minimal camera-node authentication: a shared API key per node, checked
against a static config map. Deliberately no user accounts, sessions, or
per-endpoint roles -- a prototype-scope decision, not an oversight;
revisit before exposing this server beyond a trusted local network.
"""

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings
from .schemas.detection import IngestRequest


def verify_camera_node_api_key(
    body: IngestRequest,
    x_api_key: str | None = Header(None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise 401 unless x_api_key is present and matches the configured key for body.camera_node_id."""
    expected = settings.camera_node_api_key_map().get(body.camera_node_id)
    if expected is None or x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key for camera node")
