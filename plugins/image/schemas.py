"""Request/response models for the image plugin."""

from pydantic import BaseModel


class StoreRequest(BaseModel):
    # One of: http/https URL  OR  base64 data-URI (data:image/...;base64,...)
    url: str | None = None
    data: str | None = None  # base64 data-URI


class StoreResponse(BaseModel):
    # Stable URL served by this plugin: http://localhost:8002/images/<hash>.ext
    url: str
    filename: str
    size_bytes: int
    content_type: str


class EmbedRequest(BaseModel):
    image_url: str  # http/https URL or data-URI


class EmbedResponse(BaseModel):
    vector: list[float]
    dim: int
    caption: str
    model: str
