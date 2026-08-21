"""
main.py

FastAPI application entry point: wires up CORS and the API routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import aggregates, detections, frames, ingest, occupancy, summary, visits, zone_geometry

app = FastAPI(title="RetailVision server")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(detections.router)
app.include_router(occupancy.router)
app.include_router(aggregates.router)
app.include_router(frames.router)
app.include_router(summary.router)
app.include_router(visits.router)
app.include_router(zone_geometry.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}
