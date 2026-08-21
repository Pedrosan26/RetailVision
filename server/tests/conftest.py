"""
conftest.py

Shared pytest fixtures for server tests: an in-memory SQLite database and
an httpx AsyncClient wired to the FastAPI app, standing in for the real
Postgres/TimescaleDB used in local development so tests need no external
services running.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.db import get_db
from app.main import app
from app.models.detection import Base

TEST_CAMERA_NODE_ID = "test-node"
TEST_API_KEY = "test-key"

# Several nodes, for the cross-camera cases: deduplication only means anything
# when more than one camera can authenticate and report on the same zone.
TEST_CAMERA_NODES = {"cam-a": "key-a", "cam-b": "key-b", "cam-c": "key-c"}


def _test_settings() -> Settings:
    """Settings override with a known, predictable camera-node API key for tests."""
    nodes = {TEST_CAMERA_NODE_ID: TEST_API_KEY, **TEST_CAMERA_NODES}
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        camera_node_api_keys=",".join(f"{node}:{key}" for node, key in nodes.items()),
    )


@pytest.fixture
async def db_session_factory():
    """A session factory bound to a fresh in-memory SQLite DB, shared across the test via a single connection."""
    # StaticPool keeps the one in-memory SQLite connection alive across
    # sessions -- without it, each new connection would see a separate,
    # empty ":memory:" database instead of the tables created below.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(engine, expire_on_commit=False)

    await engine.dispose()


@pytest.fixture
async def client(db_session_factory):
    """An httpx AsyncClient wired to the FastAPI app, using db_session_factory instead of real Postgres."""

    async def override_get_db():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = _test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
