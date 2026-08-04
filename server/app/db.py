"""
db.py

Async SQLAlchemy engine and session factory, plus the get_db()
dependency FastAPI routes use to get a request-scoped database session.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

_settings = get_settings()
engine = create_async_engine(_settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async database session, closing it after the request."""
    async with AsyncSessionLocal() as session:
        yield session
