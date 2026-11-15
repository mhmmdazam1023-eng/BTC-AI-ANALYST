"""Async SQLAlchemy engine and session management.

Provides a lazily-initialised, process-wide async engine plus a
``session_scope`` context manager that commits on success and rolls back
on error — the single, consistent way to touch the database.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.async_database_url,
            pool_pre_ping=True,   # detect stale connections before use
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        logger.info(
            "Async database engine initialised (host=%s port=%s db=%s)",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, rollback (and log) on error.

    Usage::

        async with session_scope() as session:
            session.add(row)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to an error")
            raise


async def dispose_engine() -> None:
    """Dispose the engine and its connection pool (call at shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed")
