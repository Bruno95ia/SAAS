from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

logger = logging.getLogger(__name__)


settings = get_settings()


class Base(DeclarativeBase):
    pass


def _create_engine() -> Engine:
    engine = create_engine(settings.database.url, echo=settings.debug, future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-redef]
        # compatibility placeholder for sqlite - not used but keeps alembic happy
        pass

    return engine


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_database() -> None:
    logger.info("Ensuring database schema")
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
