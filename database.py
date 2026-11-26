import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Resolved relative to this file rather than the working directory, so the app
# and the migration script find the same .env no matter where they are run from.
load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger("forenscreen.database")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to backend/.env before starting the server."
    )

# Two deliberate settings here.
#
# pool_pre_ping / pool_recycle: Neon is serverless and closes idle connections
# underneath us, so a pooled connection can be dead by the time it is reused.
#
# connect_timeout: without it, an unreachable host blocks the connect
# indefinitely. That matters because the startup path creates tables, so a
# network blip turns into a process that hangs at import with no output at all -
# much harder to diagnose than a fast, explicit failure.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"connect_timeout": 10},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema():
    """Create any missing tables, tolerating an unreachable database.

    Called from main.py at import time. If the database cannot be reached we
    log and carry on, so the process still starts and can answer /health rather
    than hanging on a socket forever. Routes that need the database will fail
    with a clear error of their own.
    """
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning(
            "Could not create tables; is the database reachable? (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return False
