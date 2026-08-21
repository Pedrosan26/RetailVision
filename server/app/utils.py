"""
utils.py

Small helpers shared across routers.
"""

from datetime import datetime, timezone


def as_utc(value: datetime) -> datetime:
    """Treat a naive datetime as UTC.

    Postgres' TIMESTAMPTZ preserves timezone info across a round trip;
    SQLite (used in tests, see tests/conftest.py) does not, so values
    read back from the test DB come back naive even though everything
    written is already UTC. This normalizes both cases to the same
    aware form so downstream comparisons/arithmetic behave identically
    regardless of which database produced the row.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
