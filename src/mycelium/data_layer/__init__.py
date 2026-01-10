"""Mycelium Data Layer - SQLite database access."""

from mycelium.data_layer.connection import (
    ConnectionManager,
    get_db,
    reset_db,
    EMBEDDING_DIM,
)
from mycelium.data_layer.schema import (
    SQLITE_SCHEMA,
    STEP_SCHEMA,
    get_schema,
    init_db,
)

db = get_db()

__all__ = [
    "db",
    "get_db",
    "reset_db",
    "ConnectionManager",
    "EMBEDDING_DIM",
    "SQLITE_SCHEMA",
    "STEP_SCHEMA",
    "get_schema",
    "init_db",
]
