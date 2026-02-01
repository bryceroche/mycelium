"""Mycelium Data Layer - SQLite database access."""

from mycelium.data_layer.connection import (
    ConnectionManager,
    configure_connection,
    create_connection_manager,
    get_db,
    reset_db,
    retry_on_locked,
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
    "create_connection_manager",
    "ConnectionManager",
    "retry_on_locked",
    "configure_connection",
    "EMBEDDING_DIM",
    "SQLITE_SCHEMA",
    "STEP_SCHEMA",
    "get_schema",
    "init_db",
]
