"""Database connection manager - SQLite backend."""

import json
import logging
import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Retry configuration for database locked errors
DB_RETRY_MAX_ATTEMPTS = 5
DB_RETRY_BASE_DELAY = 0.1  # 100ms base delay
DB_RETRY_MAX_DELAY = 2.0   # 2 second max delay

T = TypeVar('T')


def retry_on_locked(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator for database operations that may encounter locking.

    Uses exponential backoff with jitter to handle concurrent write contention.
    Per CLAUDE.md "System Independence": Automated retry without manual intervention.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        last_error = None
        for attempt in range(DB_RETRY_MAX_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) or "database is busy" in str(e):
                    last_error = e
                    if attempt < DB_RETRY_MAX_ATTEMPTS - 1:
                        # Exponential backoff with jitter
                        delay = min(DB_RETRY_BASE_DELAY * (2 ** attempt), DB_RETRY_MAX_DELAY)
                        delay *= (0.5 + random.random())  # Add jitter
                        logger.debug(
                            "[db_retry] Attempt %d/%d failed (locked), retrying in %.2fs",
                            attempt + 1, DB_RETRY_MAX_ATTEMPTS, delay
                        )
                        time.sleep(delay)
                    continue
                raise
        # All retries exhausted
        logger.warning(
            "[db_retry] All %d attempts failed for %s",
            DB_RETRY_MAX_ATTEMPTS, func.__name__
        )
        raise last_error
    return wrapper

# Import config for DB path and embedding dimension - allows branch-specific databases
try:
    from mycelium.config import DB_PATH as CONFIG_DB_PATH, EMBEDDING_DIM
except ImportError:
    CONFIG_DB_PATH = "mycelium.db"
    EMBEDDING_DIM = 3072  # Fallback if config not available

DEFAULT_DB_PATH = os.getenv("MYCELIUM_DB_PATH", CONFIG_DB_PATH)


def configure_connection(conn: sqlite3.Connection, enable_foreign_keys: bool = True) -> None:
    """Configure SQLite connection with consistent PRAGMA settings.

    Ensures WAL mode, performance settings, and timeout are consistent across all DB access.
    Call this immediately after sqlite3.connect().

    Args:
        conn: SQLite connection to configure
        enable_foreign_keys: Whether to enable foreign key constraints (default True)
    """
    conn.row_factory = sqlite3.Row

    # --- Core Settings ---
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 60000")  # 60s timeout for concurrent workers
    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")

    # --- Performance Tuning ---
    # synchronous=NORMAL is safe with WAL and much faster than FULL (default)
    conn.execute("PRAGMA synchronous = NORMAL")

    # Increase page cache to 64MB (negative = KB). Default ~2MB causes excess disk I/O
    conn.execute("PRAGMA cache_size = -65536")

    # Memory-mapped I/O for read-heavy embedding searches (256MB)
    conn.execute("PRAGMA mmap_size = 268435456")

    # Keep temp tables/indexes in RAM for complex queries
    conn.execute("PRAGMA temp_store = MEMORY")


class ConnectionManager:
    """SQLite connection manager with thread-local connections.

    Can be used as:
    - Singleton (default): ConnectionManager() returns the same instance
    - Factory: create_connection_manager(db_path) creates a new instance
    """

    _instance: Optional["ConnectionManager"] = None
    _lock = threading.Lock()

    def __new__(cls, _use_singleton: bool = True, **kwargs) -> "ConnectionManager":
        # Note: **kwargs accepts db_path, enable_foreign_keys for __init__
        if not _use_singleton:
            # Factory mode: create a new instance
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        # Singleton mode
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, _use_singleton: bool = True, db_path: str = None, enable_foreign_keys: bool = True):
        with self._lock:
            if self._initialized:
                return
            self._db_path = db_path or os.getenv("MYCELIUM_DB_PATH", DEFAULT_DB_PATH)
            self._enable_foreign_keys = enable_foreign_keys
            self._local = threading.local()
            self._initialized = True

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def is_postgresql(self) -> bool:
        return False

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False, timeout=30.0
            )
            configure_connection(self._local.conn, enable_foreign_keys=self._enable_foreign_keys)
        return self._local.conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection with automatic commit/rollback and retry on lock."""
        conn = self._get_connection()
        try:
            yield conn
            self._commit_with_retry(conn)
        except Exception:
            conn.rollback()
            raise

    def _commit_with_retry(self, conn: sqlite3.Connection) -> None:
        """Commit with retry logic for locked database.

        Per CLAUDE.md "System Independence": Automated retry without manual intervention.
        """
        last_error = None
        for attempt in range(DB_RETRY_MAX_ATTEMPTS):
            try:
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) or "database is busy" in str(e):
                    last_error = e
                    if attempt < DB_RETRY_MAX_ATTEMPTS - 1:
                        delay = min(DB_RETRY_BASE_DELAY * (2 ** attempt), DB_RETRY_MAX_DELAY)
                        delay *= (0.5 + random.random())
                        logger.debug(
                            "[db_retry] Commit attempt %d/%d failed (locked), retrying in %.2fs",
                            attempt + 1, DB_RETRY_MAX_ATTEMPTS, delay
                        )
                        time.sleep(delay)
                    continue
                raise
        logger.warning("[db_retry] All %d commit attempts failed", DB_RETRY_MAX_ATTEMPTS)
        raise last_error

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self.connection() as conn:
            yield conn

    def q(self, sql: str) -> str:
        return sql

    def pack_vector(self, embedding: Union[list, np.ndarray]) -> str:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return json.dumps(embedding)

    def unpack_vector(self, data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, str):
            return np.array(json.loads(data), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return None

    @retry_on_locked
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor

    @retry_on_locked
    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchone()

    @retry_on_locked
    def fetchall(self, sql: str, params: tuple = ()) -> list:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._local.conn = None

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None


_db: Optional[ConnectionManager] = None


def get_db():
    """Get the database connection manager.

    Always uses SQLite for simplicity. GCP mode only affects LLM/embeddings
    (Vertex AI APIs), not the database.
    """
    global _db
    if _db is None:
        _db = ConnectionManager()
        logger.info(f"[connection] Using SQLite: {_db.db_path}")
    return _db


def reset_db():
    global _db
    ConnectionManager.reset()
    _db = None


def create_connection_manager(db_path: str, enable_foreign_keys: bool = True) -> ConnectionManager:
    """Create a non-singleton ConnectionManager for a specific path.

    Use this when you need a connection to a database other than the default.
    The returned manager has the same interface as get_db() but is independent.

    Per CLAUDE.md "New Favorite Pattern": All DB connections through data layer.

    Args:
        db_path: Path to the SQLite database file.
        enable_foreign_keys: Whether to enable foreign key constraints (default True).
            Set False for caches like embedding_cache.db that don't need FK.

    Returns:
        A ConnectionManager instance for the specified path.
    """
    return ConnectionManager(_use_singleton=False, db_path=db_path, enable_foreign_keys=enable_foreign_keys)
