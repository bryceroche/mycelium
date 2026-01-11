"""Database connection manager - SQLite backend."""

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import config for DB path - allows branch-specific databases
try:
    from mycelium.config import DB_PATH as CONFIG_DB_PATH
except ImportError:
    CONFIG_DB_PATH = "mycelium.db"

EMBEDDING_DIM = 768  # Updated for all-mpnet-base-v2 (was 384 for MiniLM)
DEFAULT_DB_PATH = os.getenv("MYCELIUM_DB_PATH", CONFIG_DB_PATH)


class ConnectionManager:
    """SQLite connection manager with thread-local connections."""

    _instance: Optional["ConnectionManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConnectionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return
            self._db_path = os.getenv("MYCELIUM_DB_PATH", DEFAULT_DB_PATH)
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
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchone()

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


def get_db() -> ConnectionManager:
    global _db
    if _db is None:
        _db = ConnectionManager()
    return _db


def reset_db():
    global _db
    ConnectionManager.reset()
    _db = None
