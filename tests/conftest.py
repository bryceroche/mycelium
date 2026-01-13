"""Pytest configuration for mycelium tests."""

import os
import pytest
import tempfile

from mycelium.data_layer.connection import reset_db


@pytest.fixture(scope="function", autouse=True)
def clean_test_db():
    """Use a fresh temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["MYCELIUM_DB_PATH"] = path
    reset_db()
    yield
    reset_db()
    try:
        os.unlink(path)
    except Exception:
        pass
