from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import pytest

from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore


@pytest.fixture
def qdrant_store() -> Callable[..., QdrantVectorStore]:
    """管理 QdrantVectorStore 生命周期，测试结束无论成败都释放 Local 文件锁"""
    stores: list[QdrantVectorStore] = []

    def _make(path: Path, **kwargs: object) -> QdrantVectorStore:
        store = QdrantVectorStore(path, **kwargs)  # type: ignore[arg-type]
        stores.append(store)
        return store

    yield _make

    for store in stores:
        try:
            store.close()
        except Exception:
            pass


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port  # socket released; daemon can bind to this port


@pytest.fixture
async def running_daemon(free_port: int) -> AsyncGenerator[subprocess.Popen[bytes], None]:
    env = os.environ.copy()
    env["KAMA_PORT"] = str(free_port)
    env["KAMA_LOG_FILE"] = ""
    env["KAMA_LOG_LEVEL"] = "WARNING"

    proc = subprocess.Popen([sys.executable, "-m", "kama_claude.core"], env=env)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            writer.close()
            await writer.wait_closed()
            break
        except (ConnectionRefusedError, OSError):
            pass
    else:
        proc.terminate()
        proc.wait()
        pytest.fail("Daemon did not start within 3 seconds")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
