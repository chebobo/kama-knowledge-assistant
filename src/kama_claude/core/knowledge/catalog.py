from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.knowledge.models import KnowledgeDocument


class SqliteDocumentCatalog:
    # 初始化 SQLite 文档目录并创建必要表结构
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    # 创建文档表和 scope 查询索引
    def _initialize(self) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK (scope IN ('global', 'project')),
                    project_id TEXT,
                    user_id TEXT,
                    mime_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT,
                    source_mtime REAL,
                    indexed_at TEXT,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_scope "
                "ON documents(user_id, scope, project_id)"
            )
            connection.commit()

    # 保存或更新一条文档目录记录
    async def upsert(self, document: KnowledgeDocument) -> None:
        row = self._to_row(document)
        async with self._lock:
            await asyncio.to_thread(self._upsert_sync, row)

    # 按文档 ID 读取目录记录
    async def get(self, document_id: str) -> KnowledgeDocument | None:
        async with self._lock:
            row = await asyncio.to_thread(self._get_sync, document_id)
        return self._from_row(row) if row is not None else None

    # 返回全部文档目录记录，供管理和测试使用
    async def list_documents(self) -> list[KnowledgeDocument]:
        async with self._lock:
            rows = await asyncio.to_thread(self._list_sync)
        return [self._from_row(row) for row in rows]

    # 删除指定文档的目录记录
    async def delete(self, document_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, document_id)

    # 同步执行文档 upsert，供 worker thread 调用
    def _upsert_sync(self, row: tuple[object, ...]) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source, title, scope, project_id, user_id,
                    mime_type, metadata_json, content_hash, source_mtime,
                    indexed_at, status, chunk_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source=excluded.source,
                    title=excluded.title,
                    scope=excluded.scope,
                    project_id=excluded.project_id,
                    user_id=excluded.user_id,
                    mime_type=excluded.mime_type,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash,
                    source_mtime=excluded.source_mtime,
                    indexed_at=excluded.indexed_at,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    error_message=excluded.error_message
                """,
                row,
            )
            connection.commit()

    # 同步读取一条文档记录
    def _get_sync(self, document_id: str) -> tuple[object, ...] | None:
        with sqlite3.connect(self._path) as connection:
            cursor = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            )
            row = cursor.fetchone()
            return tuple(row) if row is not None else None

    # 同步读取全部文档记录
    def _list_sync(self) -> list[tuple[object, ...]]:
        with sqlite3.connect(self._path) as connection:
            cursor = connection.execute("SELECT * FROM documents ORDER BY document_id")
            return cursor.fetchall()

    # 同步删除一条文档记录
    def _delete_sync(self, document_id: str) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            connection.commit()

    # 将 Document 转换成 SQLite 行
    @staticmethod
    def _to_row(document: KnowledgeDocument) -> tuple[object, ...]:
        indexed_at = (
            document.indexed_at.astimezone(UTC).isoformat()
            if document.indexed_at
            else None
        )
        return (
            document.id,
            document.source,
            document.title,
            document.scope,
            document.project_id,
            document.user_id,
            document.mime_type,
            json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
            document.content_hash,
            document.source_mtime,
            indexed_at,
            document.status,
            document.chunk_count,
            document.error_message,
        )

    # 将 SQLite 行还原成 Document
    @staticmethod
    def _from_row(row: tuple[object, ...]) -> KnowledgeDocument:
        indexed_at_raw = row[10]
        indexed_at = datetime.fromisoformat(str(indexed_at_raw)) if indexed_at_raw else None
        return KnowledgeDocument(
            id=str(row[0]),
            source=str(row[1]),
            title=str(row[2]),
            scope=str(row[3]),  # type: ignore[arg-type]
            project_id=str(row[4]) if row[4] is not None else None,
            user_id=str(row[5]) if row[5] is not None else None,
            mime_type=str(row[6]),
            metadata=json.loads(str(row[7])),
            content_hash=str(row[8]) if row[8] is not None else None,
            source_mtime=float(str(row[9])) if row[9] is not None else None,
            indexed_at=indexed_at,
            status=str(row[11]),
            chunk_count=int(str(row[12])),
            error_message=str(row[13]) if row[13] is not None else None,
        )
