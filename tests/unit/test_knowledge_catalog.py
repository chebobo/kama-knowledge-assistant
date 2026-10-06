from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.knowledge.catalog import SqliteDocumentCatalog
from kama_claude.core.knowledge.models import KnowledgeDocument


def _document() -> KnowledgeDocument:
    return KnowledgeDocument(
        id="doc-1",
        source="docs/agent.md",
        title="Agent Architecture",
        scope="project",
        project_id="project-a",
        user_id="user-1",
        metadata={"section": "loop"},
        content_hash="abc",
        source_mtime=12.5,
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
        status="ready",
        chunk_count=3,
    )


# 功能：验证 SQLite Catalog 可以保存并恢复完整 Document 字段
# 设计：使用 tmp_path 临时数据库和带 scope、metadata、时间字段的记录，覆盖目录持久化边界
async def test_catalog_roundtrip(tmp_path: Path) -> None:
    catalog = SqliteDocumentCatalog(tmp_path / "catalog-roundtrip.db")
    document = _document()

    await catalog.upsert(document)
    loaded = await catalog.get("doc-1")

    assert loaded == document


# 功能：验证同一 document_id upsert 会更新索引状态而不是产生重复记录
# 设计：连续写入同一 ID 后检查列表长度和最新状态，覆盖增量索引需要的幂等行为
async def test_catalog_upsert_is_idempotent(tmp_path: Path) -> None:
    catalog = SqliteDocumentCatalog(tmp_path / "catalog-upsert.db")
    document = _document()
    await catalog.upsert(document)
    document.status = "failed"
    document.error_message = "parser failed"
    await catalog.upsert(document)

    documents = await catalog.list_documents()

    assert len(documents) == 1
    assert documents[0].status == "failed"
    assert documents[0].error_message == "parser failed"


# 功能：验证删除文档会移除 Catalog 记录
# 设计：先写入再删除并读取，确认 Document 生命周期的删除边界
async def test_catalog_delete(tmp_path: Path) -> None:
    catalog = SqliteDocumentCatalog(tmp_path / "catalog-delete.db")
    await catalog.upsert(_document())

    await catalog.delete("doc-1")

    assert await catalog.get("doc-1") is None
