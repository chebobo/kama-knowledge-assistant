from __future__ import annotations

from pathlib import Path

from kama_claude.core.knowledge.catalog import SqliteDocumentCatalog
from kama_claude.core.knowledge.models import KnowledgeDocument
from kama_claude.core.knowledge.service import KnowledgeService


class _Embedding:
    # 返回固定向量，隔离文档入库测试与真实模型
    dimension = 2

    # 为每个 chunk 返回一个固定向量
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Store:
    # 初始化记录删除和写入行为的向量存储替身
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted = []

    # 记录批量写入的片段
    async def upsert(self, chunks, embeddings) -> None:
        self.upserted.append((list(chunks), list(embeddings)))

    # 记录文档向量删除
    async def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)

    # 入库测试不需要检索实现
    async def search(self, embedding, *, top_k, filters):
        return []


# 功能：验证 KnowledgeService 能从 Markdown 文件解析、分段并写入 Catalog 和 Store
# 设计：使用真实临时文本内容和 fake embedding/store，覆盖 parser、chunker、catalog 的完整索引链路
async def test_index_file_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "agent.md"
    source.write_text(
        "# Architecture\n\n"
        "Agent loop details with enough content to exceed the minimum chunk size "
        "so this section is retained by the chunker.\n\n"
        "## Tools\n\n"
        "Tool registry and invocation details that are long enough to be kept as well.",
        encoding="utf-8",
    )
    catalog = SqliteDocumentCatalog(tmp_path / "catalog.db")
    store = _Store()
    service = KnowledgeService(embedding=_Embedding(), store=store, catalog=catalog)
    document = KnowledgeDocument(id="doc-1", source=str(source), title="")

    result = await service.index(document)

    loaded = await catalog.get("doc-1")
    assert result.chunk_count == 2
    assert loaded is not None
    assert loaded.status == "ready"
    assert loaded.mime_type == "text/markdown"
    assert loaded.chunk_count == 2
    assert len(store.upserted) == 1
    assert store.deleted == ["doc-1"]
    assert store.upserted[0][0][0].metadata["section"] == "Architecture"


# 功能：验证 reindex(document_id) 会从 Catalog 恢复 source 并重新建立索引
# 设计：先完成一次真实文件索引，再通过 ID 重建，确认调用方不需要手持 chunks
async def test_reindex_by_document_id(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text(
        "first version with enough content to exceed the minimum chunk size threshold.",
        encoding="utf-8",
    )
    catalog = SqliteDocumentCatalog(tmp_path / "catalog.db")
    store = _Store()
    service = KnowledgeService(embedding=_Embedding(), store=store, catalog=catalog)
    document = KnowledgeDocument(id="doc-1", source=str(source), title="Notes")
    await service.index(document)

    source.write_text(
        "second version with different content that is also long enough for indexing.",
        encoding="utf-8",
    )
    result = await service.reindex("doc-1")

    assert result.chunk_count == 1
    assert len(store.upserted) == 2
    assert store.upserted[-1][0][0].content == (
        "second version with different content that is also long enough for indexing."
    )


# 功能：验证 parser 失败时 Catalog 会记录 failed 状态和错误信息
# 设计：传入不存在的 source，检查失败状态持久化，确保后续管理操作可诊断
async def test_index_failure_updates_catalog(tmp_path: Path) -> None:
    catalog = SqliteDocumentCatalog(tmp_path / "catalog.db")
    service = KnowledgeService(embedding=_Embedding(), store=_Store(), catalog=catalog)
    document = KnowledgeDocument(
        id="doc-failed", source=str(tmp_path / "missing.md"), title="Missing"
    )

    try:
        await service.index(document)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("index should fail for missing source")

    loaded = await catalog.get("doc-failed")
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error_message


# 功能：验证 FileParser 提取 frontmatter 标题和描述作为文档元数据
# 设计：构造带 YAML frontmatter 的 Markdown，断言 parsed.title 取 frontmatter title、
#       description/keywords 进入 metadata
def test_parser_extracts_frontmatter(tmp_path: Path) -> None:
    from kama_claude.core.knowledge.parser.file import FileParser

    source = tmp_path / "agent.md"
    source.write_text(
        "---\n"
        "title: 多Agent架构面试全解析\n"
        "description: 深度拆解主Agent子Agent通信\n"
        "keywords: [多Agent, Agent通信, 编排]\n"
        "---\n"
        "# 正文标题\n\n正文内容。",
        encoding="utf-8",
    )
    parsed = FileParser().parse(source)
    assert parsed.title == "多Agent架构面试全解析"
    assert parsed.metadata["description"] == "深度拆解主Agent子Agent通信"
    assert "多Agent" in parsed.metadata["keywords"]
    assert "Agent通信" in parsed.metadata["keywords"]

