from __future__ import annotations

from dataclasses import dataclass

from kama_claude.core.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeFilter,
    KnowledgeSearchResult,
)
from kama_claude.core.knowledge.service import KnowledgeService
from kama_claude.core.knowledge.token_budget import CharacterTokenCounter
from kama_claude.core.tools.builtin.knowledge_search import KnowledgeSearchTool


@dataclass
class _Embedding:
    # 初始化固定维度的 embedding 测试替身
    dimension: int = 2

    # 记录最近一次输入的文本列表，便于断言标题注入
    last_texts: list[str] | None = None

    # 为每段文本返回固定向量，避免依赖真实模型
    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.last_texts = list(texts)
        return [[1.0, 0.0] for _ in texts]


class _Store:
    # 初始化记录调用参数的向量存储测试替身
    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self.filters: list[KnowledgeFilter] = []
        self.deleted: list[str] = []
        self.fail_upsert = False
        self.fail_delete = False

    # 记录写入的片段和向量数量
    async def upsert(self, chunks, embeddings) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant write failed")
        assert len(chunks) == len(embeddings)
        self.chunks = list(chunks)

    # 按 scope 返回固定结果，模拟存储层过滤
    async def search(self, embedding, *, top_k, filters):
        self.filters.append(filters)
        return [
            KnowledgeSearchResult(chunk=chunk, score=1.0 - i * 0.1)
            for i, chunk in enumerate(self.chunks[:top_k])
            if filters.project_id is None or chunk.project_id in (None, filters.project_id)
        ]

    # 记录删除的文档 ID
    async def delete_document(self, document_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("qdrant delete failed")
        self.deleted.append(document_id)


class _LexicalStore:
    # 初始化记录 lexical index 操作的测试替身
    def __init__(self) -> None:
        self.documents: dict[str, list[KnowledgeChunk]] = {}
        self.fail_replace = False
        self.fail_delete = False

    # 按文档替换 lexical chunks，可注入写入异常
    async def replace_document(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        if self.fail_replace:
            raise RuntimeError("fts write failed")
        self.documents[document_id] = list(chunks)

    # 按文档删除 lexical chunks，可注入删除异常
    async def delete_document(self, document_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("fts delete failed")
        self.documents.pop(document_id, None)


class _Catalog:
    # 初始化记录文档状态的目录测试替身
    def __init__(self) -> None:
        self.documents: dict[str, KnowledgeDocument] = {}

    # 保存文档状态副本
    async def upsert(self, document: KnowledgeDocument) -> None:
        self.documents[document.id] = document

    # 根据 document ID 返回状态记录
    async def get(self, document_id: str) -> KnowledgeDocument | None:
        return self.documents.get(document_id)

    # 删除文档状态记录
    async def delete(self, document_id: str) -> None:
        self.documents.pop(document_id, None)


def _chunk(content: str = "Agent architecture details") -> KnowledgeChunk:
    return KnowledgeChunk(
        id="chunk-1",
        document_id="doc-1",
        content=content,
        source="docs/agent.md",
        title="Agent Architecture",
        scope="project",
        project_id="project-a",
    )


# 功能：验证文档索引会先生成 embedding、删除旧文档向量并写入新片段
# 设计：使用固定 embedding 和内存 store，断言完整调用结果而不依赖 Qdrant 或模型文件
async def test_index_writes_chunks_and_updates_status() -> None:
    store = _Store()
    service = KnowledgeService(embedding=_Embedding(), store=store)
    document = KnowledgeDocument(
        id="doc-1", source="docs/agent.md", title="Agent Architecture", scope="project", project_id="project-a"
    )

    result = await service.index(document, [_chunk()])

    assert result.document_id == "doc-1"
    assert result.chunk_count == 1
    assert document.status == "ready"
    assert store.deleted == ["doc-1"]
    assert store.chunks[0].content == "Agent architecture details"


# 功能：验证 search 将当前 user/project scope 传递给底层存储
# 设计：检查 store 捕获的过滤对象，确保调用方不能绕过 KnowledgeService 自行过滤
async def test_search_passes_scope_filter() -> None:
    store = _Store()
    service = KnowledgeService(embedding=_Embedding(), store=store)
    await service.index(
        KnowledgeDocument(id="doc-1", source="x", title="x", scope="project", project_id="project-a"),
        [_chunk()],
    )

    results = await service.search("architecture", user_id="user-1", project_id="project-a", top_k=8)

    assert len(results) == 1
    assert store.filters[-1] == KnowledgeFilter(user_id="user-1", project_id="project-a")


# 功能：验证 retrieve_context 保留标题、来源和分数，并遵守 token budget
# 设计：使用字符 token 替身构造超预算内容，确认结果包含来源且不会无限追加文本
async def test_retrieve_context_formats_and_limits_results() -> None:
    store = _Store()
    service = KnowledgeService(
        embedding=_Embedding(),
        store=store,
        token_counter=CharacterTokenCounter(chars_per_token=1),
        max_retrieved_tokens=220,
    )
    await service.index(
        KnowledgeDocument(id="doc-1", source="x", title="x"),
        [_chunk("A" * 300)],
    )

    context = await service.retrieve_context("architecture")

    assert "Retrieved Knowledge" in context
    assert "source: docs/agent.md" in context
    assert len(context) <= 220


# 功能：验证空查询不调用向量存储并返回空结果
# 设计：空白查询是工具输入的边界情况，直接断言空列表避免产生无意义 embedding
async def test_empty_query_returns_no_results() -> None:
    store = _Store()
    service = KnowledgeService(embedding=_Embedding(), store=store)

    assert await service.search("   ") == []
    assert store.filters == []


# 功能：验证索引时 embedding 输入包含标题、section 和描述前缀（元数据注入）
# 设计：fake embedding 记录输入文本，断言 title/section/description 参与向量计算而不只是正文
async def test_index_embeds_title_and_section_prefix() -> None:
    store = _Store()
    embedding = _Embedding()
    service = KnowledgeService(embedding=embedding, store=store)
    document = KnowledgeDocument(
        id="doc-1", source="x", title="Agent Architecture",
        metadata={"description": "主Agent与子Agent通信"},
    )
    chunk = _chunk("Agent loop details")
    chunk.metadata = {"section": "Tools", "chunk_index": 0}

    await service.index(document, [chunk])

    assert embedding.last_texts is not None
    text = embedding.last_texts[0]
    assert "title: Agent Architecture" in text
    assert "section: Tools" in text
    assert "description: 主Agent与子Agent通信" in text
    assert "Agent loop details" in text


# 功能：验证索引成功时 Qdrant store 与 lexical store 收到相同 chunks
# 设计：注入两个内存 store 并比较 chunk ID 集合，验证双写后才标记 ready
async def test_index_updates_vector_and_lexical_stores() -> None:
    store = _Store()
    lexical = _LexicalStore()
    catalog = _Catalog()
    service = KnowledgeService(
        embedding=_Embedding(), store=store, lexical_store=lexical, catalog=catalog
    )
    document = KnowledgeDocument(id="doc-1", source="x", title="x")

    await service.index(document, [_chunk()])

    assert [chunk.id for chunk in store.chunks] == [chunk.id for chunk in lexical.documents["doc-1"]]
    assert catalog.documents["doc-1"].status == "ready"


# 功能：验证 lexical 写入失败时 Catalog 标为 failed 并可通过 reindex 修复
# 设计：首次注入 FTS 错误后关闭故障开关重试，覆盖跨存储失败恢复路径
async def test_lexical_index_failure_is_recorded_and_reindex_repairs() -> None:
    store = _Store()
    lexical = _LexicalStore()
    lexical.fail_replace = True
    catalog = _Catalog()
    service = KnowledgeService(
        embedding=_Embedding(), store=store, lexical_store=lexical, catalog=catalog
    )
    document = KnowledgeDocument(id="doc-1", source="x", title="x")

    try:
        await service.index(document, [_chunk()])
    except RuntimeError as exc:
        assert str(exc) == "fts write failed"
    else:
        raise AssertionError("expected lexical write failure")
    assert catalog.documents["doc-1"].status == "failed"
    assert "fts write failed" in str(catalog.documents["doc-1"].error_message)

    lexical.fail_replace = False
    await service.reindex(document, [_chunk()])

    assert catalog.documents["doc-1"].status == "ready"
    assert [chunk.id for chunk in lexical.documents["doc-1"]] == ["chunk-1"]


# 功能：验证删除单边失败时仍尝试另一路并保留 failed Catalog 记录
# 设计：使 lexical 删除失败而 vector 删除成功，确认状态不被误报为文档已删除
async def test_delete_partial_failure_keeps_failed_catalog_record() -> None:
    store = _Store()
    lexical = _LexicalStore()
    lexical.fail_delete = True
    catalog = _Catalog()
    document = KnowledgeDocument(id="doc-1", source="x", title="x", status="ready")
    catalog.documents[document.id] = document
    service = KnowledgeService(
        embedding=_Embedding(), store=store, lexical_store=lexical, catalog=catalog
    )

    try:
        await service.delete("doc-1")
    except RuntimeError as exc:
        assert "failed to delete" in str(exc)
    else:
        raise AssertionError("expected deletion failure")

    assert store.deleted == ["doc-1"]
    assert catalog.documents["doc-1"].status == "failed"


# 功能：验证 Qdrant 写入失败时保留 failed 状态并由 reindex 恢复双存储
# 设计：使 FTS 清理成功而向量 upsert 失败，再关闭故障重试以覆盖另一侧故障恢复
async def test_vector_index_failure_is_recorded_and_reindex_repairs() -> None:
    store = _Store()
    store.fail_upsert = True
    lexical = _LexicalStore()
    catalog = _Catalog()
    service = KnowledgeService(
        embedding=_Embedding(), store=store, lexical_store=lexical, catalog=catalog
    )
    document = KnowledgeDocument(id="doc-1", source="x", title="x")

    try:
        await service.index(document, [_chunk()])
    except RuntimeError as exc:
        assert str(exc) == "qdrant write failed"
    else:
        raise AssertionError("expected vector write failure")
    assert catalog.documents["doc-1"].status == "failed"
    assert "doc-1" not in lexical.documents

    store.fail_upsert = False
    await service.reindex(document, [_chunk()])

    assert catalog.documents["doc-1"].status == "ready"
    assert [chunk.id for chunk in store.chunks] == [chunk.id for chunk in lexical.documents["doc-1"]]


# 功能：验证内容 hash 对相同文本稳定且对变更文本敏感
# 设计：使用服务的默认实例覆盖 Document Catalog 判断增量索引的基础不变量
def test_content_hash_is_stable() -> None:
    service = KnowledgeService(embedding=_Embedding(), store=_Store())
    first = service.content_hash("same")
    second = service.content_hash("same")
    changed = service.content_hash("changed")

    assert first == second
    assert first != changed
# 功能：验证 knowledge_search 只使用绑定的用户和项目 scope
# 设计：使用最小异步 stub 记录参数，确认工具输入中不存在可覆盖 scope 的字段
async def test_knowledge_search_inherits_scope() -> None:
    class Stub:
        async def retrieve_context(self, query: str, *, user_id: str | None, project_id: str | None, top_k: int) -> str:
            assert query == "architecture"
            assert user_id == "user-a"
            assert project_id == "project-a"
            assert top_k == 3
            return "knowledge"

    tool = KnowledgeSearchTool(Stub(), user_id="user-a", project_id="project-a", default_top_k=8)  # type: ignore[arg-type]
    result = await tool.invoke({"query": "architecture", "top_k": 3, "project_id": "project-b"})
    assert result.content == "knowledge"
    assert not result.is_error


# 功能：验证 KnowledgeService 不可用时工具返回明确错误
# 设计：直接传入 None，覆盖初始化失败降级后的工具行为，不依赖真实向量后端
async def test_knowledge_search_unavailable() -> None:
    tool = KnowledgeSearchTool(None, user_id=None, project_id=None)
    result = await tool.invoke({"query": "anything"})
    assert result.is_error
    assert result.error_type == "runtime_error"
