from __future__ import annotations

import pytest

from kama_claude.core.knowledge.models import (
    KnowledgeChunk,
    KnowledgeFilter,
    KnowledgeSearchResult,
)
from kama_claude.core.knowledge.retriever.dense import DenseRetriever


class _Embedding:
    # 初始化记录查询 embedding 输入的测试替身
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or [[0.25, 0.75]]
        self.calls: list[list[str]] = []

    # 返回预设向量并记录查询文本
    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.vectors


class _Store:
    # 初始化记录向量检索参数的存储替身
    def __init__(self, results: list[KnowledgeSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[float], int, KnowledgeFilter]] = []
        self.error: Exception | None = None

    # 捕获向量、top_k 和过滤器并返回预设结果
    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        self.calls.append((embedding, top_k, filters))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


# 构造带固定来源信息的检索结果
def _result(identifier: str) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        chunk=KnowledgeChunk(
            id=identifier,
            document_id="doc-1",
            content=identifier,
            source="doc.md",
            title="Document",
        ),
        score=0.8,
    )


# 功能：验证 DenseRetriever 传递 top_k 和权限过滤并标注稳定 rank
# 设计：使用内存替身隔离 embedding 与 Qdrant，直接检查存储收到的原始过滤条件
@pytest.mark.asyncio
async def test_dense_retriever_passes_filters_and_annotates_results() -> None:
    embedding = _Embedding()
    filters = KnowledgeFilter(user_id="user-a", project_id="project-a")
    store = _Store([_result("first"), _result("second")])
    retriever = DenseRetriever(embedding=embedding, store=store)  # type: ignore[arg-type]

    results = await retriever.retrieve("query", top_k=2, filters=filters)

    assert embedding.calls == [["query"]]
    assert store.calls == [([0.25, 0.75], 2, filters)]
    assert [(item.retriever, item.rank) for item in results] == [("dense", 1), ("dense", 2)]


# 功能：验证 DenseRetriever 在存储无命中时返回空列表
# 设计：显式检查空候选的正常路径，避免调用方将无结果误判为后端异常
@pytest.mark.asyncio
async def test_dense_retriever_returns_empty_results() -> None:
    retriever = DenseRetriever(embedding=_Embedding(), store=_Store())  # type: ignore[arg-type]

    assert await retriever.retrieve("query", top_k=4, filters=KnowledgeFilter()) == []


# 功能：验证 DenseRetriever 将向量存储异常交还给调用方
# 设计：注入同一个异常对象并确认异常类型与消息未被静默吞掉或改写
@pytest.mark.asyncio
async def test_dense_retriever_propagates_store_errors() -> None:
    store = _Store()
    store.error = RuntimeError("store unavailable")
    retriever = DenseRetriever(embedding=_Embedding(), store=store)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="store unavailable"):
        await retriever.retrieve("query", top_k=4, filters=KnowledgeFilter())


# 功能：验证 embedding provider 未返回唯一 query vector 时明确失败
# 设计：返回两个向量触发协议校验，确保错误在访问向量存储前暴露
@pytest.mark.asyncio
async def test_dense_retriever_rejects_invalid_query_embedding_count() -> None:
    store = _Store()
    retriever = DenseRetriever(
        embedding=_Embedding([[1.0, 0.0], [0.0, 1.0]]),
        store=store,
    )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="one vector"):
        await retriever.retrieve("query", top_k=4, filters=KnowledgeFilter())

    assert store.calls == []
