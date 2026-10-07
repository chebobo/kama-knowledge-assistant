from __future__ import annotations

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter, KnowledgeSearchResult
from kama_claude.core.knowledge.retriever.hybrid import HybridRetriever


class _Retriever:
    # 初始化记录查询并返回候选的检索替身
    def __init__(
        self,
        results: list[KnowledgeSearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int, KnowledgeFilter]] = []

    # 记录 query、top_k 和 filter 后返回或抛出预设结果
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        self.calls.append((query, top_k, filters))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


# 构造指定 rank 的检索候选
def _result(identifier: str, rank: int) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        chunk=KnowledgeChunk(
            id=identifier,
            document_id="doc-1",
            content=identifier,
            source="doc.md",
            title="Document",
        ),
        score=0.9,
        rank=rank,
    )


# 功能：验证 Hybrid 并行传递 scope 和 candidate pool 并融合两路结果
# 设计：注入内存检索替身记录两路参数，检查共享 chunk 优先且最终截断到 top_k
async def test_hybrid_retrieves_candidates_and_fuses_results() -> None:
    filters = KnowledgeFilter(user_id="user-a", project_id="project-a")
    dense = _Retriever([_result("shared", 1), _result("dense", 2)])
    bm25 = _Retriever([_result("shared", 1), _result("lexical", 2)])
    hybrid = HybridRetriever(dense=dense, bm25=bm25, candidate_multiplier=3)

    results = await hybrid.retrieve("query", top_k=2, filters=filters)

    assert dense.calls == [("query", 6, filters)]
    assert bm25.calls == [("query", 6, filters)]
    assert [result.chunk.id for result in results] == ["shared", "dense"]
    assert [result.rank for result in results] == [1, 2]


# 功能：验证一路异常时 Hybrid 仍返回另一路可用候选
# 设计：令 BM25 抛错并让 Dense 命中，覆盖部分失效的降级路径
async def test_hybrid_continues_when_one_source_fails() -> None:
    dense = _Retriever([_result("dense", 1)])
    bm25 = _Retriever(error=RuntimeError("fts unavailable"))
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    results = await hybrid.retrieve("query", top_k=4, filters=KnowledgeFilter())

    assert [result.chunk.id for result in results] == ["dense"]


# 功能：验证两路都失败时 Hybrid 抛出明确的整体检索错误
# 设计：两路注入不同后端异常，确保不能把全失败误报成空命中
async def test_hybrid_raises_when_all_sources_fail() -> None:
    hybrid = HybridRetriever(
        dense=_Retriever(error=RuntimeError("qdrant unavailable")),
        bm25=_Retriever(error=RuntimeError("fts unavailable")),
    )

    try:
        await hybrid.retrieve("query", top_k=4, filters=KnowledgeFilter())
    except RuntimeError as exc:
        assert str(exc) == "all hybrid retrieval sources failed"
    else:
        raise AssertionError("expected all-source failure")
