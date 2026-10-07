from __future__ import annotations

import pytest

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeSearchResult
from kama_claude.core.knowledge.retriever.fusion import (
    aggregate_document_results,
    reciprocal_rank_fusion,
)


# 构造可指定检索 rank 的融合输入
def _result(identifier: str, rank: int | None = None) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        chunk=KnowledgeChunk(
            id=identifier,
            document_id=f"doc-{identifier}",
            content=identifier,
            source="source.md",
            title="Source",
        ),
        score=0.9,
        rank=rank,
    )


# 功能：验证重复 chunk 的 RRF 分数等于各路 reciprocal rank 贡献之和
# 设计：为两路提供不对称 rank 的同一 chunk，直接断言公式结果和最终 rank
def test_rrf_combines_duplicate_chunk_ranks() -> None:
    result = reciprocal_rank_fusion(
        [[_result("shared", 1), _result("dense-only", 2)], [_result("shared", 3)]],
        rrf_k=10,
    )

    assert [item.chunk.id for item in result] == ["shared", "dense-only"]
    assert result[0].score == pytest.approx(1 / 11 + 1 / 13)
    assert result[1].score == pytest.approx(1 / 12)
    assert [(item.retriever, item.rank) for item in result] == [("hybrid", 1), ("hybrid", 2)]


# 功能：验证空候选列表、稳定并列顺序和 top_k 截断
# 设计：以无 rank 的等分结果构造完全并列，检查输入路顺序作为稳定次序
def test_rrf_handles_empty_lists_ties_and_limit() -> None:
    result = reciprocal_rank_fusion(
        [[], [_result("first"), _result("second")], [_result("third")]],
        rrf_k=60,
        top_k=2,
    )

    assert [item.chunk.id for item in result] == ["first", "third"]


# 功能：验证 RRF 拒绝负参数
# 设计：分别覆盖公式常数和截断上限，避免无效配置产生不透明排序
def test_rrf_rejects_negative_parameters() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], rrf_k=-1)
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=-1)


# 功能：验证 document aggregation 按 document_id 求和并重新编号
# 设计：同一文档两个 chunk 应合并为一个结果，且不能占据两个输出席位
def test_document_aggregation_sums_chunks_and_limits_documents() -> None:
    results = [
        KnowledgeSearchResult(
            chunk=KnowledgeChunk(
                id="a1", document_id="doc-a", content="a1", source="a.md", title="A"
            ),
            score=0.5,
            retriever="hybrid",
            rank=1,
        ),
        KnowledgeSearchResult(
            chunk=KnowledgeChunk(
                id="b1", document_id="doc-b", content="b1", source="b.md", title="B"
            ),
            score=0.8,
            retriever="hybrid",
            rank=2,
        ),
        KnowledgeSearchResult(
            chunk=KnowledgeChunk(
                id="a2", document_id="doc-a", content="a2", source="a.md", title="A"
            ),
            score=0.5,
            retriever="hybrid",
            rank=3,
        ),
    ]

    aggregated = aggregate_document_results(results, aggregation="sum", top_k=1)

    assert len(aggregated) == 1
    assert aggregated[0].chunk.document_id == "doc-a"
    assert aggregated[0].score == pytest.approx(1.0)
    assert aggregated[0].rank == 1


# 功能：验证 document aggregation 拒绝生产路径尚未支持的函数
def test_document_aggregation_rejects_unknown_function() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_document_results([], aggregation="mean")  # type: ignore[arg-type]
