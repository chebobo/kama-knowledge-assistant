from __future__ import annotations

from typing import Literal

from kama_claude.core.knowledge.models import KnowledgeSearchResult

DocumentAggregation = Literal["sum", "max", "first-seen"]


# 使用 Reciprocal Rank Fusion 稳定合并多个候选列表
def reciprocal_rank_fusion(
    result_lists: list[list[KnowledgeSearchResult]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[KnowledgeSearchResult]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    fused: dict[str, tuple[KnowledgeSearchResult, float, int]] = {}
    order = 0
    for results in result_lists:
        for index, result in enumerate(results, start=1):
            key = result.chunk.id
            contribution = 1.0 / (rrf_k + (result.rank or index))
            if key in fused:
                existing, score, first_seen = fused[key]
                fused[key] = (existing, score + contribution, first_seen)
            else:
                fused[key] = (result, contribution, order)
                order += 1
    ordered = sorted(fused.values(), key=lambda item: (-item[1], item[2]))
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        ordered = ordered[:top_k]
    return [
        KnowledgeSearchResult(
            chunk=result.chunk,
            score=score,
            retriever="hybrid",
            rank=rank,
        )
        for rank, (result, score, _) in enumerate(ordered, start=1)
    ]


# 在 RRF 结果上按 document_id 聚合，缓解大文档多个 chunk 占位
def aggregate_document_results(
    results: list[KnowledgeSearchResult],
    *,
    aggregation: DocumentAggregation = "sum",
    top_k: int | None = None,
) -> list[KnowledgeSearchResult]:
    if aggregation not in {"sum", "max", "first-seen"}:
        raise ValueError("aggregation must be sum, max, or first-seen")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")

    grouped: dict[str, tuple[KnowledgeSearchResult, float, int]] = {}
    for first_seen, result in enumerate(results):
        document_id = result.chunk.document_id
        if document_id not in grouped:
            grouped[document_id] = (result, result.score, first_seen)
            continue
        representative, score, original_order = grouped[document_id]
        if aggregation == "sum":
            score += result.score
        elif aggregation == "max":
            score = max(score, result.score)
        grouped[document_id] = (representative, score, original_order)

    ordered = sorted(grouped.values(), key=lambda item: (-item[1], item[2]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        KnowledgeSearchResult(
            chunk=result.chunk,
            score=score,
            retriever="hybrid",
            rank=rank,
        )
        for rank, (result, score, _) in enumerate(ordered, start=1)
    ]
