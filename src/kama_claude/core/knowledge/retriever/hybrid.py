from __future__ import annotations

import asyncio
import logging

from kama_claude.core.knowledge.models import KnowledgeFilter, KnowledgeSearchResult
from kama_claude.core.knowledge.retriever.base import Retriever
from kama_claude.core.knowledge.retriever.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    # 初始化 Dense 与 BM25 双路融合参数
    def __init__(
        self,
        *,
        dense: Retriever,
        bm25: Retriever,
        candidate_multiplier: int = 2,
        rrf_k: int = 60,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self._dense = dense
        self._bm25 = bm25
        self._candidate_multiplier = candidate_multiplier
        self._rrf_k = rrf_k

    # 并行检索两路候选并在单路故障时保留可用结果
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidate_k = top_k * self._candidate_multiplier
        outcomes = await asyncio.gather(
            self._dense.retrieve(query, top_k=candidate_k, filters=filters),
            self._bm25.retrieve(query, top_k=candidate_k, filters=filters),
            return_exceptions=True,
        )
        result_lists: list[list[KnowledgeSearchResult]] = []
        for source, outcome in zip(("dense", "bm25"), outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                logger.warning(
                    "%s retrieval failed; continuing with remaining source: %s",
                    source,
                    outcome,
                )
            else:
                result_lists.append(outcome)
        if not result_lists:
            errors = [item for item in outcomes if isinstance(item, BaseException)]
            raise RuntimeError("all hybrid retrieval sources failed") from BaseExceptionGroup(
                "hybrid retrieval failures", errors
            )
        return reciprocal_rank_fusion(result_lists, rrf_k=self._rrf_k, top_k=top_k)
