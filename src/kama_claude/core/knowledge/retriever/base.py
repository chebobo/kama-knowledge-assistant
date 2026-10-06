from __future__ import annotations

from typing import Protocol

from kama_claude.core.knowledge.models import KnowledgeFilter, KnowledgeSearchResult


class Retriever(Protocol):
    # 根据查询文本和范围返回排序后的知识结果
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]: ...

