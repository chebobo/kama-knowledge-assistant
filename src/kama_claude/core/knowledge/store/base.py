from __future__ import annotations

from typing import Protocol

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter, KnowledgeSearchResult


class VectorStore(Protocol):
    # 写入知识片段和对应的 embedding 向量
    async def upsert(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
    ) -> None: ...

    # 按向量和 scope 过滤条件检索知识片段
    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]: ...

    # 删除指定文档下的全部知识片段
    async def delete_document(self, document_id: str) -> None: ...

