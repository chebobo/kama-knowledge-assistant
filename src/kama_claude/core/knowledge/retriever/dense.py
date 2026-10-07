from __future__ import annotations

from kama_claude.core.knowledge.embedding.base import EmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeFilter, KnowledgeSearchResult
from kama_claude.core.knowledge.store.base import VectorStore


class DenseRetriever:
    # 使用本地 embedding 和向量存储执行 Dense 检索
    def __init__(self, *, embedding: EmbeddingProvider, store: VectorStore) -> None:
        self._embedding = embedding
        self._store = store

    # 生成查询向量并在存储层应用 scope filter
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        vectors = await self._embedding.embed([query])
        if len(vectors) != 1:
            raise ValueError("query embedding provider must return one vector")
        results = await self._store.search(vectors[0], top_k=top_k, filters=filters)
        return [
            KnowledgeSearchResult(
                chunk=result.chunk,
                score=result.score,
                retriever="dense",
                rank=index,
            )
            for index, result in enumerate(results, start=1)
        ]
