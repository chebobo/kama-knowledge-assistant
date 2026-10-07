from __future__ import annotations

from kama_claude.core.knowledge.models import KnowledgeFilter, KnowledgeSearchResult
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore


class BM25Retriever:
    # 通过 SQLite FTS5 trigram lexical index 执行 BM25 检索
    def __init__(self, store: SqliteLexicalStore) -> None:
        self._store = store

    # 查询 lexical store 并为返回结果添加来源和一基 rank
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        results = await self._store.search(query, top_k=top_k, filters=filters)
        return [
            KnowledgeSearchResult(
                chunk=result.chunk,
                score=result.score,
                retriever="bm25",
                rank=index,
            )
            for index, result in enumerate(results, start=1)
        ]
