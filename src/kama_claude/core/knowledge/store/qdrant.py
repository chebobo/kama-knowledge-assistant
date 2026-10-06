from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter, KnowledgeSearchResult

# 固定命名空间：用于把业务 chunk_id 稳定映射为 Qdrant 接受的 UUID
_POINT_ID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class QdrantUnavailableError(RuntimeError):
    # 表示 Qdrant 客户端依赖或本地存储初始化不可用
    pass


class QdrantVectorStore:
    # 初始化 Qdrant Local 存储配置并延迟创建客户端
    def __init__(
        self,
        path: Path,
        *,
        dimension: int,
        collection: str = "kama_knowledge",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._path = path.expanduser()
        self._dimension = dimension
        self._collection = collection
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    # 把业务 chunk_id 稳定映射为 Qdrant 可接受的 UUID（同一 chunk_id 永远映射同一 UUID）
    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(_POINT_ID_NS, chunk_id))

    # 初始化 Qdrant Local 客户端和 collection
    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    # 写入片段和向量
    async def upsert(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> None:
        self._validate_vectors(chunks, embeddings)
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._upsert_sync, chunks, embeddings)

    # 按 payload scope filter 执行向量搜索
    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        if len(embedding) != self._dimension:
            raise ValueError("query embedding dimension does not match collection dimension")
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, embedding, top_k, filters)

    # 按 document_id 删除全部片段
    async def delete_document(self, document_id: str) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._delete_document_sync, document_id)

    # 关闭底层 Qdrant 客户端，释放 Local 模式目录文件锁
    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # 导入可选 Qdrant 依赖并初始化 collection
    def _initialize_sync(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise QdrantUnavailableError(
                "qdrant-client is required for QdrantVectorStore"
            ) from exc
        self._path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(self._path))
        collections = {item.name for item in client.get_collections().collections}
        if self._collection not in collections:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            info = client.get_collection(self._collection)
            configured_vectors = info.config.params.vectors
            configured_size = getattr(configured_vectors, "size", None)
            if configured_size != self._dimension:
                raise QdrantUnavailableError(
                    f"Qdrant collection dimension mismatch: {configured_size} != {self._dimension}"
                )
        self._client = client

    # 同步写入 Qdrant points
    def _upsert_sync(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> None:
        if self._client is None:
            raise QdrantUnavailableError("Qdrant store is not initialized")
        from qdrant_client.http import models

        points = [
            models.PointStruct(
                id=self._point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "title": chunk.title,
                    "scope": chunk.scope,
                    "project_id": chunk.project_id,
                    "user_id": chunk.user_id,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    # 同步执行 Qdrant 过滤搜索
    def _search_sync(
        self,
        embedding: list[float],
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        if self._client is None:
            raise QdrantUnavailableError("Qdrant store is not initialized")
        from qdrant_client.http import models

        must: list[Any] = []
        if filters.user_id is not None:
            must.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=filters.user_id),
                )
            )
        scope_should: list[Any] = [
            models.FieldCondition(
                key="scope", match=models.MatchValue(value="global")
            )
        ]
        if filters.project_id is not None:
            scope_should.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="scope", match=models.MatchValue(value="project")
                        ),
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=filters.project_id),
                        ),
                    ]
                )
            )
        query_filter = models.Filter(must=must, should=scope_should)
        response = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results: list[KnowledgeSearchResult] = []
        for item in response.points:
            payload = item.payload or {}
            results.append(
                KnowledgeSearchResult(
                    chunk=KnowledgeChunk(
                        id=str(payload.get("chunk_id", item.id)),
                        document_id=str(payload.get("document_id", "")),
                        content=str(payload.get("content", "")),
                        source=str(payload.get("source", "")),
                        title=str(payload.get("title", "")),
                        scope=str(payload.get("scope", "global")),  # type: ignore[arg-type]
                        project_id=payload.get("project_id"),
                        user_id=payload.get("user_id"),
                        metadata=dict(payload.get("metadata", {})),
                    ),
                    score=float(item.score),
                )
            )
        return results

    # 同步按 document_id 删除 Qdrant points
    def _delete_document_sync(self, document_id: str) -> None:
        if self._client is None:
            raise QdrantUnavailableError("Qdrant store is not initialized")
        from qdrant_client.http import models

        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    # 校验批量向量数量和维度
    def _validate_vectors(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("embedding count does not match chunk count")
        if any(len(vector) != self._dimension for vector in embeddings):
            raise ValueError("embedding dimension does not match configured dimension")
