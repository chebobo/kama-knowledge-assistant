from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter
from kama_claude.core.knowledge.store.qdrant import QdrantUnavailableError, QdrantVectorStore


def _chunk(identifier: str, project_id: str | None) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=identifier,
        document_id=identifier,
        content=f"content {identifier}",
        source=f"{identifier}.md",
        title=identifier,
        scope="project" if project_id else "global",
        project_id=project_id,
        user_id="user-a",
    )


# 功能：验证 Qdrant Local collection 能持久化写入并按 project scope 检索
# 设计：使用项目目录中的临时路径和固定二维向量，覆盖 Local 初始化、重建客户端和 metadata filter
@pytest.mark.asyncio
async def test_qdrant_local_persistence_and_scope(
    tmp_path: Path,
    qdrant_store: Callable[..., QdrantVectorStore],
) -> None:
    path = tmp_path / "qdrant"
    store = qdrant_store(path, dimension=2)
    await store.initialize()
    await store.upsert(
        [_chunk("global", None), _chunk("project-a", "project-a"), _chunk("project-b", "project-b")],
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
    )

    results = await store.search(
        [1.0, 0.0], top_k=8, filters=KnowledgeFilter(user_id="user-a", project_id="project-a")
    )
    ids = {item.chunk.id for item in results}
    assert "global" in ids
    assert "project-a" in ids
    assert "project-b" not in ids

    # 关闭旧客户端释放 Local 目录文件锁，模拟进程重启后重新打开
    store.close()
    reopened = qdrant_store(path, dimension=2)
    reopened_results = await reopened.search(
        [1.0, 0.0], top_k=8, filters=KnowledgeFilter(user_id="user-a", project_id="project-a")
    )
    assert {item.chunk.id for item in reopened_results} == ids


# 功能：验证已有 collection 的向量维度不匹配时拒绝初始化
# 设计：先创建二维 collection，再用三维配置重新打开，确保模型切换不会静默写入错误向量
@pytest.mark.asyncio
async def test_qdrant_dimension_mismatch(
    tmp_path: Path,
    qdrant_store: Callable[..., QdrantVectorStore],
) -> None:
    path = tmp_path / "qdrant"
    two_dim = qdrant_store(path, dimension=2)
    await two_dim.initialize()
    two_dim.close()  # 释放锁，才能用三维配置重新打开同一目录
    with pytest.raises(QdrantUnavailableError, match="dimension mismatch"):
        await qdrant_store(path, dimension=3).initialize()
