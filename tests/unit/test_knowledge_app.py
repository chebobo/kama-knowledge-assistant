from __future__ import annotations

from pathlib import Path

import pytest

import kama_claude.core.app as app_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.knowledge.models import KnowledgeFilter
from kama_claude.core.knowledge.retriever.dense import DenseRetriever


class _Embedding:
    # 提供初始化所需的固定向量维度和 tokenizer
    dimension = 2
    tokenizer = object()

    # 为 Dense fallback 查询返回固定二维向量
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _VectorStore:
    # 初始化记录 Dense fallback 搜索的向量存储替身
    def __init__(self, path: Path, *, dimension: int, collection: str) -> None:
        self.filters: list[KnowledgeFilter] = []

    # 模拟异步 Qdrant 初始化
    async def initialize(self) -> None:
        return None

    # 捕获过滤条件并返回空结果
    async def search(self, embedding, *, top_k, filters):
        self.filters.append(filters)
        return []


class _Catalog:
    # 提供 CoreApp 初始化所需的 Catalog 构造替身
    def __init__(self, path: Path) -> None:
        self.path = path


# 功能：验证 Hybrid lexical 初始化失败时 CoreApp 仍创建 Dense KnowledgeService
# 设计：替换模型和存储构造器并注入 FTS 初始化错误，检查最终服务仍能按 scope 检索
@pytest.mark.asyncio
async def test_core_app_falls_back_to_dense_when_fts_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "LocalEmbeddingProvider", lambda model: _Embedding())
    monkeypatch.setattr(app_module, "QdrantVectorStore", _VectorStore)
    monkeypatch.setattr(app_module, "SqliteDocumentCatalog", _Catalog)
    monkeypatch.setattr(app_module, "HuggingFaceTokenCounter", lambda tokenizer: object())

    class _UnavailableLexicalStore:
        # 模拟 FTS5 tokenizer 不可用时的初始化错误
        def __init__(self, path: Path) -> None:
            raise RuntimeError("FTS5 trigram unavailable")

    monkeypatch.setattr(app_module, "SqliteLexicalStore", _UnavailableLexicalStore)
    config = KamaConfig()
    config.knowledge.enabled = True
    config.knowledge.hybrid_enabled = True

    service = await app_module.CoreApp()._build_knowledge_service(config)

    assert service is not None
    assert isinstance(service._retriever, DenseRetriever)
    assert await service.search("query", user_id="user-a", project_id="project-a") == []
