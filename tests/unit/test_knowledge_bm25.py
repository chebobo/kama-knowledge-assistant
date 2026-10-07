from __future__ import annotations

import pytest

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore


# 构造带 scope 和 section 元数据的知识片段
def _chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    *,
    scope: str = "global",
    project_id: str | None = None,
    user_id: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        document_id=document_id,
        content=content,
        source=f"{document_id}.md",
        title="注意力机制",
        scope=scope,  # type: ignore[arg-type]
        project_id=project_id,
        user_id=user_id,
        metadata={"section": "多头结构"},
    )


# 功能：验证 trigram FTS5 可用性探测返回运行时 SQLite 版本
# 设计：直接创建临时内存虚表，不依赖开发机版本常量或外部数据库文件
def test_trigram_runtime_probe() -> None:
    version = SqliteLexicalStore.probe_trigram_support()

    assert version.count(".") == 2


# 功能：验证中文 BM25 命中并按负分升序返回来源与 rank
# 设计：对照同一查询词出现频率不同的 chunk，验证 FTS5 原生相关度排序
@pytest.mark.asyncio
async def test_bm25_retrieves_chinese_and_orders_negative_scores() -> None:
    store = SqliteLexicalStore(":memory:")
    await store.replace_document(
        "doc-a",
        [
            _chunk("rare", "doc-a", "多头注意力机制将特征拆分成独立子空间"),
            _chunk("repeated", "doc-a", "多头注意力 多头注意力 多头注意力机制"),
        ],
    )
    retriever = BM25Retriever(store)

    results = await retriever.retrieve("多头注意力", top_k=4, filters=KnowledgeFilter())

    assert {result.chunk.id for result in results} == {"rare", "repeated"}
    assert results[0].score < results[1].score < 0
    assert [(result.retriever, result.rank) for result in results] == [("bm25", 1), ("bm25", 2)]
    await store.close()


# 功能：验证 lexical SQL 只返回 global 和当前 project 的匹配行
# 设计：同一 query 同时写入 global、当前项目和跨项目行，隔离验证存储层 where 条件
@pytest.mark.asyncio
async def test_bm25_applies_user_and_project_scope_in_sql() -> None:
    store = SqliteLexicalStore(":memory:")
    await store.replace_document(
        "docs",
        [
            _chunk("global", "docs", "多头注意力全局说明", user_id="user-a"),
            _chunk(
                "project-a",
                "docs",
                "多头注意力项目说明",
                scope="project",
                project_id="project-a",
                user_id="user-a",
            ),
            _chunk(
                "project-b",
                "docs",
                "多头注意力其他项目说明",
                scope="project",
                project_id="project-b",
                user_id="user-a",
            ),
            _chunk("other-user", "docs", "多头注意力他人内容", user_id="user-b"),
        ],
    )

    results = await BM25Retriever(store).retrieve(
        "多头注意力",
        top_k=8,
        filters=KnowledgeFilter(user_id="user-a", project_id="project-a"),
    )

    assert {result.chunk.id for result in results} == {"global", "project-a"}
    await store.close()


# 功能：验证一至二字 query 安全返回空结果，三字中文 query 可命中
# 设计：同一条已知中文记录测试 trigram 最小窗口边界，固定短 query fallback 契约
@pytest.mark.asyncio
async def test_bm25_short_query_boundary() -> None:
    store = SqliteLexicalStore(":memory:")
    await store.replace_document("doc-a", [_chunk("c1", "doc-a", "多头注意力拆分")])
    retriever = BM25Retriever(store)

    assert await retriever.retrieve("多", top_k=4, filters=KnowledgeFilter()) == []
    assert await retriever.retrieve("多头", top_k=4, filters=KnowledgeFilter()) == []
    assert await retriever.retrieve("多头注", top_k=4, filters=KnowledgeFilter())
    await store.close()


# 功能：验证较长自然语言 query 可通过重叠 trigram 命中相关子串
# 设计：查询句式与文档不同但共享核心概念，确保 lexical 检索不要求整句精确出现
@pytest.mark.asyncio
async def test_bm25_matches_trigrams_inside_long_query() -> None:
    store = SqliteLexicalStore(":memory:")
    await store.replace_document("doc-a", [_chunk("c1", "doc-a", "多头注意力机制如何拆分")])

    results = await BM25Retriever(store).retrieve(
        "如何拆分多头注意力的计算过程", top_k=4, filters=KnowledgeFilter()
    )

    assert results
    assert results[0].chunk.id == "c1"
    await store.close()


# 功能：验证 document-level 替换和删除不会留下旧 FTS rows
# 设计：重复替换后删除单个文档，再查询另一个文档内容检验幂等和删除边界
@pytest.mark.asyncio
async def test_lexical_document_replace_and_delete() -> None:
    store = SqliteLexicalStore(":memory:")
    await store.replace_document("doc-a", [_chunk("old", "doc-a", "多头注意力旧版本")])
    await store.replace_document("doc-a", [_chunk("new", "doc-a", "多头注意力新版本")])
    await store.replace_document("doc-b", [_chunk("keep", "doc-b", "多头注意力保留内容")])
    await store.delete_document("doc-a")

    results = await BM25Retriever(store).retrieve(
        "多头注意力", top_k=8, filters=KnowledgeFilter()
    )

    assert [result.chunk.id for result in results] == ["keep"]
    await store.close()
