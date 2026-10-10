# -*- coding: utf-8 -*-
"""Q18 深挖：多个 query 变体对比 Dense/BM25 候选池与 hybrid top8 是否含目标文档。
定位失败环节：query 表达 vs 检索实现 vs 文档内容。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider  # noqa: E402
from kama_claude.core.knowledge.models import KnowledgeFilter  # noqa: E402
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever  # noqa: E402
from kama_claude.core.knowledge.retriever.dense import DenseRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.hybrid import HybridRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore  # noqa: E402
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402

TARGET = "agent_framework_comparison.md"

QUERIES = [
    ("original", "主流 Agent 框架有什么异同"),
    ("rewritten", "主流 AI Agent 框架（LangChain、LlamaIndex、AutoGPT、AutoGen、MetaGPT、CrewAI 等）异同对比 架构差异 设计特点"),
    ("doc_keywords", "OpenClaw Hermes Agent Claude Code 三框架 横评 对比"),
    ("framework_compare", "框架对比 架构差异 设计特点 记忆机制 工具调用 上下文管理"),
    ("generic_framework", "Agent 框架有哪些 区别"),
]


def main() -> None:
    home = Path.home()
    model_root = home / ".cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
    if (model_root / "snapshots").is_dir():
        snapshots = sorted((model_root / "snapshots").iterdir())
        if len(snapshots) != 1:
            raise ValueError("select a specific embedding snapshot")
        model_root = snapshots[0]
    qdrant_path = home / ".kama/knowledge/qdrant"
    lexical_path = home / ".kama/knowledge/lexical.db"
    collection = "kama_knowledge"

    embedding = LocalEmbeddingProvider(model_root)
    store = QdrantVectorStore(qdrant_path, dimension=embedding.dimension, collection=collection)
    lexical = SqliteLexicalStore(lexical_path)
    dense = DenseRetriever(embedding=embedding, store=store)
    bm25 = BM25Retriever(lexical)
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, candidate_multiplier=2, rrf_k=60, document_aggregation="sum"
    )

    async def run() -> None:
        for label, query in QUERIES:
            d_hits = await dense.retrieve(query, top_k=16, filters=KnowledgeFilter(user_id="local"))
            b_hits = await bm25.retrieve(query, top_k=16, filters=KnowledgeFilter(user_id="local"))
            h_hits = await hybrid.retrieve(query, top_k=8, filters=KnowledgeFilter(user_id="local"))
            d_srcs = [PureWindowsPath(h.chunk.source).name for h in d_hits]
            b_srcs = [PureWindowsPath(h.chunk.source).name for h in b_hits]
            h_srcs = [PureWindowsPath(h.chunk.source).name for h in h_hits]
            print(f"\n=== [{label}] {query[:40]}... ===")
            print(f"  dense top16 含目标: {TARGET in d_srcs}  (rank={d_srcs.index(TARGET) + 1 if TARGET in d_srcs else '-'})")
            print(f"  bm25  top16 含目标: {TARGET in b_srcs}  (rank={b_srcs.index(TARGET) + 1 if TARGET in b_srcs else '-'})")
            print(f"  hybrid top8 含目标: {TARGET in h_srcs}  (rank={h_srcs.index(TARGET) + 1 if TARGET in h_srcs else '-'})")
            print(f"  hybrid top8: {h_srcs[:5]}")
            if TARGET not in d_srcs:
                print(f"  dense top5: {d_srcs[:5]}")
            if TARGET not in b_srcs:
                print(f"  bm25 top5: {b_srcs[:5]}")
        store.close()
        await lexical.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
