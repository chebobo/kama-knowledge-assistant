# -*- coding: utf-8 -*-
"""Q18 prompt v2 手工验证：禁止实体、强制主题词，验证改写后能否命中目标文档。"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kama_claude.core.config import get_config  # noqa: E402
from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider  # noqa: E402
from kama_claude.core.knowledge.models import KnowledgeFilter  # noqa: E402
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever  # noqa: E402
from kama_claude.core.knowledge.retriever.dense import DenseRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.hybrid import HybridRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore  # noqa: E402
from kama_claude.core.knowledge.rewrite import QueryRewriter  # noqa: E402
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402

TARGET = "agent_framework_comparison.md"

PROMPT_V2 = """你是一个面向本地技术知识库的检索查询改写助手。
你的任务是将用户的原始问题转换为更适合向量检索和关键词检索的查询表达。

改写规则：
- 保留原问题的核心意图、问题类型和关键约束，不得改变用户实际想问的问题。
- 原问题中明确出现的具体实体（框架、产品、技术、概念名称）必须原样保留。
- 对模糊的类别表达（如"主流""框架""异同""对比"等检索价值低的泛词），直接用主题维度词替换或补充：如"架构设计、记忆机制、工具调用、上下文管理、安全机制、适用场景、优劣对比"等。
- 输出应是一组检索价值高的关键词组合，可以精简原问题中的泛化措辞，不必逐字保留原问题。
- 禁止引入原问题未提及的任何具体框架、产品、工具或技术名称。
- 不回答用户的问题，不添加解释、结论、推荐或事实性断言。
- 输出一条简洁、完整的查询语句，不输出标题、引号、推理过程或其他说明。

原始问题：{query}
改写后的查询：
"""


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

    question = "主流 Agent 框架有什么异同"

    async def run() -> None:
        config = get_config()
        rw = QueryRewriter()  # 复用 provider/client，但绕过模块级 PROMPT
        provider = rw._provider
        if provider is None:
            from kama_claude.core.llm.provider import AnthropicProvider

            provider = AnthropicProvider(rw.model)
            rw._provider = provider
        try:
            resp = await provider.complete_text(
                PROMPT_V2.format(query=question), max_tokens=1024, temperature=0.0
            )
            raw = resp.text.strip().strip('"“”').strip()
            print("[prompt v2 原始输出]")
            print(raw)
            print("\n[诊断]", json.dumps({
                "stop_reason": resp.stop_reason,
                "output_tokens": resp.usage.output_tokens if resp.usage else None,
                "thinking_chars": sum(len(b.get("thinking", "")) for b in resp.thinking_blocks),
            }, ensure_ascii=False, indent=2))

            if not raw:
                print("\n!! 空输出")
                return
            hits = await hybrid.retrieve(raw, top_k=8, filters=KnowledgeFilter(user_id="local"))
            srcs = [PureWindowsPath(h.chunk.source).name for h in hits]
            print("\n[改写后检索] top8:")
            for s in srcs:
                print("  -", s)
            print("\ntarget in top5:", TARGET in srcs[:5])
        finally:
            await rw.aclose()
            store.close()
            await lexical.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
