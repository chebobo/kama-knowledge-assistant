"""RAG V1 检索基线测试：真实 Qdrant 检索，记录 Top-5 命中供人工评估。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeFilter
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
TOP_K = 5

# (问题, 预期命中的文件名关键词)
QUESTIONS = [
    ("Agent 混合路由优化是什么，为什么单模型走不通", "hybrid_routing"),
    ("ReAct 循环是怎么工作的", "chapter1"),
    ("RAG 检索增强生成的核心思路是什么", "rag"),
    ("级联降级是怎么做的，先小模型后大模型", "hybrid_routing"),
    ("Agent 的记忆系统有哪些类型", "memory"),
    ("Transformer 的注意力机制如何计算", "attention"),
    ("什么是模型蒸馏，有什么用", "distillation"),
    ("Claude Code 是怎么管理上下文窗口的", "context"),
    ("什么是 function calling 函数调用", "function_calling"),
    ("多 Agent 之间如何通信协作", "multi_agent"),
]


async def main() -> None:
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    filters = KnowledgeFilter(user_id="local")

    for q, expected in QUESTIONS:
        emb = (await embedding.embed([q]))[0]
        hits = await store.search(emb, top_k=TOP_K, filters=filters)
        print(f"\nQ: {q}")
        print(f"  预期来源: {expected}")
        for h in hits:
            src = Path(h.chunk.source).name
            sec = h.chunk.metadata.get("section", "") or "(frontmatter/无标题)"
            preview = h.chunk.content[:70].replace("\n", "⏎")
            print(f"    score={h.score:.3f}  src={src}")
            print(f"      section={sec[:36]!r}  content={preview!r}")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
