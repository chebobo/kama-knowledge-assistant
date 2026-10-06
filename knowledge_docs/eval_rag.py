"""RAG 指标化评估：加载 golden_set.json，计算 Recall@k / MRR@k / NDCG@k。

匹配规则：document 级（chunk 的 source 文件名 == expected_doc）。
用法：python eval_rag.py [k]
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeFilter
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
GOLDEN = json.loads(
    Path(r"D:\tinyclaude\knowledge_docs\golden_set.json").read_text(encoding="utf-8")
)


def dcg(rels: list[int]) -> float:
    # rels: 每个位置的二值相关性（1=命中预期文档）
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


async def main(k: int) -> None:
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    filters = KnowledgeFilter(user_id="local")

    per_q = []
    for q in GOLDEN["questions"]:
        expected = q["expected_doc"]
        emb = (await embedding.embed([q["question"]]))[0]
        hits = await store.search(emb, top_k=k, filters=filters)
        top_src = [Path(h.chunk.source).name for h in hits]
        # 文档级去重：预期文档只按首次出现计 rel=1，后续同文档 chunk 不计
        rels: list[int] = []
        seen = False
        for src in top_src:
            if src == expected and not seen:
                rels.append(1)
                seen = True
            else:
                rels.append(0)
        rank = rels.index(1) + 1 if 1 in rels else 0
        recall = 1.0 if rank else 0.0
        mrr = 1.0 / rank if rank else 0.0
        ndcg = dcg(rels)  # 去重后最多一个 rel=1，IDCG=1，NDCG ≤ 1
        per_q.append({
            "qid": q["qid"], "expected": expected, "rank": rank,
            "recall@k": recall, "mrr": mrr, "ndcg": ndcg,
            "top_src": top_src,
        })
        mark = "OK " if rank else "MISS"
        detail = f"top5={top_src[:5]}" if not rank else ""
        print(f"[{mark}] Q{q['qid']:>2} rank={rank:>2}  ndcg={ndcg:.3f}  expected={expected} {detail}")

    n = len(per_q)
    avg_recall = sum(x["recall@k"] for x in per_q) / n
    avg_mrr = sum(x["mrr"] for x in per_q) / n
    avg_ndcg = sum(x["ndcg"] for x in per_q) / n
    print(f"\n=== 汇总 (k={k}, {n} 问, document 级匹配) ===")
    print(f"Recall@{k} : {avg_recall:.3f}  ({sum(x['recall@k'] for x in per_q)}/{n})")
    print(f"MRR@{k}    : {avg_mrr:.3f}")
    print(f"NDCG@{k}   : {avg_ndcg:.3f}")
    store.close()


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(main(k))
