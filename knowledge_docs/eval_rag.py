"""RAG 指标化评估（双口径）：加载 golden_set.json，计算 Recall@k / MRR@k / NDCG@k。

匹配规则：document 级（chunk 的 source 文件名 == 目标文档）。
- single（主指标）：命中 expected_doc 才 rel=1
- multi（辅助观测）：命中任一 related_docs 即 rel=1（仅对标注了 related_docs 的题生效，未标注题两口径一致）
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
    # rels: 每个位置的二值相关性（1=命中任一目标文档）
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def rel_sequence(top_src: list[str], targets: set[str]) -> tuple[list[int], int]:
    """按目标集合计算 rel 序列与首个命中 rank（0 表示未命中）。
    文档级去重：同一目标文档的后续 chunk 不再重复计 rel=1（每篇最多贡献一个 1）。"""
    rels: list[int] = []
    seen: set[str] = set()
    rank = 0
    for src in top_src:
        if src in targets and src not in seen:
            rels.append(1)
            seen.add(src)
            if rank == 0:
                rank = len(rels)
        else:
            rels.append(0)
    return rels, rank


async def main(k: int) -> None:
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    filters = KnowledgeFilter(user_id="local")

    per_q = []
    for q in GOLDEN["questions"]:
        targets_single = {q["expected_doc"]}
        targets_multi = set(q.get("related_docs", [q["expected_doc"]]))
        emb = (await embedding.embed([q["question"]]))[0]
        hits = await store.search(emb, top_k=k, filters=filters)
        top_src = [Path(h.chunk.source).name for h in hits]

        rels_s, rank_s = rel_sequence(top_src, targets_single)
        rels_m, rank_m = rel_sequence(top_src, targets_multi)

        def metrics(rels, rank):
            recall = 1.0 if rank else 0.0
            mrr = 1.0 / rank if rank else 0.0
            # multi 口径下可能多个 rel=1：NDCG 的 IDCG = 相关文档数完美排序的增益
            idcg = dcg([1] * sum(rels)) or 1.0
            ndcg = dcg(rels) / idcg
            return recall, mrr, ndcg

        r_s = metrics(rels_s, rank_s)
        r_m = metrics(rels_m, rank_m)
        per_q.append({
            "qid": q["qid"], "expected": q["expected_doc"],
            "related": sorted(targets_multi),
            "rank_s": rank_s, "rank_m": rank_m,
            "recall_s": r_s[0], "mrr_s": r_s[1], "ndcg_s": r_s[2],
            "recall_m": r_m[0], "mrr_m": r_m[1], "ndcg_m": r_m[2],
            "top_src": top_src,
        })

        # 两口径判定差异时打特殊标记
        mark_s = "OK " if rank_s else "MISS"
        mark_m = "OK " if rank_m else "MISS"
        diff = "  <-multi修正" if rank_m and not rank_s else ""
        multi_note = f"  multi: rank={rank_m:>2} {mark_m}" if len(targets_multi) > 1 else ""
        print(f"[{mark_s}] Q{q['qid']:>2} single rank={rank_s:>2}  expected={q['expected_doc']}{multi_note}{diff}")
        if not rank_s:
            print(f"        top{k}={top_src[:k]}")

    n = len(per_q)

    def avg(key):
        return sum(x[key] for x in per_q) / n

    print(f"\n=== 汇总 (k={k}, {n} 问, document 级) ===")
    print(f"[single 主指标] Recall@{k}: {avg('recall_s'):.3f} ({sum(x['recall_s'] for x in per_q)}/{n})  MRR: {avg('mrr_s'):.3f}  NDCG: {avg('ndcg_s'):.3f}")
    print(f"[multi  辅助]  Recall@{k}: {avg('recall_m'):.3f} ({sum(x['recall_m'] for x in per_q)}/{n})  MRR: {avg('mrr_m'):.3f}  NDCG: {avg('ndcg_m'):.3f}")
    store.close()


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(main(k))
