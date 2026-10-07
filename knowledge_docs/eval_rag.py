"""RAG 指标化评估（双口径）：加载 golden_set.json，计算 Recall@k / MRR@k / NDCG@k。

匹配规则：document 级（chunk 的 source 文件名 == 目标文档）。
- single（主指标）：命中 expected_doc 才 rel=1
- multi（辅助观测）：命中任一 related_docs 即 rel=1；未标注题两口径一致
用法：python eval_rag.py [k]
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeFilter
from kama_claude.core.knowledge.retriever.base import Retriever
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever
from kama_claude.core.knowledge.retriever.dense import DenseRetriever
from kama_claude.core.knowledge.retriever.hybrid import HybridRetriever
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
LEXICAL_DB = Path.home() / ".kama" / "knowledge" / "lexical.db"
GOLDEN = json.loads(
    Path(r"D:\tinyclaude\knowledge_docs\golden_set.json").read_text(encoding="utf-8")
)


def dcg(rels: list[int]) -> float:
    # rels: 每个位置的二值相关性（1=命中任一目标文档）
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def rel_sequence(top_src: list[str], targets: set[str]) -> tuple[list[int], int, int]:
    """按目标集合计算 rel 序列、首个命中 rank 和相关文档命中数。"""
    rels: list[int] = []
    seen: set[str] = set()
    rank = 0
    matched = 0
    for src in top_src:
        if src in targets and src not in seen:
            rels.append(1)
            seen.add(src)
            matched += 1
            if rank == 0:
                rank = len(rels)
        else:
            rels.append(0)
    return rels, rank, matched


async def main(k: int, mode: str) -> None:
    if mode not in {"dense", "hybrid"}:
        raise SystemExit("retriever mode must be 'dense' or 'hybrid'")
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    dense = DenseRetriever(embedding=embedding, store=store)
    lexical_store = None
    retriever: Retriever = dense
    if mode == "hybrid":
        lexical_store = SqliteLexicalStore(LEXICAL_DB)
        retriever = HybridRetriever(dense=dense, bm25=BM25Retriever(lexical_store))
    filters = KnowledgeFilter(user_id="local")
    print(
        f"run_at={datetime.now(UTC).isoformat()} mode={mode} k={k} "
        f"qdrant={QDRANT_DIR} lexical={LEXICAL_DB if mode == 'hybrid' else 'disabled'}"
    )

    per_q = []
    for q in GOLDEN["questions"]:
        targets_single = {q["expected_doc"]}
        targets_multi = set(q.get("related_docs", [q["expected_doc"]]))
        hits = await retriever.retrieve(q["question"], top_k=k, filters=filters)
        top_src = [Path(h.chunk.source).name for h in hits]

        rels_s, rank_s, matched_s = rel_sequence(top_src, targets_single)
        rels_m, rank_m, matched_m = rel_sequence(top_src, targets_multi)

        def metrics(rels, rank, target_count, matched_count):
            query_hit = 1.0 if rank else 0.0
            recall = matched_count / target_count
            mrr = 1.0 / rank if rank else 0.0
            # IDCG 使用评测集合中的相关文档数，而不是当前结果中已命中的数量
            idcg = dcg([1] * min(target_count, k)) or 1.0
            ndcg = dcg(rels) / idcg
            return query_hit, recall, mrr, ndcg

        r_s = metrics(rels_s, rank_s, len(targets_single), matched_s)
        r_m = metrics(rels_m, rank_m, len(targets_multi), matched_m)
        per_q.append({
            "qid": q["qid"], "expected": q["expected_doc"],
            "related": sorted(targets_multi),
            "rank_s": rank_s, "rank_m": rank_m,
            "hit_s": r_s[0], "recall_s": r_s[1], "mrr_s": r_s[2], "ndcg_s": r_s[3],
            "hit_m": r_m[0], "recall_m": r_m[1], "mrr_m": r_m[2], "ndcg_m": r_m[3],
            "top_src": top_src,
        })

        # 两口径判定差异时打特殊标记
        mark_s = "OK " if rank_s else "MISS"
        mark_m = "OK " if rank_m else "MISS"
        diff = "  <-multi修正" if rank_m and not rank_s else ""
        multi_note = f"  multi: rank={rank_m:>2} {mark_m}" if len(targets_multi) > 1 else ""
        print(
            f"[{mark_s}] Q{q['qid']:>2} single rank={rank_s:>2} "
            f"expected={q['expected_doc']}{multi_note}{diff}"
        )
        if not rank_s:
            print(f"        top{k}={top_src[:k]}")

    n = len(per_q)

    def avg(key):
        return sum(x[key] for x in per_q) / n

    print(f"\n=== 汇总 (k={k}, {n} 问, document 级) ===")
    print(
        f"[single 主指标] Query Hit Rate@{k}: {avg('hit_s'):.3f}  "
        f"Recall@{k}: {avg('recall_s'):.3f}  MRR: {avg('mrr_s'):.3f}  "
        f"NDCG: {avg('ndcg_s'):.3f}"
    )
    print(
        f"[multi  辅助]  Query Hit Rate@{k}: {avg('hit_m'):.3f}  "
        f"Recall@{k}: {avg('recall_m'):.3f}  MRR: {avg('mrr_m'):.3f}  "
        f"NDCG: {avg('ndcg_m'):.3f}"
    )
    store.close()
    if lexical_store is not None:
        await lexical_store.close()


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mode = sys.argv[2] if len(sys.argv) > 2 else "dense"
    asyncio.run(main(k, mode))
