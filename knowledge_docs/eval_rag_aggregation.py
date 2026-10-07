"""独立评估 RRF 后的 document aggregation。

用法：
    uv run python knowledge_docs/eval_rag_aggregation.py 5 hybrid
    uv run python knowledge_docs/eval_rag_aggregation.py 8 hybrid

评测先取与 Hybrid baseline 相同的候选池（每路 2 * k），再做完整 RRF，
最后按 chunk.document_id 聚合为 document top-k。该脚本不改变既有
eval_rag.py 的 dense/hybrid 评测路径。
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeFilter, KnowledgeSearchResult
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever
from kama_claude.core.knowledge.retriever.dense import DenseRetriever
from kama_claude.core.knowledge.retriever.fusion import reciprocal_rank_fusion
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
LEXICAL_DB = Path.home() / ".kama" / "knowledge" / "lexical.db"
GOLDEN = json.loads(
    Path(r"D:\tinyclaude\knowledge_docs\golden_set.json").read_text(encoding="utf-8")
)

Aggregation = Literal["sum", "max", "first-seen"]


@dataclass(frozen=True)
class AggregatedDocument:
    """保存一个文档的代表 chunk、聚合分数和首次出现顺序。"""

    document_id: str
    representative: KnowledgeSearchResult
    score: float
    first_seen: int


def aggregate_documents(
    results: list[KnowledgeSearchResult],
    *,
    aggregation: Aggregation,
) -> list[AggregatedDocument]:
    """按 document_id 聚合已按 RRF 顺序排列的 chunk 结果。"""
    if aggregation not in {"sum", "max", "first-seen"}:
        raise ValueError("aggregation must be sum, max, or first-seen")

    grouped: dict[str, list[tuple[int, KnowledgeSearchResult]]] = {}
    for index, result in enumerate(results):
        grouped.setdefault(result.chunk.document_id, []).append((index, result))

    documents: list[AggregatedDocument] = []
    for document_id, chunks in grouped.items():
        first_seen, representative = chunks[0]
        scores = [result.score for _, result in chunks]
        if aggregation == "sum":
            score = sum(scores)
        elif aggregation == "max":
            score = max(scores)
        else:
            score = representative.score
        documents.append(
            AggregatedDocument(
                document_id=document_id,
                representative=representative,
                score=score,
                first_seen=first_seen,
            )
        )

    # 聚合分数相等时保留 RRF 首次出现顺序，保证跨次运行的排序稳定。
    return sorted(documents, key=lambda item: (-item.score, item.first_seen))


def dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(rels))


def rel_sequence(top_src: list[str], targets: set[str]) -> tuple[list[int], int, int]:
    """计算文档级相关性、首个命中位置和命中文档数量。"""
    rels: list[int] = []
    seen: set[str] = set()
    rank = 0
    matched = 0
    for source in top_src:
        if source in targets and source not in seen:
            rels.append(1)
            seen.add(source)
            matched += 1
            if rank == 0:
                rank = len(rels)
        else:
            rels.append(0)
    return rels, rank, matched


def metrics(
    rels: list[int],
    rank: int,
    *,
    target_count: int,
    matched_count: int,
    k: int,
) -> tuple[float, float, float, float]:
    query_hit = 1.0 if rank else 0.0
    recall = matched_count / target_count
    mrr = 1.0 / rank if rank else 0.0
    idcg = dcg([1] * min(target_count, k)) or 1.0
    ndcg = dcg(rels) / idcg
    return query_hit, recall, mrr, ndcg


async def main(k: int, mode: str) -> None:
    if k <= 0:
        raise SystemExit("k must be positive")
    if mode not in {"dense", "hybrid"}:
        raise SystemExit("retriever mode must be 'dense' or 'hybrid'")

    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    dense = DenseRetriever(embedding=embedding, store=store)
    lexical_store = SqliteLexicalStore(LEXICAL_DB) if mode == "hybrid" else None
    bm25 = BM25Retriever(lexical_store) if lexical_store is not None else None
    filters = KnowledgeFilter(user_id="local")
    candidate_k = k * 2
    print(
        f"run_at={datetime.now(UTC).isoformat()} mode={mode} k={k} "
        f"candidate_k={candidate_k} rrf_k=60 "
        f"qdrant={QDRANT_DIR} lexical={LEXICAL_DB if mode == 'hybrid' else 'disabled'}"
    )

    strategies = ("chunk-top-k", "sum", "max", "first-seen")
    per_method: dict[str, list[dict[str, float | int | str | list[str]]]] = {
        strategy: [] for strategy in strategies
    }
    for question in GOLDEN["questions"]:
        dense_hits = await dense.retrieve(
            question["question"], top_k=candidate_k, filters=filters
        )
        if bm25 is None:
            fused = dense_hits
        else:
            bm25_hits = await bm25.retrieve(
                question["question"], top_k=candidate_k, filters=filters
            )
            # 不提前截断融合结果，确保 document aggregation 能看到同一候选池中的所有文档。
            fused = reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=60)

        for strategy in per_method:
            if strategy == "chunk-top-k":
                top_src = [Path(item.chunk.source).name for item in fused[:k]]
            else:
                documents = aggregate_documents(fused, aggregation=strategy)[:k]
                top_src = [
                    Path(item.representative.chunk.source).name for item in documents
                ]
            targets_single = {question["expected_doc"]}
            targets_multi = set(question.get("related_docs", [question["expected_doc"]]))
            rels_s, rank_s, matched_s = rel_sequence(top_src, targets_single)
            rels_m, rank_m, matched_m = rel_sequence(top_src, targets_multi)
            values_s = metrics(
                rels_s,
                rank_s,
                target_count=len(targets_single),
                matched_count=matched_s,
                k=k,
            )
            values_m = metrics(
                rels_m,
                rank_m,
                target_count=len(targets_multi),
                matched_count=matched_m,
                k=k,
            )
            per_method[strategy].append(
                {
                    "qid": question["qid"],
                    "hit_s": values_s[0],
                    "recall_s": values_s[1],
                    "mrr_s": values_s[2],
                    "ndcg_s": values_s[3],
                    "hit_m": values_m[0],
                    "recall_m": values_m[1],
                    "mrr_m": values_m[2],
                    "ndcg_m": values_m[3],
                    "top_src": top_src,
                }
            )

    for strategy, rows in per_method.items():
        def avg(key: str) -> float:
            return sum(float(row[key]) for row in rows) / len(rows)

        print(f"\n=== {strategy} (k={k}, {len(rows)} 问) ===")
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
        q9 = next(row for row in rows if row["qid"] == 9)
        print(f"Q9 top{k} results: {q9['top_src']}")

    store.close()
    if lexical_store is not None:
        await lexical_store.close()


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    asyncio.run(main(k, mode))
