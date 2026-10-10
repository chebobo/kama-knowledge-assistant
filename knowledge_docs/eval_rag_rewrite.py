r"""第一阶段 Query Rewrite 评测，默认关闭且不调用 LLM。

    .venv\Scripts\python.exe knowledge_docs/eval_rag_rewrite.py
    .venv\Scripts\python.exe knowledge_docs/eval_rag_rewrite.py --rewrite-enabled
    .venv\Scripts\python.exe knowledge_docs/eval_rag_rewrite.py --rewrite-enabled --c-enabled

检索使用生产 HybridRetriever；旧评测入口保持原样。结果写入 --output，
改写缓存仅在模型、prompt、参数和问题完全匹配时复用；--refresh-cache 显式重生成。
C 组（--c-enabled）在原查询 + 改写查询双路 Hybrid 后按 document_id 分数相加融合。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kama_claude.core.config import get_config  # noqa: E402
from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider  # noqa: E402
from kama_claude.core.knowledge.models import KnowledgeFilter  # noqa: E402
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever  # noqa: E402
from kama_claude.core.knowledge.retriever.dense import DenseRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.hybrid import HybridRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore  # noqa: E402
from kama_claude.core.knowledge.rewrite import (  # noqa: E402
    MAX_TOKENS,
    PROMPT,
    PROMPT_VERSION,
    TEMPERATURE,
    QueryRewriter,
    RewriteResult,
)
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def cache_metadata(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout_budget_s": 15.0,
    }


async def select_query(
    query: str, *, rewrite_enabled: bool = False, rewriter: Any = None
) -> tuple[str, RewriteResult, float]:
    """本轮唯一调用方编排路径：默认关闭，不创建或调用 rewriter。"""
    if not rewrite_enabled:
        return query, RewriteResult(None, "fallback", "disabled"), 0.0
    if rewriter is None:
        raise ValueError("enabled rewriting requires a rewriter")
    start = time.perf_counter()
    result = await rewriter.rewrite(query)
    effective = result.rewritten_query if result.status == "success" else None
    return effective or query, result, (time.perf_counter() - start) * 1000


def question_metrics(ids: list[str], targets: set[str], k: int) -> dict[str, float | int]:
    # 去重后截断，相关性严格按 document_id。
    ids = list(dict.fromkeys(ids))[:k]
    rels = [int(identifier in targets) for identifier in ids]
    matched = sum(rels)
    rank = next((i for i, rel in enumerate(rels, 1) if rel), 0)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(targets))))
    return {
        "hit": float(bool(matched)),
        "recall": matched / len(targets),
        "mrr": 1 / rank if rank else 0.0,
        "ndcg": dcg / idcg,
        "matched": matched,
        "target_count": len(targets),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for cutoff in (5, 8):
        for label in ("s", "m"):
            values = [row[f"metrics_{label}@{cutoff}"] for row in rows]
            result[f"{'single' if label == 's' else 'multi'}@{cutoff}"] = {
                key: sum(item[key] for item in values) / len(values)
                for key in ("hit", "recall", "mrr", "ndcg")
            }
            result[f"{'single' if label == 's' else 'multi'}@{cutoff}"]["micro_recall"] = sum(
                item["matched"] for item in values
            ) / sum(item["target_count"] for item in values)
    return result


def fuse_documents(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """文档级分数相加融合：同一 document_id 的 RRF 分数累加，否则保留单路分数。

    两路分数同为 Hybrid（RRF + sum 文档聚合）输出，量纲一致可直接相加。
    排序键为 (-score, 首次出现顺序)，与 aggregate_document_results 语义一致。
    """
    merged: dict[str, dict[str, Any]] = {}
    order = 0
    for rows in (rows_a, rows_b):
        for item in rows:
            identifier = item["document_id"]
            if identifier in merged:
                merged[identifier]["score"] += item["score"]
            else:
                merged[identifier] = {
                    "document_id": identifier,
                    "source": item["source"],
                    "score": item["score"],
                    "_order": order,
                }
                order += 1
    ordered = sorted(merged.values(), key=lambda item: (-item["score"], item["_order"]))[:top_k]
    for item in ordered:
        item.pop("_order", None)
    return ordered


async def retrieve_row_dual(
    question: dict,
    group: str,
    orig_query: str,
    rewrite_query: str,
    result: RewriteResult,
    latency: float,
    hybrid: Any,
    dense: Any,
    bm25: Any,
    targets: tuple[set[str], set[str]],
    *,
    top_k: int = 8,
) -> dict:
    """C 组：原查询与改写查询各自走生产 Hybrid，再文档级分数相加融合。"""
    hits_a = await hybrid.retrieve(orig_query, top_k=top_k, filters=KnowledgeFilter(user_id="local"))
    hits_b = await hybrid.retrieve(rewrite_query, top_k=top_k, filters=KnowledgeFilter(user_id="local"))
    if dense.error or bm25.error:
        raise RuntimeError("retrieval source failed; C evaluation is invalid")
    documents_a = [
        {"document_id": hit.chunk.document_id, "source": PureWindowsPath(hit.chunk.source).name, "score": hit.score}
        for hit in hits_a
    ]
    documents_b = [
        {"document_id": hit.chunk.document_id, "source": PureWindowsPath(hit.chunk.source).name, "score": hit.score}
        for hit in hits_b
    ]
    fused = fuse_documents(documents_a, documents_b, top_k)
    ids = [item["document_id"] for item in fused]
    row = {
        "group": group,
        "query_id": question["qid"],
        "original_query": question["question"],
        "rewritten_query": result.rewritten_query,
        "effective_query": rewrite_query,
        "rewrite_status": "disabled" if result.fallback_reason == "disabled" else result.status,
        "fallback_reason": result.fallback_reason,
        "retrieval_mode": "hybrid-sum-orig+rewrite",
        "retrieved_documents": fused,
        "rewrite_latency_ms": latency,
        "llm_tokens": {"input": result.input_tokens, "output": result.output_tokens},
        "candidate_pools": {
            source: [
                {"document_id": hit.chunk.document_id, "chunk_id": hit.chunk.id, "source": PureWindowsPath(hit.chunk.source).name}
                for hit in recorder.hits
            ]
            for source, recorder in (("dense", dense), ("bm25", bm25))
        },
    }
    for k in (5, 8):
        for label, relevant in zip(("s", "m"), targets, strict=True):
            values = question_metrics(ids, relevant, k)
            row[f"metrics_{label}@{k}"] = values
            if k == 5:
                row.update({f"{key}_{label}": values[key] for key in ("hit", "recall", "mrr", "ndcg")})
    return row


def c_acceptance(baseline: list[dict], c_rows: list[dict], questions: list[dict]) -> dict:
    """C 组验收：单文档命中不降、多文档每题召回不降、Q18 命中 top5。"""
    c_summary = summarize(c_rows)
    q18 = next(row for row in c_rows if row["query_id"] == 18)
    return {
        "single_hit_at_least_095": c_summary["single@5"]["hit"] >= 0.95,
        "preserve_baseline_hits": all(
            a["hit_s"] <= row["hit_s"] for a, row in zip(baseline, c_rows, strict=True)
        ),
        "multi_each_question_non_decreasing": all(
            a["hit_m"] <= row["hit_m"] and a["recall_m"] <= row["recall_m"]
            for q, a, row in zip(questions, baseline, c_rows, strict=True)
            if q.get("related_docs")
        ),
        "multi_micro_recall_non_decreasing": (
            c_summary["multi@5"]["micro_recall"] >= summarize(baseline)["multi@5"]["micro_recall"]
        ),
        "Q18_success_top5": q18["rewrite_status"] == "success"
        and any(item["source"] == "agent_framework_comparison.md" for item in q18["retrieved_documents"][:5]),
    }


class RecordingRetriever:
    """只记录生产候选和错误；禁止静默单路降级进入评测结论。"""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.hits: list[Any] = []
        self.error: str | None = None

    async def retrieve(self, query: str, *, top_k: int, filters: KnowledgeFilter) -> list[Any]:
        self.error = None
        try:
            self.hits = await self.retriever.retrieve(query, top_k=top_k, filters=filters)
            return self.hits
        except Exception as exc:
            self.error = type(exc).__name__
            raise


class InjectedClient:
    """走真实 QueryRewriter + Provider 解析/超时路径，不依赖线上故障。"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls = 0
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.kind == "timeout":
            await asyncio.sleep(60)
        if self.kind == "exception":
            raise RuntimeError("injected")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="  ")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


class CachedRewriter:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    async def rewrite(self, query: str) -> RewriteResult:
        if query != self.row["original_query"]:
            raise ValueError("cache query mismatch")
        return RewriteResult(**self.row["result"])


def fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != ".lock":
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def snapshot_indices(source_qdrant: Path, source_lexical: Path, target: Path) -> None:
    """使用同一独立快照评测各组；复跑复用快照，不改个人索引。"""
    if target.exists():
        if not (target / "manifest.json").exists():
            raise ValueError("incomplete index snapshot; choose a fresh --output directory")
        return
    if not (source_qdrant / "meta.json").is_file() or not source_lexical.is_file():
        raise FileNotFoundError("evaluation indices are missing")
    before = fingerprint(source_qdrant)
    target.mkdir(parents=True)
    shutil.copytree(source_qdrant, target / "qdrant", ignore=shutil.ignore_patterns(".lock"))
    # SQLite backup API 包含 WAL 的已提交数据，避免裸复制遗漏数据。
    with sqlite3.connect(source_lexical.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(target / "lexical.db") as destination:
            source.backup(destination)
    if before != fingerprint(source_qdrant):
        raise RuntimeError("Qdrant source changed during snapshot; choose fresh output and retry")
    write_json(target / "manifest.json", {"created_at": now(), "source_sha256": before})


def label_map(qdrant_path: Path, lexical_path: Path, collection: str) -> tuple[dict, dict]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))
    chunks = {}
    names: dict[str, set[str]] = {}
    try:
        info = client.get_collection(collection)
        offset = None
        while True:
            points, offset = client.scroll(
                collection, offset=offset, limit=256, with_payload=True, with_vectors=False
            )
            for point in points:
                payload = point.payload or {}
                identifier = payload["document_id"]
                chunks[payload["chunk_id"]] = identifier
                names.setdefault(PureWindowsPath(payload["source"]).name, set()).add(identifier)
            if offset is None:
                break
    finally:
        client.close()
    with sqlite3.connect(lexical_path.as_uri() + "?mode=ro", uri=True) as db:
        lexical_chunks = dict(db.execute("SELECT chunk_id, document_id FROM knowledge_chunks_fts"))
    if chunks != lexical_chunks:
        raise ValueError("Dense/BM25 snapshots do not contain the same chunks/documents")
    vectors = info.config.params.vectors
    return names, {
        "chunks": len(chunks),
        "documents": len(set(chunks.values())),
        "dimension": vectors.size,
    }


def resolve_targets(question: dict, names: dict) -> tuple[set[str], set[str]]:
    def lookup(name: str) -> str:
        ids = names.get(name, set())
        if len(ids) != 1:
            raise ValueError(f"golden label missing or ambiguous: {name}")
        return next(iter(ids))

    primary = {lookup(question["expected_doc"])}
    multi = primary | {lookup(name) for name in question.get("related_docs", [])}
    return primary, multi


async def retrieve_row(
    question: dict,
    group: str,
    query: str,
    result: RewriteResult,
    latency: float,
    hybrid: Any,
    dense: Any,
    bm25: Any,
    targets: tuple[set[str], set[str]],
) -> dict:
    hits = await hybrid.retrieve(query, top_k=8, filters=KnowledgeFilter(user_id="local"))
    if dense.error or bm25.error:
        raise RuntimeError("retrieval source failed; A/B evaluation is invalid")
    documents = [
        {
            "document_id": hit.chunk.document_id,
            "source": PureWindowsPath(hit.chunk.source).name,
            "score": hit.score,
        }
        for hit in hits
    ]
    ids = [hit.chunk.document_id for hit in hits]
    row = {
        "group": group,
        "query_id": question["qid"],
        "original_query": question["question"],
        "rewritten_query": result.rewritten_query,
        "effective_query": query,
        "rewrite_status": "disabled" if result.fallback_reason == "disabled" else result.status,
        "fallback_reason": result.fallback_reason,
        "retrieval_mode": "hybrid-sum",
        "retrieved_documents": documents,
        "rewrite_latency_ms": latency,
        "llm_tokens": {"input": result.input_tokens, "output": result.output_tokens},
        "candidate_pools": {
            source: [
                {
                    "document_id": hit.chunk.document_id,
                    "chunk_id": hit.chunk.id,
                    "source": PureWindowsPath(hit.chunk.source).name,
                }
                for hit in recorder.hits
            ]
            for source, recorder in (("dense", dense), ("bm25", bm25))
        },
    }
    for k in (5, 8):
        for label, relevant in zip(("s", "m"), targets, strict=True):
            values = question_metrics(ids, relevant, k)
            row[f"metrics_{label}@{k}"] = values
            if k == 5:
                row.update(
                    {f"{key}_{label}": values[key] for key in ("hit", "recall", "mrr", "ndcg")}
                )
    return row


def acceptance(
    groups: dict[str, list[dict]], questions: list[dict], stability: bool, d1_calls: int, d2: dict
) -> dict:
    baseline = groups["A"]
    checks = {
        "D1_zero_llm_calls": d1_calls == 0,
        "D1_equals_A": all(
            a["retrieved_documents"] == d["retrieved_documents"]
            for a, d in zip(baseline, groups["D1"], strict=True)
        ),
        "D2_fallback_status_query_results": all(d2.values()),
    }
    if "B" not in groups:
        return {"checks": checks, "ab_status": "not_run", "passed": None}
    b = groups["B"]
    a_summary, b_summary = summarize(baseline), summarize(b)
    q18 = next(row for row in b if row["query_id"] == 18)
    checks.update(
        {
            "Q18_success_top5": q18["rewrite_status"] == "success"
            and any(
                item["source"] == "agent_framework_comparison.md"
                for item in q18["retrieved_documents"][:5]
            ),
            "single_hit_at_least_095": b_summary["single@5"]["hit"] >= 0.95,
            "preserve_baseline_hits": all(
                a["hit_s"] <= row["hit_s"] for a, row in zip(baseline, b, strict=True)
            ),
            "single_mrr_non_decreasing": b_summary["single@5"]["mrr"]
            >= a_summary["single@5"]["mrr"],
            "single_ndcg_non_decreasing": b_summary["single@5"]["ndcg"]
            >= a_summary["single@5"]["ndcg"],
            "multi_each_question_non_decreasing": all(
                a["hit_m"] <= row["hit_m"] and a["recall_m"] <= row["recall_m"]
                for q, a, row in zip(questions, baseline, b, strict=True)
                if q.get("related_docs")
            ),
            "multi_micro_recall_non_decreasing": (
                b_summary["multi@5"]["micro_recall"] >= a_summary["multi@5"]["micro_recall"]
            ),
            "fixed_rewrite_retrieval_stable": stability,
        }
    )
    return {"checks": checks, "ab_status": "completed", "passed": all(checks.values())}


async def run(args: argparse.Namespace) -> dict:
    config = get_config()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    questions = golden["questions"]
    if len(questions) != 20 or len({q["qid"] for q in questions}) != 20:
        raise ValueError("this acceptance suite requires all 20 unique golden questions")
    model_root = args.embedding.expanduser().resolve()
    if (model_root / "snapshots").is_dir():
        snapshots = sorted((model_root / "snapshots").iterdir())
        if len(snapshots) != 1:
            raise ValueError("select a specific embedding snapshot with --embedding")
        model_root = snapshots[0]
    snapshot = output / "index_snapshot"
    snapshot_indices(
        args.qdrant.expanduser().resolve(), args.lexical.expanduser().resolve(), snapshot
    )
    # manifest 自身不在指纹中，Qdrant Local 的锁文件忽略。
    index_hash = fingerprint(snapshot)
    names, counts = label_map(snapshot / "qdrant", snapshot / "lexical.db", args.collection)
    targets = [resolve_targets(q, names) for q in questions]
    embedding = LocalEmbeddingProvider(model_root)
    dimension = await asyncio.to_thread(lambda: embedding.dimension)
    if dimension != counts["dimension"]:
        raise ValueError("embedding/index dimension mismatch")
    store = QdrantVectorStore(snapshot / "qdrant", dimension=dimension, collection=args.collection)
    lexical = SqliteLexicalStore(snapshot / "lexical.db")
    dense = RecordingRetriever(DenseRetriever(embedding=embedding, store=store))
    bm25 = RecordingRetriever(BM25Retriever(lexical))
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, candidate_multiplier=2, rrf_k=60, document_aggregation="sum"
    )
    groups: dict[str, list[dict]] = {"A": [], "D1": []}
    cache_path = args.cache.resolve() if args.cache else output / "rewrite_cache.json"
    meta = cache_metadata(args.model or config.llm.default_model)
    cache: dict = {"metadata": meta, "entries": {}}
    if args.rewrite_enabled and cache_path.exists() and not args.refresh_cache:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache["metadata"] != meta:
            raise ValueError("cache model/prompt/parameters mismatch; use --refresh-cache")
    rewriter = QueryRewriter(model=meta["model"]) if args.rewrite_enabled else None
    disabled_client = InjectedClient("exception")
    disabled_rewriter = QueryRewriter(model="injected", client=disabled_client)
    d2_checks = {}
    stable = True
    live_calls = 0
    live_latencies = []
    try:
        for q, labels in zip(questions, targets, strict=True):
            for group, injected in (("A", None), ("D1", disabled_rewriter)):
                effective, result, latency = await select_query(q["question"], rewriter=injected)
                groups[group].append(
                    await retrieve_row(
                        q, group, effective, result, latency, hybrid, dense, bm25, labels
                    )
                )
        print("A/D1 completed", flush=True)
        for kind in ("timeout", "exception", "empty"):
            client = InjectedClient(kind)
            injected = QueryRewriter(model="injected", client=client, timeout_budget_s=0.01)
            group = f"D2_{kind}"
            groups[group] = []
            for q, labels, baseline in zip(questions, targets, groups["A"], strict=True):
                effective, result, latency = await select_query(
                    q["question"], rewrite_enabled=True, rewriter=injected
                )
                row = await retrieve_row(
                    q, group, effective, result, latency, hybrid, dense, bm25, labels
                )
                groups[group].append(row)
                expected = (
                    ("empty", "empty_output")
                    if kind == "empty"
                    else ("fallback", "timeout" if kind == "timeout" else "error:RuntimeError")
                )
                d2_checks[f"{kind}_Q{q['qid']}"] = (
                    (result.status, result.fallback_reason) == expected
                    and result.rewritten_query is None
                    and effective == q["question"]
                    and row["retrieved_documents"] == baseline["retrieved_documents"]
                )
            d2_checks[f"{kind}_calls"] = client.calls == len(questions)
        print("D2 completed", flush=True)
        if args.rewrite_enabled:
            groups["B"] = []
            for q, labels in zip(questions, targets, strict=True):
                key = str(q["qid"])
                entry = cache["entries"].get(key)
                if entry is not None:
                    # 缓存读取也经过同一 effective_query 编排路径。
                    effective, result, _ = await select_query(
                        q["question"], rewrite_enabled=True, rewriter=CachedRewriter(entry)
                    )
                    latency = entry["rewrite_latency_ms"]
                    cache_hit = True
                else:
                    effective, result, latency = await select_query(
                        q["question"], rewrite_enabled=True, rewriter=rewriter
                    )
                    live_calls += 1
                    live_latencies.append(latency)
                    entry = {
                        "original_query": q["question"],
                        "result": asdict(result),
                        "rewrite_latency_ms": latency,
                        "generated_at": now(),
                    }
                    cache["entries"][key] = entry
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(cache_path, cache)
                    cache_hit = False
                row = await retrieve_row(
                    q, "B", effective, result, latency, hybrid, dense, bm25, labels
                )
                row.update({"cache_hit": cache_hit, "generated_at": entry["generated_at"]})
                groups["B"].append(row)
                repeated = await retrieve_row(
                    q, "B_repeat", effective, result, latency, hybrid, dense, bm25, labels
                )
                row["fixed_query_repeat_equal"] = (
                    row["retrieved_documents"] == repeated["retrieved_documents"]
                )
                stable = stable and row["fixed_query_repeat_equal"]
                print(f"B Q{q['qid']}: {result.status}", flush=True)
        if args.rewrite_enabled and args.c_enabled:
            groups["C"] = []
            for q, labels in zip(questions, targets, strict=True):
                key = str(q["qid"])
                entry = cache["entries"].get(key)
                if entry is None:
                    raise ValueError(f"C group requires a rewrite cache entry for Q{q['qid']}")
                effective, result, _ = await select_query(
                    q["question"], rewrite_enabled=True, rewriter=CachedRewriter(entry)
                )
                if result.status == "success" and result.rewritten_query:
                    row = await retrieve_row_dual(
                        q, "C", q["question"], result.rewritten_query, result,
                        entry["rewrite_latency_ms"], hybrid, dense, bm25, labels,
                    )
                else:
                    # 改写失败/空输出：C 组退化为原问题单路（与 A 组一致）。
                    row = await retrieve_row(
                        q, "C", q["question"], result, entry["rewrite_latency_ms"],
                        hybrid, dense, bm25, labels,
                    )
                    row["retrieval_mode"] = "hybrid-sum-orig-only(fallback)"
                row.update({"cache_hit": True, "generated_at": entry["generated_at"]})
                groups["C"].append(row)
                print(f"C Q{q['qid']}: {result.status}", flush=True)
    finally:
        store.close()
        await lexical.close()
        if rewriter is not None:
            await rewriter.aclose()
    if index_hash != fingerprint(snapshot):
        raise RuntimeError("evaluation index snapshot changed during retrieval")
    with (output / "per_question.jsonl").open("w", encoding="utf-8") as stream:
        for rows in groups.values():
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "run_at": now(),
        "scope": "phase1_evaluation_orchestration_only",
        "configuration": {
            "rewrite_enabled": args.rewrite_enabled,
            "c_enabled": args.c_enabled,
            "top_k": 8,
            "candidate_k_per_source": 16,
            "rrf_k": 60,
            "aggregation": "sum",
            "metric_k": 5,
            "user_id": "local",
        },
        "environment": {
            **counts,
            "embedding_snapshot": str(model_root),
            "index_sha256": index_hash,
            "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        },
        "rewrite_metadata": meta,
        "metrics": {key: summarize(rows) for key, rows in groups.items()},
        "acceptance": acceptance(groups, questions, stable, disabled_client.calls, d2_checks),
        "c_acceptance": (
            c_acceptance(groups["A"], groups["C"], questions)
            if "C" in groups
            else None
        ),
        "D1_llm_calls": disabled_client.calls,
        "D2_checks": d2_checks,
        "live_rewrite_calls": live_calls,
    }
    if "B" in groups:
        rows = groups["B"]
        result["rewrite_statistics"] = {
            "success_rate": sum(row["rewrite_status"] == "success" for row in rows) / len(rows),
            "empty_rate": sum(row["rewrite_status"] == "empty" for row in rows) / len(rows),
            "fallback_rate": sum(row["rewrite_status"] == "fallback" for row in rows) / len(rows),
            "mean_generation_latency_ms": sum(row["rewrite_latency_ms"] for row in rows)
            / len(rows),
            "live_mean_latency_ms": sum(live_latencies) / len(live_latencies)
            if live_latencies
            else None,
            "llm_tokens": {
                key: sum(row["llm_tokens"][key] or 0 for row in rows) for key in ("input", "output")
            },
            "tokens_note": "completed calls only; timed-out or retried usage may be unknown",
        }
    write_json(output / "summary.json", result)
    write_report(output, result, groups, questions)
    print(json.dumps(result["acceptance"], ensure_ascii=False), flush=True)
    return result


def write_report(output: Path, result: dict, groups: dict, questions: list[dict]) -> None:
    lines = [
        "# Query Rewrite 第一阶段评测报告",
        "",
        f"运行时间（UTC）：{result['run_at']}",
        "",
        "参数：每路 16 个 chunk，RRF=60，sum 聚合后最多 8 个文档；主验收在文档 top-5。",
        "历史 0.950/0.863/0.885 为 single @8；本报告 @5 使用同环境 A 组未舍入值。",
        "D1/D2 只验证本评测入口编排，不代表生产 Service 已接入或验证。",
        "",
        "## 指标（未舍入）",
        "",
        "|组|口径|Hit|macro Recall|micro Recall|MRR|NDCG|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group, summary in result["metrics"].items():
        for label, values in summary.items():
            lines.append(
                f"|{group}|{label}|{values['hit']}|{values['recall']}|"
                f"{values['micro_recall']}|{values['mrr']}|{values['ndcg']}|"
            )
    lines += [
        "",
        "## 验收",
        "",
        f"A/B 状态：{result['acceptance']['ab_status']}；"
        f"全部通过：{result['acceptance']['passed']}",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in result["acceptance"]["checks"].items()]
    lines += ["", "## Q18 候选池及结果", ""]
    for group in ("A", "B"):
        if group not in groups:
            continue
        row = next(row for row in groups[group] if row["query_id"] == 18)
        target = "agent_framework_comparison.md"
        pool_presence = {
            key: any(x["source"] == target for x in pool)
            for key, pool in row["candidate_pools"].items()
        }
        lines += [
            f"- {group}: status={row['rewrite_status']}; effective_query={row['effective_query']}",
            f"  - 目标文档进入候选池：{pool_presence}",
            f"  - top-5：{[x['source'] for x in row['retrieved_documents'][:5]]}",
        ]
    if "B" in groups:
        lines += [
            "",
            "## 逐题 A/B 对比",
            "",
            "|Q|原问题|改写|状态|A Hit/MRR|B Hit/MRR|A multi Recall|B multi Recall|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a, b in zip(groups["A"], groups["B"], strict=True):
            rewrite = (b["rewritten_query"] or "").replace("|", "\\|")
            lines.append(
                f"|{a['query_id']}|{a['original_query']}|{rewrite}|{b['rewrite_status']}|"
                f"{a['hit_s']}/{a['mrr_s']}|{b['hit_s']}/{b['mrr_s']}|"
                f"{a['recall_m']}|{b['recall_m']}|"
            )
        regressions = [
            b["query_id"]
            for a, b in zip(groups["A"], groups["B"], strict=True)
            if any(b[key] < a[key] for key in ("hit_s", "mrr_s", "ndcg_s", "hit_m", "recall_m"))
        ]
        lines += [
            "",
            f"任一核心逐题指标下降题目：{regressions}",
            "",
            "## 改写性能",
            "",
            "```json",
            json.dumps(result["rewrite_statistics"], ensure_ascii=False, indent=2),
            "```",
        ]
    if "C" in groups:
        lines += [
            "",
            "## C 组（原查询 + 改写查询双路融合）",
            "",
            f"C 组验收：{json.dumps(result['c_acceptance'], ensure_ascii=False)}",
            "",
            "|Q|状态|A multi Recall|B multi Recall|C multi Recall|C top-5|",
            "|---|---|---|---|---|---|",
        ]
        for a, b, c in zip(groups["A"], groups["B"], groups["C"], strict=True):
            lines.append(
                f"|{a['query_id']}|{c['rewrite_status']}|{a['recall_m']}|{b['recall_m']}|{c['recall_m']}|"
                f"{[x['source'] for x in c['retrieved_documents'][:5]]}|"
            )
    lines += [
        "",
        "完整候选、document_id、状态、耗时、token、逐题指标见 per_question.jsonl；",
        "模型、prompt/参数、快照指纹、原始验收结果见 summary.json。",
        "",
        "限于当前 20 题与固定索引；不更新简历数据。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-enabled", action="store_true", default=False)
    parser.add_argument("--c-enabled", action="store_true", default=False)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge_docs/rewrite_results")
    parser.add_argument("--golden", type=Path, default=ROOT / "knowledge_docs/golden_set.json")
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5",
    )
    parser.add_argument("--qdrant", type=Path, default=Path.home() / ".kama/knowledge/qdrant")
    parser.add_argument("--lexical", type=Path, default=Path.home() / ".kama/knowledge/lexical.db")
    parser.add_argument("--collection", default="kama_knowledge")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
