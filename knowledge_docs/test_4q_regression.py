# -*- coding: utf-8 -*-
"""prompt v2.1 验证：Q2/Q5/Q9（回归三题）+ Q18（目标题）改写后是否全部命中。"""
from __future__ import annotations

import asyncio
import json
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
from kama_claude.core.knowledge.rewrite import QueryRewriter  # noqa: E402
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402


def main() -> None:
    golden = json.loads((ROOT / "knowledge_docs/golden_set.json").read_text(encoding="utf-8"))
    qids = [2, 3, 5, 18]
    questions = {q["qid"]: q for q in golden["questions"] if q["qid"] in qids}

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
        rw = QueryRewriter()
        try:
            for qid in qids:
                q = questions[qid]["question"]
                target = questions[qid]["expected_doc"]
                res = await rw.rewrite(q)
                hit = "?"
                rank = "-"
                if res.status == "success":
                    hits = await hybrid.retrieve(
                        res.rewritten_query, top_k=8, filters=KnowledgeFilter(user_id="local")
                    )
                    srcs = [PureWindowsPath(h.chunk.source).name for h in hits]
                    hit = target in srcs[:5]
                    rank = srcs.index(target) + 1 if target in srcs else "-"
                print(json.dumps({
                    "qid": qid,
                    "original": q,
                    "status": res.status,
                    "stop_reason": res.stop_reason,
                    "rewritten": res.rewritten_query,
                    "target_in_top5": hit,
                    "target_rank": rank,
                }, ensure_ascii=False))
        finally:
            await rw.aclose()
            store.close()
            await lexical.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
