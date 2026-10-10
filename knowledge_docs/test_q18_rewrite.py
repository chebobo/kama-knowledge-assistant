# -*- coding: utf-8 -*-
"""Q18 单测：max_tokens=512 后验证改写是否成功，以及改写 query 能否命中目标文档。
复用生产 HybridRetriever + 真实 LLM（QueryRewriter）。"""
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
from kama_claude.core.knowledge.rewrite import MAX_TOKENS, PROMPT_VERSION, QueryRewriter  # noqa: E402
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402


def main() -> None:
    golden = json.loads((ROOT / "knowledge_docs/golden_set.json").read_text(encoding="utf-8"))
    q18 = next(q for q in golden["questions"] if q["qid"] == 18)
    question = q18["question"]
    expected = q18["expected_doc"]
    print("Q18:", question)
    print("expected_doc:", expected)

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
    dimension = embedding.dimension
    store = QdrantVectorStore(qdrant_path, dimension=dimension, collection=collection)
    lexical = SqliteLexicalStore(lexical_path)
    dense = DenseRetriever(embedding=embedding, store=store)
    bm25 = BM25Retriever(lexical)
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, candidate_multiplier=2, rrf_k=60, document_aggregation="sum"
    )

    async def run() -> None:
        hits = await hybrid.retrieve(question, top_k=8, filters=KnowledgeFilter(user_id="local"))
        print("\n[原 query] top8 sources:")
        for h in hits:
            print("  -", PureWindowsPath(h.chunk.source).name)

        rw = QueryRewriter()
        try:
            res = await rw.rewrite(question)
            print("\n[改写诊断]")
            print(json.dumps({
                "status": res.status,
                "fallback_reason": res.fallback_reason,
                "stop_reason": res.stop_reason,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "thinking_chars": res.thinking_chars,
                "text_chars": res.text_chars,
                "max_tokens": MAX_TOKENS,
                "prompt_version": PROMPT_VERSION,
                "rewritten_query": res.rewritten_query,
            }, ensure_ascii=False, indent=2))

            if res.status == "success":
                hits2 = await hybrid.retrieve(
                    res.rewritten_query, top_k=8, filters=KnowledgeFilter(user_id="local")
                )
                srcs2 = [PureWindowsPath(h.chunk.source).name for h in hits2]
                print("\n[改写 query] top8 sources:")
                for s in srcs2:
                    print("  -", s)
                print("\ntarget in top5:", expected in srcs2[:5])
                print("target in top8:", expected in srcs2)
        finally:
            await rw.aclose()
            store.close()
            await lexical.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
