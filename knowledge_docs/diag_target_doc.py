# -*- coding: utf-8 -*-
"""诊断 agent_framework_comparison.md 在索引中的存在性与检索可达性。"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider  # noqa: E402
from kama_claude.core.knowledge.retriever.bm25 import BM25Retriever  # noqa: E402
from kama_claude.core.knowledge.retriever.dense import DenseRetriever  # noqa: E402
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore  # noqa: E402
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore  # noqa: E402


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
    target = "agent_framework_comparison.md"

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))
    try:
        info = client.get_collection(collection)
        offset = None
        found = []
        while True:
            points, offset = client.scroll(
                collection, offset=offset, limit=256, with_payload=True, with_vectors=False
            )
            for point in points:
                payload = point.payload or {}
                name = PureWindowsPath(payload.get("source", "")).name
                if name == target:
                    found.append({
                        "chunk_id": payload.get("chunk_id"),
                        "document_id": payload.get("document_id"),
                        "title": payload.get("title"),
                        "content_preview": (payload.get("content") or "")[:200],
                    })
            if offset is None:
                break
    finally:
        client.close()

    print("== Qdrant 中 agent_framework_comparison.md ==")
    if not found:
        print("NOT FOUND in qdrant!")
    for f in found:
        print(json.dumps(f, ensure_ascii=False, indent=2))

    with sqlite3.connect(lexical_path.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT chunk_id, document_id FROM knowledge_chunks_fts"
        ).fetchall()
        lex_found = [r for r in rows if r[1] == (found[0]["document_id"] if found else "?")]
        print("\n== SQLite FTS5 中该 document_id 的 chunk 数 ==")
        print(len(lex_found))

        # 原始文档内容：看标题/内容本身为何检索不到
        try:
            doc_rows = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            print("\n== tables ==", [r[0] for r in doc_rows])
        except Exception as exc:
            print("table list error:", exc)


if __name__ == "__main__":
    main()
