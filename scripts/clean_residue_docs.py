"""删除残留的中间产物文档（子目录 report.md 等，EXCLUDE name 匹配删不掉的）。

EXCLUDE 清理 bug 说明：入库 doc_id 用"相对路径"计算，清理却用"文件名"，
子目录文档（rewrite_results_v2_2\\report.md 等）id 对不上删不掉，这里按真实路径补删。
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.catalog import SqliteDocumentCatalog
from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore
from kama_claude.core.knowledge.service import KnowledgeService
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
CATALOG_DB = Path.home() / ".kama" / "knowledge" / "catalog.db"
LEXICAL_DB = Path.home() / ".kama" / "knowledge" / "lexical.db"
ROOT = Path(r"D:\tinyclaude\knowledge_docs")

# 真实相对路径（与入库时一致，Windows 反斜杠）
RESIDUE = [
    "rewrite_checkpoint_20261010\\QueryRewrite_前置核查报告.md",
    "rewrite_checkpoint_20261010\\report.md",
    "rewrite_results\\report.md",
    "rewrite_results_v2\\report.md",
    "rewrite_results_v2_1\\report.md",
    "rewrite_results_v2_2\\report.md",
]


async def main() -> None:
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension)
    catalog = SqliteDocumentCatalog(CATALOG_DB)
    lexical_store = SqliteLexicalStore(LEXICAL_DB)
    service = KnowledgeService(
        embedding=embedding,
        store=store,
        lexical_store=lexical_store,
        catalog=catalog,
    )
    for rel in RESIDUE:
        doc_id = "doc_" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
        existing = await catalog.get(doc_id)
        if existing is not None:
            await service.delete(doc_id)
            print(f"  [deleted] {rel} ({doc_id})")
        else:
            print(f"  [not found] {rel} ({doc_id})")
    store.close()
    await lexical_store.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
