"""RAG V1 全量入库脚本：走 KnowledgeService 真实链路（AgentRunner 同款代码路径）。

用法：
    python index_knowledge.py            # 全量入库
    python index_knowledge.py 3          # 只处理前 3 篇（试跑验证）
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.catalog import SqliteDocumentCatalog
from kama_claude.core.knowledge.chunker.basic import BasicChunker
from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeDocument
from kama_claude.core.knowledge.parser.file import FileParser
from kama_claude.core.knowledge.retriever.lexical_store import SqliteLexicalStore
from kama_claude.core.knowledge.service import KnowledgeService
from kama_claude.core.knowledge.store.qdrant import QdrantVectorStore

ROOT = Path(r"D:\tinyclaude\knowledge_docs")
# 非知识文档：分析报告、基线数据等不入知识库
EXCLUDE = {
    "chunk_baseline.md", "chunk_baseline.csv",
    "search_baseline.md", "eval_baseline.md",
    "RAG_V1_实现总结与改进方向.md",
    "RAG_V2_HYBRID_PLAN.md", "RAG_V2_HYBRID_EXECUTION_PLAN.md",
    "RAG_V2_HYBRID_实现总结与改进方向.md",
    # 中间产物：QueryRewrite 相关方案/报告/检查清单不入知识库
    "QueryRewrite_前置核查报告.md",
    "QueryRewrite_实现总结与改进方向.md",
    "QueryRewrite_阶段进展与问题记录_2026-10-10.md",
    "RAG_V3_QueryRewrite_方案.md",
    "给Codex的实施前置检查清单.md",
    "report.md",
}
MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()
QDRANT_DIR = Path.home() / ".kama" / "knowledge" / "qdrant"
CATALOG_DB = Path.home() / ".kama" / "knowledge" / "catalog.db"
LEXICAL_DB = Path.home() / ".kama" / "knowledge" / "lexical.db"
COLLECTION = "kama_knowledge"
USER_ID = "local"
SCOPE = "global"


async def main(limit: int | None) -> None:
    embedding = LocalEmbeddingProvider(SNAPSHOT)
    store = QdrantVectorStore(QDRANT_DIR, dimension=embedding.dimension, collection=COLLECTION)
    catalog = SqliteDocumentCatalog(CATALOG_DB)
    lexical_store = SqliteLexicalStore(LEXICAL_DB)
    service = KnowledgeService(
        parser=FileParser(),
        chunker=BasicChunker(tokenizer=embedding.tokenizer),
        embedding=embedding,
        store=store,
        lexical_store=lexical_store,
        catalog=catalog,
    )

    files = [p for p in sorted(ROOT.rglob("*.md")) if p.name not in EXCLUDE]

    # 清理已入库但被排除的文档（如旧基线报告），保证库内容干净
    for name in EXCLUDE:
        rel = name
        eid = "doc_" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
        existing = await catalog.get(eid)
        if existing is not None:
            await service.delete(eid)
            print(f"  [clean] 已删除被排除文档: {name} ({eid})")
    if limit is not None:
        files = files[:limit]

    total_ok = total_skip = total_fail = 0
    total_chunks = 0
    t0 = time.time()
    # 并发度：embedding 在线程池并行（torch 推理释放 GIL），Qdrant 写入由 store 内部锁串行
    CONCURRENCY = 3
    semaphore = asyncio.Semaphore(CONCURRENCY)
    counters = {"ok": 0, "skip": 0, "fail": 0, "chunks": 0}

    async def index_one(path: Path) -> None:
        rel = str(path.relative_to(ROOT))
        doc_id = "doc_" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
        doc = KnowledgeDocument(
            # title 传空：让 FileParser 提取 frontmatter 的中文标题；无 frontmatter 时回退文件名
            id=doc_id, source=str(path), title="",
            scope=SCOPE, user_id=USER_ID,
        )
        async with semaphore:
            try:
                result = await service.index(doc)
                counters["chunks"] += result.chunk_count
                if result.skipped:
                    counters["skip"] += 1
                    print(f"  [skip] {rel}  (unchanged, {result.chunk_count} chunks)")
                else:
                    counters["ok"] += 1
                    print(f"  [ok]   {rel}  ({result.chunk_count} chunks)")
            except Exception as exc:
                counters["fail"] += 1
                print(f"  [FAIL] {rel}  ({exc})")

    await asyncio.gather(*(index_one(p) for p in files))

    store.close()
    await lexical_store.close()
    elapsed = time.time() - t0
    print(f"\n=== 入库完成 ===")
    print(f"处理文档 : {len(files)}")
    print(f"新增/更新: {counters['ok']}   跳过: {counters['skip']}   失败: {counters['fail']}")
    print(f"chunk 总数: {counters['chunks']}")
    print(f"耗时     : {elapsed:.1f}s")
    print(f"Qdrant  : {QDRANT_DIR}")
    print(f"Catalog : {CATALOG_DB}")
    print(f"Lexical : {LEXICAL_DB}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(int(arg) if arg is not None else None))
