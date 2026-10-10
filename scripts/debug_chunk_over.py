"""诊断 token 切分后仍超 512 的 chunk 来源（定位 _split_tokens 的漏洞）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.chunker.basic import BasicChunker
from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.models import KnowledgeDocument
from kama_claude.core.knowledge.parser.file import FileParser

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()

doc_path = Path(r"D:\tinyclaude\knowledge_docs\eval_aggregation.md")
parser = FileParser()
parsed = parser.parse(doc_path)
print(f"parsed chars: {len(parsed.content)}, mime: {parsed.mime_type}")

embedding = LocalEmbeddingProvider(SNAPSHOT)
tok = embedding.tokenizer
chunker = BasicChunker(tokenizer=tok, max_tokens=480, overlap_tokens=48)

doc = KnowledgeDocument(id="doc_test", source=str(doc_path), title="", scope="global", user_id="local")
chunks = chunker.chunk(doc, parsed)
print(f"chunks: {len(chunks)}")

over = []
for c in chunks:
    n = len(tok.encode(c.content, add_special_tokens=False))
    if n > 512:
        over.append((n, c.content[:80]))
print(f">512: {len(over)}")
for n, head in over[:10]:
    print(f"  tokens={n} head={head!r}")

# 检查整篇的 token 数与 section 切分
tokens_all = tok.encode(parsed.content, add_special_tokens=False)
print(f"whole doc tokens: {len(tokens_all)}")
