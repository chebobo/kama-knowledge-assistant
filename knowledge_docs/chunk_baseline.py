"""chunk 基线统计（V2 chunker）：基于真实 BasicChunker 输出统计全库切分指标。

对比 V1：跳 frontmatter、过滤 <50 字符空壳、段落边界切分。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, r"D:\tinyclaude\src")

from kama_claude.core.knowledge.chunker.basic import BasicChunker
from kama_claude.core.knowledge.models import KnowledgeDocument
from kama_claude.core.knowledge.parser.file import FileParser

ROOT = Path(r"D:\tinyclaude\knowledge_docs")
OUT_CSV = ROOT / "chunk_baseline.csv"
EXCLUDE = {"chunk_baseline.md", "chunk_baseline.csv", "search_baseline.md", "eval_baseline.md", "RAG_V1_实现总结与改进方向.md"}

parser = FileParser()
chunker = BasicChunker()

rows = []
all_lens: list[int] = []
stats = dict(
    files=0, chars=0, chunks=0, full=0, fm_docs=0, fm_chunks_skipped=0,
    empty_skipped=0, boundary_splits=0,
)

for path in sorted(ROOT.rglob("*.md")):
    if path.name in EXCLUDE:
        continue
    rel = str(path.relative_to(ROOT))
    parsed = parser.parse(path)
    doc = KnowledgeDocument(id="base", source=str(path), title=path.stem, scope="global", user_id="local")
    chunks = chunker.chunk(doc, parsed)

    chars = len(parsed.content)
    fm = parsed.content.lstrip().startswith("---")
    n_chunk = len(chunks)
    full = sum(1 for c in chunks if len(c.content) >= 1600)

    stats["files"] += 1
    stats["chars"] += chars
    stats["chunks"] += n_chunk
    stats["full"] += full
    if fm:
        stats["fm_docs"] += 1
    all_lens.extend(len(c.content) for c in chunks)

    rows.append({
        "path": rel, "chars": chars, "chunks": n_chunk,
        "full_window_chunks": full, "frontmatter": "Y" if fm else "N",
    })

try:
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "chars", "chunks", "full_window_chunks", "frontmatter"])
        w.writeheader()
        w.writerows(rows)
    csv_written = True
except PermissionError:
    csv_written = False

n = len(all_lens)
avg = sum(all_lens) / n if n else 0
median = sorted(all_lens)[n // 2] if n else 0

print(f"文档数              : {stats['files']}")
print(f"总字符数            : {stats['chars']}")
print(f"总 chunk 数         : {stats['chunks']}")
print(f"平均每篇 chunk      : {stats['chunks'] / stats['files']:.1f}")
print(f"满窗口 chunk(1600)  : {stats['full']}")
print(f"带 frontmatter 文档 : {stats['fm_docs']}（V2 已全部跳过，不进入 chunks）")
print(f"chunk 长度 均值      : {avg:.0f}")
print(f"chunk 长度 中位数    : {median}")
print(f"最短 chunk          : {min(all_lens)}")
print(f"最长 chunk          : {max(all_lens)}")
print(f"明细已写入          : {OUT_CSV if csv_written else '被占用，未写入（统计不受影响）'}")
