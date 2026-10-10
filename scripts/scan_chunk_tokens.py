"""扫描知识库全部 chunk 的 token 数分布，确认与 bge-small-zh-v1.5 的 512 token 上限的差距。

用途：诊断 transformers 警告 "1070 > 512" 的根因与影响面。
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QDRANT_PATH = Path("~/.kama/knowledge/qdrant").expanduser()
MODEL_PATH = (
    "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/"
    "snapshots/7999e1d3359715c523056ef9478215996d62a620"
)


def main() -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(QDRANT_PATH))
    collection = "kama_knowledge"
    info = client.get_collection(collection)
    total_points = info.points_count
    print(f"collection total points: {total_points}")

    # 分页读取全部 points 的 content
    chunks: list[str] = []
    offset: str | None = None
    while True:
        page = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = page
        for p in points:
            content = (p.payload or {}).get("content", "")
            if content:
                chunks.append(content)
        if next_offset is None:
            break
        offset = next_offset

    print(f"read chunks: {len(chunks)}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(Path(MODEL_PATH).expanduser()), local_files_only=True)
    max_seq = model.get_max_seq_length()
    tokenizer = model.tokenizer
    print(f"model max_seq_length: {max_seq}, tokenizer type: {type(tokenizer).__name__}")

    # 真实 token 数（不含特殊 token）
    counts = []
    for text in chunks:
        counts.append(len(tokenizer.encode(text, add_special_tokens=False)))

    total = len(counts)
    over_512 = sum(1 for c in counts if c > 512)
    over_1024 = sum(1 for c in counts if c > 1024)
    over_1500 = sum(1 for c in counts if c > 1500)
    max_count = max(counts)
    avg = sum(counts) / total if total else 0

    print(f"\n=== token 分布统计（真实 tokenizer）===")
    print(f"chunk 总数: {total}")
    print(f"平均 token: {avg:.1f}")
    print(f"最大 token: {max_count}")
    print(f"> 512  (截断): {over_512}  ({over_512 / total * 100:.1f}%)")
    print(f"> 1024       : {over_1024}  ({over_1024 / total * 100:.1f}%)")
    print(f"> 1500       : {over_1500}  ({over_1500 / total * 100:.1f}%)")

    # 字符数与 token 数的比例关系（验证"中文 1 字 ≈ 1 token"假设）
    ratios = []
    for text, c in zip(chunks, counts):
        if c > 0:
            ratios.append(len(text) / c)
    if ratios:
        print(f"字符/token 比: 平均 {sum(ratios) / len(ratios):.2f}, 最小 {min(ratios):.2f}, 最大 {max(ratios):.2f}")

    # 超长 chunk 的文档分布（前 12 个）
    doc_counter: Counter[str] = Counter()
    for text in chunks:
        if len(tokenizer.encode(text, add_special_tokens=False)) > 512:
            # 从内容首行猜文档名（不可靠但够诊断用）
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")[:40]
            doc_counter[first_line] += 1
    print(f"\n=== >512 的 chunk 内容开头分布（Top 12）===")
    for name, cnt in doc_counter.most_common(12):
        print(f"  {cnt:>4}  {name}")

    # 最大 5 个 chunk 明细
    big = sorted(zip(chunks, counts), key=lambda x: -x[1])[:5]
    print(f"\n=== 最大 5 个 chunk ===")
    for text, c in big:
        print(f"  tokens={c} chars={len(text)} head={text[:60]!r}")

    # 直方图 bucket（供可视化）
    buckets = [0] * 16
    for c in counts:
        idx = min(c // 100, 15)
        buckets[idx] += 1
    print("\n=== token 分布 bucket（每 100 token 一档，最后一档 1500+）===")
    for i, n in enumerate(buckets):
        lo = i * 100
        hi = lo + 100 if i < 15 else None
        label = f"{lo}-{hi}" if hi else f"{lo}+"
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
