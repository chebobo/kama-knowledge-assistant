"""RAG V1 环境验证脚本：bge 模型加载 + Qdrant Local 全流程"""
import sys
import time
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5"
SNAPSHOT = next(MODEL_DIR.glob("snapshots/*")).resolve()

def test_embedding() -> list[list[float]]:
    print("=== 1. bge-small-zh-v1.5 加载测试 ===")
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(str(SNAPSHOT))
    print(f"  模型加载成功，耗时 {time.time()-t0:.1f}s")

    texts = ["RAG 检索增强生成是什么", "Agent 如何调用工具", "黄金投资建议"]
    emb = model.encode(texts, normalize_embeddings=True)
    print(f"  生成 {len(emb)} 个向量，维度 {emb.shape[1]}")
    print(f"  向量示例前5维: {[round(x, 4) for x in emb[0][:5]]}")

    # 相似度 sanity check
    import numpy as np
    sim_01 = float(emb[0] @ emb[1])
    sim_02 = float(emb[0] @ emb[2])
    print(f"  相似度[RAG vs Agent]={sim_01:.3f}, [RAG vs 黄金]={sim_02:.3f}")
    assert sim_01 > sim_02, "相似度检查失败：相关文本应比无关文本更相似"
    print("  ✓ 相似度 sanity check 通过")
    return emb.tolist()

def test_qdrant(embeddings: list[list[float]]) -> None:
    print("\n=== 2. Qdrant Local 模式测试 ===")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    qdrant_dir = Path.home() / ".kama" / "knowledge" / "qdrant_test"
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))
    print(f"  Qdrant Local 初始化成功: {qdrant_dir}")

    # 2.1 建 collection（512 维，bge-small-zh 是 512 维）
    COLL = "test_kb"
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        collection_name=COLL,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print("  ✓ collection 创建成功 (512维, cosine)")

    # 2.2 upsert 带 metadata（scope / project_id）
    chunks = [
        PointStruct(id=1, vector=embeddings[0], payload={"content": "RAG 检索增强生成", "scope": "global", "project_id": None, "document_id": "doc_1"}),
        PointStruct(id=2, vector=embeddings[1], payload={"content": "Agent 工具调用", "scope": "project", "project_id": "tinyclaude", "document_id": "doc_1"}),
        PointStruct(id=3, vector=embeddings[2], payload={"content": "黄金投资建议", "scope": "project", "project_id": "other_proj", "document_id": "doc_2"}),
    ]
    client.upsert(collection_name=COLL, points=chunks)
    print("  ✓ upsert 3 个 chunk 成功")

    # 2.3 search：查询向量搜全部（qdrant-client >=1.10 用 query_points）
    hits = client.query_points(collection_name=COLL, query=embeddings[0], limit=3)
    print(f"  ✓ 全量搜索返回 {len(hits.points)} 条，top1 score={hits.points[0].score:.3f}")

    # 2.4 search + metadata filter（project_a 只能看 global + 自己的 project）
    filtered = client.query_points(
        collection_name=COLL,
        query=embeddings[0],
        limit=3,
        query_filter=Filter(
            must=[
                FieldCondition(key="scope", match=MatchValue(value="global")),
            ]
        ),
    )
    print(f"  ✓ scope=global 过滤返回 {len(filtered.points)} 条（应只含全局知识）")

    # 2.5 delete by document_id（用 payload filter 删除整篇文档）
    client.delete(
        collection_name=COLL,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value="doc_1"))]
        ),
    )
    remaining = client.count(collection_name=COLL, exact=True)
    print(f"  ✓ 按 document_id=doc_1 删除后剩余 {remaining.count} 条（应剩 1 条）")
    assert remaining.count == 1, "delete by document_id 失败"
    client.close()
    print("\n=== 全部验证通过 ✅ ===")

if __name__ == "__main__":
    emb = test_embedding()
    test_qdrant(emb)
