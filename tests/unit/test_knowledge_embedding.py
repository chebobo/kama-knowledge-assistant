from __future__ import annotations

import asyncio

from kama_claude.core.knowledge.embedding.local import LocalEmbeddingProvider
from kama_claude.core.knowledge.token_budget import HuggingFaceTokenCounter

MODEL_PATH = (
    r"C:\Users\13683\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5"
    r"\snapshots\7999e1d3359715c523056ef9478215996d62a620"
)


# 功能：验证本地 BGE provider 能加载模型并返回动态维度向量
# 设计：使用已缓存的本地模型执行真实推理，确认 dimension 不由业务代码硬编码
def test_local_embedding_dimension_and_vector() -> None:
    provider = LocalEmbeddingProvider(MODEL_PATH)

    assert provider.dimension == 512
    vectors = asyncio.run(provider.embed(["个人助手知识检索"]))

    assert len(vectors) == 1
    assert len(vectors[0]) == provider.dimension


# 功能：验证 BGE tokenizer-backed counter 可以统计并截断文本
# 设计：复用 provider 的 tokenizer，断言截断后 token 数不超过预算
def test_bge_token_counter() -> None:
    provider = LocalEmbeddingProvider(MODEL_PATH)
    counter = HuggingFaceTokenCounter(provider.tokenizer)
    text = "这是用于测试知识检索上下文预算的文本。" * 20

    truncated = counter.truncate(text, 16)

    assert counter.count(truncated) <= 16
    assert counter.count(text) > counter.count(truncated)
