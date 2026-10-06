from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    # 返回 embedding 向量维度
    @property
    def dimension(self) -> int: ...

    # 将多段文本转换为 embedding 向量
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

