from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class LocalEmbeddingUnavailableError(RuntimeError):
    # 表示本地 embedding 依赖或模型不可用

    pass


class LocalEmbeddingProvider:
    # 初始化本地 SentenceTransformer provider，模型加载延迟到首次调用
    def __init__(self, model_path: str | Path, *, normalize: bool = True) -> None:
        self._model_path = str(Path(model_path).expanduser())
        self._normalize = normalize
        self._model: Any | None = None
        self._dimension: int | None = None
        self._tokenizer: Any | None = None
        self._load_lock = asyncio.Lock()

    # 返回本地模型实际输出的向量维度
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_sync()
        assert self._dimension is not None
        return self._dimension

    # 返回模型使用的 tokenizer，供 TokenCounter 复用
    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._load_sync()
        return self._tokenizer

    # 批量生成 embedding，避免阻塞 asyncio event loop
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        return await asyncio.to_thread(self._embed_sync, texts)

    # 在线程中加载模型和 tokenizer
    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)

    # 同步加载本地 SentenceTransformer 模型
    def _load_sync(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalEmbeddingUnavailableError(
                "sentence-transformers is required for LocalEmbeddingProvider"
            ) from exc
        try:
            model = SentenceTransformer(self._model_path, local_files_only=True)
        except Exception as exc:
            raise LocalEmbeddingUnavailableError(
                f"failed to load local embedding model: {self._model_path}"
            ) from exc
        self._model = model
        get_dimension = getattr(model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = model.get_sentence_embedding_dimension
        dimension: Any = get_dimension()
        self._dimension = int(dimension)
        self._tokenizer = model.tokenizer

    # 同步批量生成并转换为普通 Python float 列表
    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise LocalEmbeddingUnavailableError("embedding model is not loaded")
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]
