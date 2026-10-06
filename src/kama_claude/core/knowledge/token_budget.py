from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    # 统计文本对应的 token 数
    def count(self, text: str) -> int: ...

    # 将文本截断到指定 token 预算以内
    def truncate(self, text: str, max_tokens: int) -> str: ...


class CharacterTokenCounter:
    # 初始化字符近似 token 比例，供 Phase 0 测试替身使用
    def __init__(self, chars_per_token: int = 4) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    # 使用字符数近似统计 token 数
    def count(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + self._chars_per_token - 1) // self._chars_per_token

    # 使用字符边界近似截断文本
    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        max_chars = max_tokens * self._chars_per_token
        return text[:max_chars]


class HuggingFaceTokenCounter:
    # 使用本地 HuggingFace tokenizer 统计和截断 token
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    # 统计文本编码后的 token 数
    def count(self, text: str) -> int:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return len(encoded)

    # 按 token 边界截断文本，避免字符截断破坏 tokenizer 结果
    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        encoded = self._tokenizer.encode(  # type: ignore[attr-defined]
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        decoded = self._tokenizer.decode(encoded, skip_special_tokens=True)  # type: ignore[attr-defined]
        return str(decoded)
