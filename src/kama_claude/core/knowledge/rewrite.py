"""受约束的检索查询改写；检索开关和 effective_query 选择归调用方。"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

from kama_claude.core.config import get_config
from kama_claude.core.llm.provider import AnthropicProvider

PROMPT_VERSION = "query-rewrite-v2.2"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
PROMPT = """你是一个面向本地技术知识库的检索查询改写助手。
你的任务是将用户的原始问题转换为更适合向量检索和关键词检索的查询表达。

改写规则：
- 保留原问题的核心意图、问题类型和关键约束，不得改变用户实际想问的问题。
- 先判断原问题类型，二选一执行：
  1) 原问题包含具体实体或明确概念（如具体框架、技术、机制、方法名称）：保留原问题的全部关键词，仅删除"是什么/如何/为什么/哪些"等提问辅助词；不得添加英文全称、同义表达或其他新词。
  2) 原问题是模糊的类别比较或概念提问（如"主流 Agent 框架有什么异同"）：删除"主流""框架""异同"等检索价值低的泛词，用主题维度词补充（如"架构设计、记忆机制、工具调用、上下文管理、安全机制、适用场景、优劣对比"）。
- 禁止引入原问题未提及的任何具体框架、产品、工具或技术名称。
- 不回答用户的问题，不添加解释、结论、推荐或事实性断言。
- 输出一条简洁、完整的查询语句，不输出标题、引号、推理过程或其他说明。

原始问题：{query}
改写后的查询：
"""  # noqa: E501


@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str | None
    status: Literal["success", "empty", "fallback"]
    fallback_reason: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    thinking_chars: int | None = None
    text_chars: int | None = None


class QueryRewriter:
    def __init__(
        self, model: str | None = None, client: Any = None, timeout_budget_s: float = 15.0
    ) -> None:
        if not math.isfinite(timeout_budget_s) or timeout_budget_s <= 0:
            raise ValueError("timeout_budget_s must be finite and positive")
        self.model = model if model is not None else get_config().llm.default_model
        self.timeout_budget_s = timeout_budget_s
        self._client = client
        self._provider: AnthropicProvider | None = None

    async def rewrite(self, query: str) -> RewriteResult:
        if not query.strip():
            return RewriteResult(None, "empty", "empty_input")
        try:
            # SDK 的全部请求、重试和退避均位于同一个总预算内。
            async with asyncio.timeout(self.timeout_budget_s):
                if self._provider is None:
                    if self._client is None and not os.environ.get("ANTHROPIC_API_KEY"):
                        return RewriteResult(None, "fallback", "missing_api_key")
                    self._provider = AnthropicProvider(self.model, client=self._client)
                response = await self._provider.complete_text(
                    PROMPT.format(query=query), max_tokens=MAX_TOKENS, temperature=TEMPERATURE
                )
                text = response.text.strip().strip('"“”').strip()
                usage = response.usage
                input_tokens: int | None = usage.input_tokens if usage else None
                output_tokens: int | None = usage.output_tokens if usage else None
                stop_reason: str | None = response.stop_reason
                thinking_chars: int = sum(
                    len(str(b.get("thinking", "")))
                    for b in (response.thinking_blocks or [])
                )
                text_chars: int = len(text)
                if not text:
                    return RewriteResult(
                        None, "empty", "empty_output",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        thinking_chars=thinking_chars,
                        text_chars=text_chars,
                    )
                if response.stop_reason != "end_turn":
                    return RewriteResult(
                        None, "fallback", "incomplete_output",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        thinking_chars=thinking_chars,
                        text_chars=text_chars,
                    )
                if "\n" in text or text.startswith(("改写后的查询：", "```")):
                    return RewriteResult(
                        None, "fallback", "invalid_output",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        thinking_chars=thinking_chars,
                        text_chars=text_chars,
                    )
                return RewriteResult(
                    text, "success", None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stop_reason=stop_reason,
                    thinking_chars=thinking_chars,
                    text_chars=text_chars,
                )
        except TimeoutError:
            return RewriteResult(None, "fallback", "timeout")
        except Exception as exc:
            # 不落盘异常正文，避免 SDK 异常暴露 URL/请求头/凭据。
            return RewriteResult(None, "fallback", f"error:{type(exc).__name__}")

    async def aclose(self) -> None:
        if self._provider is not None and self._client is None:
            await self._provider.aclose()
            self._provider = None
