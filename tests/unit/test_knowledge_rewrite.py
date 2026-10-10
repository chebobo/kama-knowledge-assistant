from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.knowledge.rewrite import MAX_TOKENS, QueryRewriter


def client(text="检索查询", stop_reason="end_turn"):
    sdk = MagicMock()
    sdk.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )
    )
    return sdk


async def test_success_uses_existing_provider_and_generation_parameters():
    sdk = client(" “检索查询” ")
    result = await QueryRewriter(model="test-model", client=sdk).rewrite("原始问题")
    assert (result.status, result.rewritten_query, result.fallback_reason) == (
        "success",
        "检索查询",
        None,
    )
    assert (result.input_tokens, result.output_tokens) == (100, 10)
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["extra_body"] == {"temperature": 0.0}
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert "原始问题：原始问题" in kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    ("text", "stop", "status", "reason"),
    [
        ("  ", "end_turn", "empty", "empty_output"),
        ("部分查询", "max_tokens", "fallback", "incomplete_output"),
        ("查询\n解释", "end_turn", "fallback", "invalid_output"),
    ],
)
async def test_invalid_results_do_not_silently_return_original(text, stop, status, reason):
    result = await QueryRewriter(model="test", client=client(text, stop)).rewrite("原问题")
    assert (result.status, result.fallback_reason) == (status, reason)
    assert result.rewritten_query is None


async def test_total_timeout_covers_client_retry_and_backoff():
    sdk = client()
    attempts = 0

    async def retries(**kwargs):
        nonlocal attempts
        for _ in range(3):
            attempts += 1
            await asyncio.sleep(0)
        await asyncio.sleep(60)

    sdk.messages.create.side_effect = retries
    start = time.perf_counter()
    result = await QueryRewriter(model="test", client=sdk, timeout_budget_s=0.04).rewrite("q")
    assert time.perf_counter() - start < 0.5
    assert attempts == 3
    assert (result.status, result.rewritten_query, result.fallback_reason) == (
        "fallback",
        None,
        "timeout",
    )


async def test_cancellation_propagates():
    sdk = client()
    sdk.messages.create.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await QueryRewriter(model="test", client=sdk).rewrite("q")


async def test_exception_reason_does_not_leak_credentials():
    sdk = client()
    sdk.messages.create.side_effect = RuntimeError("secret-api-key")
    result = await QueryRewriter(model="test", client=sdk).rewrite("q")
    assert result.fallback_reason == "error:RuntimeError"
    assert result.rewritten_query is None


async def test_missing_key_is_structured_failure(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await QueryRewriter(model="test").rewrite("q")
    assert result.fallback_reason == "missing_api_key"


async def test_empty_input_skips_llm():
    sdk = client()
    result = await QueryRewriter(model="test", client=sdk).rewrite("  ")
    assert result.status == "empty"
    sdk.messages.create.assert_not_called()


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        QueryRewriter(model="test", timeout_budget_s=budget)
