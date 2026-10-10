from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.knowledge.rewrite import RewriteResult
from kama_claude.core.knowledge.service import KnowledgeService


def make_retriever(hits):
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=hits)
    return retriever


def make_rewriter(result):
    rewriter = MagicMock()
    rewriter.rewrite = AsyncMock(return_value=result)
    return rewriter


def make_service(*, rewrite_enabled=False, rewriter=None, retriever=None, budget=0.5):
    retriever = retriever or make_retriever([])
    return KnowledgeService(
        embedding=MagicMock(),
        store=MagicMock(),
        retriever=retriever,
        rewrite_enabled=rewrite_enabled,
        rewriter=rewriter,
        rewrite_timeout_budget_s=budget,
    )


async def test_disabled_uses_original_query_and_never_calls_rewriter():
    retriever = make_retriever(["hit"])
    rewriter = make_rewriter(RewriteResult("改写", "success", None))
    service = make_service(rewrite_enabled=False, rewriter=rewriter, retriever=retriever)
    result = await service.search("原问题", user_id="u")
    assert result == ["hit"]
    retriever.retrieve.assert_awaited_once()
    assert retriever.retrieve.call_args.args[0] == "原问题"
    assert retriever.retrieve.call_args.kwargs["top_k"] == 8
    rewriter.rewrite.assert_not_awaited()


async def test_enabled_success_uses_rewritten_query():
    retriever = make_retriever(["hit"])
    rewriter = make_rewriter(RewriteResult("改写后查询", "success", None))
    service = make_service(rewrite_enabled=True, rewriter=rewriter, retriever=retriever)
    result = await service.search("原问题", user_id="u")
    assert result == ["hit"]
    rewriter.rewrite.assert_awaited_once()
    assert retriever.retrieve.call_args.args[0] == "改写后查询"


async def test_enabled_fallback_keeps_original_query():
    retriever = make_retriever(["hit"])
    rewriter = make_rewriter(RewriteResult(None, "fallback", "timeout"))
    service = make_service(rewrite_enabled=True, rewriter=rewriter, retriever=retriever)
    result = await service.search("原问题", user_id="u")
    assert result == ["hit"]
    assert retriever.retrieve.call_args.args[0] == "原问题"


async def test_enabled_empty_keeps_original_query():
    retriever = make_retriever(["hit"])
    rewriter = make_rewriter(RewriteResult(None, "empty", "empty_output"))
    service = make_service(rewrite_enabled=True, rewriter=rewriter, retriever=retriever)
    result = await service.search("原问题", user_id="u")
    assert result == ["hit"]
    assert retriever.retrieve.call_args.args[0] == "原问题"


async def test_disabled_does_not_lazily_create_rewriter(monkeypatch):
    retriever = make_retriever([])
    created = []

    class Boom:
        def __init__(self, *args, **kwargs):
            created.append(1)
            raise AssertionError("rewriter must not be created when disabled")

    monkeypatch.setattr("kama_claude.core.knowledge.service.QueryRewriter", Boom)
    service = make_service(rewrite_enabled=False, rewriter=None, retriever=retriever)
    await service.search("q")
    assert created == []


async def test_enabled_lazily_creates_with_timeout_budget(monkeypatch):
    retriever = make_retriever(["hit"])
    calls: dict[str, float] = {}

    class FakeRewriter:
        def __init__(self, timeout_budget_s):
            calls["budget"] = timeout_budget_s

        async def rewrite(self, query):
            return RewriteResult("改写", "success", None)

    monkeypatch.setattr("kama_claude.core.knowledge.service.QueryRewriter", FakeRewriter)
    service = make_service(rewrite_enabled=True, rewriter=None, retriever=retriever, budget=0.5)
    result = await service.search("q")
    assert result == ["hit"]
    assert calls["budget"] == 0.5
    assert retriever.retrieve.call_args.args[0] == "改写"


def test_invalid_timeout_budget_rejected():
    with pytest.raises(ValueError):
        KnowledgeService(
            embedding=MagicMock(),
            store=MagicMock(),
            retriever=make_retriever([]),
            rewrite_timeout_budget_s=0,
        )
