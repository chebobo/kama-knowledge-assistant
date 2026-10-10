import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kama_claude.core.knowledge.rewrite import RewriteResult

_PATH = Path(__file__).parents[2] / "knowledge_docs/eval_rag_rewrite.py"
_SPEC = importlib.util.spec_from_file_location("eval_rag_rewrite", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


async def test_disabled_orchestration_never_calls_llm():
    rewriter = AsyncMock()
    effective, result, latency = await _MODULE.select_query("original", rewriter=rewriter)
    assert effective == "original" and latency == 0
    assert result.fallback_reason == "disabled"
    rewriter.rewrite.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        RewriteResult(None, "empty", "empty_output"),
        RewriteResult(None, "fallback", "timeout"),
        RewriteResult(None, "fallback", "error:RuntimeError"),
    ],
)
async def test_fallback_orchestration_returns_original(result):
    rewriter = AsyncMock()
    rewriter.rewrite.return_value = result
    effective, actual, _ = await _MODULE.select_query(
        "original", rewrite_enabled=True, rewriter=rewriter
    )
    assert effective == "original" and actual == result


async def test_success_orchestration_only_uses_rewrite():
    rewriter = AsyncMock()
    rewriter.rewrite.return_value = RewriteResult("rewritten", "success", None)
    effective, _, _ = await _MODULE.select_query(
        "original", rewrite_enabled=True, rewriter=rewriter
    )
    assert effective == "rewritten"


def test_metrics_deduplicate_document_ids_and_truncate_at_five():
    result = _MODULE.question_metrics(["a", "a", "b", "c", "d", "e", "target"], {"a", "target"}, 5)
    assert result["hit"] == 1 and result["recall"] == 0.5
    assert result["mrr"] == 1
    assert result["ndcg"] == pytest.approx(1 / (1 + 1 / 1.584962500721156))


def test_label_mapping_rejects_missing_or_ambiguous_document_ids():
    with pytest.raises(ValueError):
        _MODULE.resolve_targets({"expected_doc": "x.md"}, {})
    with pytest.raises(ValueError):
        _MODULE.resolve_targets({"expected_doc": "x.md"}, {"x.md": {"id1", "id2"}})


def test_cache_identity_tracks_prompt_and_model():
    metadata = _MODULE.cache_metadata("test-model")
    assert metadata["model"] == "test-model"
    assert metadata["prompt_version"]
    assert len(metadata["prompt_sha256"]) == 64
