"""Document aggregation 评测纯函数测试。"""

import importlib.util
import sys
from pathlib import Path

import pytest

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeSearchResult

_MODULE_PATH = Path(__file__).parents[2] / "knowledge_docs" / "eval_rag_aggregation.py"
_SPEC = importlib.util.spec_from_file_location("eval_rag_aggregation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def result(document_id: str, chunk_id: str, score: float) -> KnowledgeSearchResult:
    chunk = KnowledgeChunk(
        id=chunk_id,
        document_id=document_id,
        content=chunk_id,
        source=f"{document_id}.md",
        title=document_id,
    )
    return KnowledgeSearchResult(chunk=chunk, score=score, rank=1)


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("sum", [("doc-a", 0.9), ("doc-b", 0.8)]),
        ("max", [("doc-b", 0.8), ("doc-a", 0.7)]),
        ("first-seen", [("doc-b", 0.8), ("doc-a", 0.7)]),
    ],
)
def test_aggregate_documents_uses_requested_function(aggregation, expected):
    """三种聚合函数应在同一候选顺序上产生各自稳定的文档排序。"""
    results = [
        result("doc-a", "a1", 0.7),
        result("doc-b", "b1", 0.8),
        result("doc-a", "a2", 0.2),
        result("doc-b", "b2", 0.0),
    ]
    actual = _MODULE.aggregate_documents(results, aggregation=aggregation)
    assert [item.document_id for item in actual] == [item[0] for item in expected]
    assert [item.score for item in actual] == pytest.approx([item[1] for item in expected])


def test_aggregate_documents_rejects_unknown_function():
    """未知聚合函数应明确失败。"""
    with pytest.raises(ValueError, match="aggregation"):
        _MODULE.aggregate_documents([], aggregation="mean")
