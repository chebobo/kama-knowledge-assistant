from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kama_claude.core.mcp.client import McpClient, McpToolDef
from kama_claude.core.mcp.exposure import McpToolExposure
from kama_claude.core.mcp.tool import McpTool
from kama_claude.core.tools.builtin.tool_search import ToolSearchTool


# 构造用于工具发现测试的目录
def _exposure(**kwargs: object) -> McpToolExposure:
    tools = [McpTool(AsyncMock(spec=McpClient), "files", McpToolDef(n, "Read a file", {
        "type": "object", "properties": {"path": {"type": "string"}},
    })) for n in ("a", "b", "c")]
    return McpToolExposure(tools, mode="lazy", **kwargs)


# 功能：搜索激活工具，返回摘要不带 schema，limit 被内部钳制
# 设计：以多个越界 limit 调用真实 ToolSearchTool，按最终可见状态断言
@pytest.mark.parametrize("limit, count", [(-10, 1), (0, 1), (100, 2)])
async def test_search_activates_and_clamps(limit: int, count: int) -> None:
    exposure = _exposure(search_max_results=2)
    tool = ToolSearchTool(exposure)
    assert tool.validate_params({"query": "files", "limit": limit}) is None
    result = await tool.invoke({"query": "files", "limit": limit})
    assert not result.is_error
    assert len(exposure.visible_names() - {"tool_search"}) == count
    assert "input_schema" not in result.content and '"properties"' not in result.content
    assert "[activated]" in result.content
    assert "next turn" in result.content


# 功能：无匹配结果正常返回可读提示
# 设计：输入不存在的名称，断言结果不是错误且未激活工具
async def test_no_matches() -> None:
    exposure = _exposure()
    result = await ToolSearchTool(exposure).invoke({"query": "nonexistent"})
    assert not result.is_error and "No MCP tools found" in result.content
    assert exposure.visible_names() == {"tool_search"}


# 功能：激活预算不足时结果准确说明原因，工具不会被错误标记为激活
# 设计：将唯一位置留给常驻工具，搜索其他工具必然拒绝
async def test_search_explains_rejection() -> None:
    exposure = _exposure(max_active_tools=1, always_expose=["files__a"])
    result = await ToolSearchTool(exposure).invoke({"query": "files__b"})
    assert "not activated: max_active_tools budget" in result.content
    assert not exposure.is_visible("files__b")


# 功能：第四次同集合搜索返回不可重试的硬错误和替代路径
# 设计：通过真实工具入口触发阈值，检查错误类型、无结果摘要和空重试集合
async def test_search_blocked_error_and_guidance() -> None:
    tool = ToolSearchTool(_exposure())
    for _ in range(3):
        result = await tool.invoke({"query": "files__a"})
        assert not result.is_error
    result = await tool.invoke({"query": "files__a"})
    assert result.is_error and result.error_type == "search_blocked"
    assert "bash" in result.content and "read_file" in result.content
    assert "Found" not in result.content
    assert tool.retryable_error_types == frozenset()


# 功能：首次新 query 返回相同集合时只提示结果重复
# 设计：共享描述让两个不同 query 命中同一集合，检查未误报 query 历史
async def test_same_result_hint_without_query_hint() -> None:
    tool = ToolSearchTool(_exposure())
    await tool.invoke({"query": "Read"})
    result = await tool.invoke({"query": "file"})
    assert "same result" in result.content
    assert "previous 8 valid searches" not in result.content


# 功能：A-B-A 非连续结果只提示 query 历史
# 设计：使用精确工具名称得到不同集合，第三次返回 A 时不误报结果重复
async def test_query_hint_without_same_result_hint() -> None:
    tool = ToolSearchTool(_exposure())
    await tool.invoke({"query": "files__a"})
    await tool.invoke({"query": "files__b"})
    result = await tool.invoke({"query": "files__a"})
    assert "previous 8 valid searches" in result.content
    assert "same result" not in result.content


# 功能：query 和结果都重复时正常返回并同时提示
# 设计：第二次查询归一化后的同词，检查两个提示独立出现且仍激活
async def test_both_repeat_hints() -> None:
    tool = ToolSearchTool(_exposure())
    await tool.invoke({"query": "FILES__A"})
    result = await tool.invoke({"query": " files__a "})
    assert not result.is_error
    assert "previous 8 valid searches" in result.content and "same result" in result.content
    assert "[activated]" in result.content


# 功能：正常空结果也有独立提示并在第四次转换为硬错误
# 设计：使用不同不存在词验证第 2、3 次只提示结果，第 4 次拦截
async def test_empty_results_hint_and_block() -> None:
    tool = ToolSearchTool(_exposure())
    first = await tool.invoke({"query": "missing-0"})
    assert not first.is_error and "Note:" not in first.content
    for index in (1, 2):
        result = await tool.invoke({"query": f"missing-{index}"})
        assert not result.is_error and "No MCP tools found" in result.content
        assert "same result" in result.content
        assert "previous 8 valid searches" not in result.content
    fourth = await tool.invoke({"query": "missing-3"})
    assert fourth.is_error and fourth.error_type == "search_blocked"


# 功能：相同 query 的正常空结果同时展示 query 和结果提示
# 设计：重复不存在词两次，验证空结果不会绕开公共提示组装分支
async def test_empty_results_both_hints() -> None:
    tool = ToolSearchTool(_exposure())
    await tool.invoke({"query": "missing"})
    result = await tool.invoke({"query": "missing"})
    assert not result.is_error
    assert "previous 8 valid searches" in result.content and "same result" in result.content


# 功能：未拦截搜索可以重新激活，被禁集合会阻止预算淘汰后的重新激活
# 设计：单位置预算使 B 淘汰 A，分别验证禁用前恢复 A 和禁用后保持 B 可见
async def test_reactivation_before_and_after_result_set_is_blocked() -> None:
    exposure = _exposure(max_active_tools=1)
    tool = ToolSearchTool(exposure)
    await tool.invoke({"query": "files__a"})
    await tool.invoke({"query": "files__b"})
    assert not exposure.is_visible("files__a")
    restored = await tool.invoke({"query": "files__a"})
    assert not restored.is_error and exposure.is_visible("files__a")
    for _ in range(3):
        blocked = await tool.invoke({"query": "files__a"})
    assert blocked.error_type == "search_blocked"
    await tool.invoke({"query": "files__b"})
    blocked = await tool.invoke({"query": "files__a"})
    assert blocked.error_type == "search_blocked"
    assert not exposure.is_visible("files__a") and exposure.is_visible("files__b")
