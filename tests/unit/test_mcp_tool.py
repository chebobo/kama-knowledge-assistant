from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.mcp.client import (
    McpClient,
    McpServerUnavailableError,
    McpToolDef,
    McpToolError,
)
from kama_claude.core.mcp.tool import McpTool
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry


def _make_tool(
    tool_name: str = "read_file",
    server_name: str = "filesystem",
    schema: dict[str, object] | None = None,
) -> tuple[McpTool, AsyncMock]:
    client = AsyncMock(spec=McpClient)
    tool_def = McpToolDef(
        name=tool_name,
        description=f"Read a file via {server_name}",
        input_schema=schema
        or {"type": "object", "properties": {"path": {"type": "string"}}},
    )
    tool = McpTool(client, server_name, tool_def)
    return tool, client


# 功能：McpTool.invoke 应调用 client.call_tool 并将返回值封装为 ToolResult
# 设计：mock client.call_tool 返回固定字符串，验证 ToolResult.content 一致
@pytest.mark.asyncio
async def test_invoke_calls_mcp_client() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(return_value="file content here")
    result = await tool.invoke({"path": "/tmp/test.txt"})
    assert not result.is_error
    assert result.content == "file content here"
    client.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})


# 功能：工具名应以 {server_name}__ 为前缀防止命名冲突
# 设计：验证 McpTool.name 格式为 "filesystem__read_file"
def test_tool_name_prefixed() -> None:
    tool, _ = _make_tool("read_file", "filesystem")
    assert tool.name == "filesystem__read_file"


# 功能：client 抛 McpServerUnavailableError 时应返回 is_error=True 且不重新抛出
# 设计：mock client.call_tool 抛该异常，断言 ToolResult.is_error=True 且消息含 server 名称
@pytest.mark.asyncio
async def test_unavailable_returns_error() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=McpServerUnavailableError("process died"))
    result = await tool.invoke({"path": "/tmp/x.txt"})
    assert result.is_error
    assert result.error_type == "connection_error"
    assert "filesystem" in result.content


# 功能：client 抛其他异常时应返回 runtime_error 类型的 ToolResult
# 设计：mock client.call_tool 抛 RuntimeError，断言 ToolResult 被正确包装
@pytest.mark.asyncio
async def test_runtime_error_caught() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=RuntimeError("unexpected failure"))
    result = await tool.invoke({"path": "/tmp/y.txt"})
    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "unexpected failure" in result.content


# 功能：input_schema 应直接使用 MCP tool_def 中的 schema，而非 pydantic model
# 设计：验证 params_model 为 None，input_schema 与 tool_def.input_schema 一致
def test_input_schema_from_tool_def() -> None:
    tool, _ = _make_tool()
    assert McpTool.params_model is None
    assert "path" in tool.input_schema.get("properties", {})


# --- 参数校验（jsonschema Draft 2020-12）--------------------------------------


def _schema_with(required: list[str] = [], *, additional: bool = True) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": required,
    }
    if not additional:
        schema["additionalProperties"] = False
    return schema


# 功能：MCP JSON Schema 校验通过时返回 None
# 设计：合法参数（含 required 字段）通过校验
def test_validate_params_accepts_valid() -> None:
    tool, _ = _make_tool(schema=_schema_with(["path"]))
    assert tool.validate_params({"path": "/tmp/x"}) is None


# 功能：缺少 required 参数时返回 schema_error 级错误文本
# 设计：空参数触发 required 校验，断言文本含字段名，且不包含完整 schema
def test_validate_params_required_missing() -> None:
    tool, _ = _make_tool(schema=_schema_with(["path"]))
    err = tool.validate_params({})
    assert err is not None
    assert "path" in err
    assert "required" in err
    assert "input_schema" not in err


# 功能：类型错误时返回 schema_error 级错误文本
# 设计：path 传 int 而非 string，断言错误文本指出类型问题
def test_validate_params_type_error() -> None:
    tool, _ = _make_tool(schema=_schema_with(["path"]))
    err = tool.validate_params({"path": 123})
    assert err is not None
    assert "123" in err


# 功能：additionalProperties: false 时额外字段被拒绝
# 设计：schema 禁止额外属性，传入未声明字段应报错
def test_validate_params_additional_properties_rejected() -> None:
    tool, _ = _make_tool(schema=_schema_with(additional=False))
    err = tool.validate_params({"path": "/tmp/x", "extra": "nope"})
    assert err is not None
    assert "extra" in err


# 功能：支持 $ref / anyOf 组合 schema
# 设计：构造带 $defs/$ref 与 anyOf 的 schema，验证合法参数通过、非法参数报错
def test_validate_params_ref_and_anyof() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "target": {
                "anyOf": [
                    {"$ref": "#/$defs/file_path"},
                    {"type": "string", "enum": ["latest"]},
                ]
            }
        },
        "required": ["target"],
        "$defs": {"file_path": {"type": "string", "minLength": 1}},
    }
    tool, _ = _make_tool(schema=schema)
    assert tool.validate_params({"target": "/a/b"}) is None
    assert tool.validate_params({"target": "latest"}) is None
    err = tool.validate_params({"target": 42})
    assert err is not None
    err = tool.validate_params({})
    assert err is not None


# --- invoke_tool 层：参数错误/重试行为 -----------------------------------------


async def _invoke_via_registry(tool: McpTool, params: dict[str, object]) -> object:
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    return await invoke_tool(
        registry,
        ToolCallBlock(id="t1", name=tool.name, input=params),
        bus,
        "r1",
        visible_names={tool.name},
    )


# 功能：参数校验失败时不调用 MCP Server
# 设计：缺少 required 字段经 invoke_tool 返回 schema_error，client.call_tool 零次调用
@pytest.mark.asyncio
async def test_schema_error_does_not_call_server() -> None:
    tool, client = _make_tool(schema=_schema_with(["path"]))
    result = await _invoke_via_registry(tool, {})
    assert result.is_error  # type: ignore[attr-defined]
    assert result.error_type == "schema_error"  # type: ignore[attr-defined]
    client.call_tool.assert_not_called()


# 功能：MCP 应用层错误返回 runtime_error 且不重试
# 设计：call_tool 恒抛 McpToolError，invoke_tool 应只调用一次（有副作用操作不被重复执行）
@pytest.mark.asyncio
async def test_runtime_error_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=McpToolError("boom"))
    monkeypatch.setattr("kama_claude.core.tools.invocation._RETRY_BASE_S", 0)
    result = await _invoke_via_registry(tool, {"path": "/tmp/x"})
    assert result.is_error  # type: ignore[attr-defined]
    assert result.error_type == "runtime_error"  # type: ignore[attr-defined]
    assert client.call_tool.await_count == 1


# 功能：MCP 连接错误返回 connection_error 且可恢复性重试
# 设计：call_tool 恒抛 McpServerUnavailableError，invoke_tool 应重试（_MAX_RETRIES=2，共 3 次）
@pytest.mark.asyncio
async def test_connection_error_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=McpServerUnavailableError("down"))
    monkeypatch.setattr("kama_claude.core.tools.invocation._RETRY_BASE_S", 0)
    result = await _invoke_via_registry(tool, {"path": "/tmp/x"})
    assert result.is_error  # type: ignore[attr-defined]
    assert result.error_type == "connection_error"  # type: ignore[attr-defined]
    assert client.call_tool.await_count == 3


# 功能：McpTool 的重试集合只含 connection_error 与 rate_limited
# 设计：验证类变量精确等于矩阵定义，防止 future 改动破坏边界
def test_retryable_error_types_restricted() -> None:
    assert McpTool.retryable_error_types == frozenset({"connection_error", "rate_limited"})
