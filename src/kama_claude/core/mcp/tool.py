from __future__ import annotations

from typing import Any, ClassVar

from jsonschema import Draft202012Validator

from kama_claude.core.mcp.client import (
    McpClient,
    McpServerUnavailableError,
    McpToolDef,
    McpToolError,
)
from kama_claude.core.tools.base import BaseTool, ToolResult


# 将 MCP 工具包装为 BaseTool，使 ToolRegistry 可透明调用
class McpTool(BaseTool):
    params_model = None  # input_schema 来自 MCP tool_def，不使用 pydantic model
    retryable_error_types: ClassVar[frozenset[str]] = frozenset(
        {"connection_error", "rate_limited"}
    )

    # 初始化 MCP 工具包装器，工具名以 server_name__ 为前缀防止命名冲突
    def __init__(self, client: McpClient, server_name: str, tool_def: McpToolDef) -> None:
        self._client = client
        self._server_name = server_name
        self._tool_def = tool_def
        self.name = f"{server_name}__{tool_def.name}"
        self.description = tool_def.description or f"MCP tool from {server_name}"
        self.input_schema: dict[str, Any] = (
            tool_def.input_schema or {"type": "object", "properties": {}}
        )

    # 使用 MCP 提供的 JSON Schema 校验参数，错误文本仅包含字段路径和原因
    def validate_params(self, params: dict[str, object]) -> str | None:
        if not self.input_schema:
            return None
        validator = Draft202012Validator(self.input_schema, format_checker=None)
        errors = sorted(validator.iter_errors(params), key=lambda e: tuple(str(p) for p in e.path))
        if not errors:
            return None
        lines = [
            f"- {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors
        ]
        return f"Invalid arguments for {self.name}:\n" + "\n".join(lines)

    # 调用 MCP server 上的工具，连接不可用或工具执行失败时返回 is_error=True
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        try:
            content = await self._client.call_tool(self._tool_def.name, dict(params))
            return ToolResult(content=content)
        except McpServerUnavailableError as exc:
            return ToolResult(
                content=f"mcp server '{self._server_name}' unavailable: {exc}",
                is_error=True,
                error_type="connection_error",
            )
        except McpToolError as exc:
            return ToolResult(
                content=f"mcp tool '{self.name}' error: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        except Exception as exc:
            return ToolResult(
                content=f"mcp tool '{self.name}' unexpected error: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
