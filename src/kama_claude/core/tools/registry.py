from __future__ import annotations

from collections.abc import Iterable

from kama_claude.core.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # 注册工具；同名覆盖
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    # 按名称查找工具，不存在返回 None
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    # 返回所有工具的 Anthropic 格式 schema 列表
    def tool_schemas(self, names: Iterable[str] | None = None) -> list[dict[str, object]]:
        tools = self._tools.values() if names is None else (
            self._tools[name] for name in names if name in self._tools
        )
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in tools
        ]

    # 按注册顺序返回所有工具名
    def tool_names(self) -> list[str]:
        return list(self._tools)
