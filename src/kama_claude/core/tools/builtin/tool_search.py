from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from kama_claude.core.mcp.exposure import McpToolExposure
from kama_claude.core.tools.base import BaseTool, ToolResult


class ToolSearchParams(BaseModel):
    query: str
    limit: int = 5


class ToolSearchTool(BaseTool):
    name = "tool_search"
    description = "Search available MCP tools by purpose, then call the activated tool next turn."
    input_schema: dict[str, object] = ToolSearchParams.model_json_schema()
    params_model = ToolSearchParams
    retryable_error_types: ClassVar[frozenset[str]] = frozenset()

    # 保存本次 run 的 exposure 控制器
    def __init__(self, exposure: McpToolExposure) -> None:
        self._exposure = exposure

    # 搜索、激活 MCP 工具并返回不带完整 schema 的摘要
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = ToolSearchParams.model_validate(params)
        query = parsed.query
        limit = max(1, min(parsed.limit, self._exposure.search_max_results))
        matches = self._exposure.search(query, limit=limit)
        if self._exposure.search_blocked:
            return ToolResult(
                content=(f"Search for '{query}' has been blocked after repeated attempts "
                         "with no new results. Use built-in tools (bash, read_file) or "
                         "answer directly."),
                is_error=True,
                error_type="search_blocked",
            )

        hints: list[str] = []
        if self._exposure.last_query_repeated:
            hints.append("this query appeared in your previous 8 valid searches")
        if self._exposure.last_search_repeated:
            hints.append("this search returned the same result as your previous one")
        hint_lines: list[str] = []
        if hints:
            hint_lines.append(
                "Note: " + " and ".join(hints)
                + ". Try a more specific keyword, use the activated tools, "
                "or use another available tool."
            )

        if not matches:
            lines = [
                f"No MCP tools found for query '{query}'. "
                "Continue with another available tool, or explain the limitation."
            ]
            lines.extend(hint_lines)
            return ToolResult(content="\n".join(lines))
        self._exposure.activate([m.name for m in matches])
        lines = [f'Found {len(matches)} MCP tools for "{query}":']
        for match in matches:
            status = "activated" if self._exposure.is_visible(match.name) else (
                f"not activated: {self._exposure.activation_failures.get(match.name, 'budget')}"
            )
            lines.append(
                f"- {match.name}  (server: {match.server})  — {match.description}  [{status}]"
            )
        lines.extend(hint_lines)
        lines.append("Use the activated tools in your next turn.")
        return ToolResult(content="\n".join(lines))
