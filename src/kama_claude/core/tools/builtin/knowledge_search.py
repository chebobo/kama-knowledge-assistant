from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from kama_claude.core.knowledge.service import KnowledgeService
from kama_claude.core.tools.base import BaseTool, ToolResult


class KnowledgeSearchParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    top_k: int | None = None


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = (
        "Search global knowledge and the current project knowledge. "
        "Scope is inherited from the current Agent."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Knowledge search query"},
            "top_k": {"type": "integer", "minimum": 1, "description": "Optional result count"},
        },
        "required": ["query"],
    }
    params_model = KnowledgeSearchParams

    # 绑定 KnowledgeService 和当前 Agent 的不可变 scope
    def __init__(
        self,
        service: KnowledgeService | None,
        *,
        user_id: str | None,
        project_id: str | None,
        default_top_k: int = 8,
    ) -> None:
        self._service = service
        self._user_id = user_id
        self._project_id = project_id
        self._default_top_k = default_top_k

    # 执行当前 scope 的知识检索，服务不可用时返回可识别错误
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = KnowledgeSearchParams.model_validate(params)
        if not parsed.query.strip():
            return ToolResult("Knowledge search query must not be empty.", True, "schema_error")
        if self._service is None:
            return ToolResult("Knowledge service is unavailable.", True, "runtime_error")
        top_k = parsed.top_k or self._default_top_k
        if top_k <= 0:
            return ToolResult("top_k must be positive.", True, "schema_error")
        top_k = min(top_k, self._default_top_k)
        try:
            content = await self._service.retrieve_context(
                parsed.query, user_id=self._user_id, project_id=self._project_id, top_k=top_k
            )
        except Exception as exc:
            return ToolResult(f"Knowledge search failed: {exc}", True, "runtime_error")
        return ToolResult(content or "No matching knowledge found.")
