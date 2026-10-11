from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # "runtime_error" | "timeout" | "schema_error" | "permission_denied"
    # | "tool_not_exposed" | "connection_error" | "rate_limited" | "search_blocked"
    error_type: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, object]
    params_model: ClassVar[type[BaseModel] | None] = None
    retryable_error_types: ClassVar[frozenset[str]] = frozenset({"runtime_error", "rate_limited"})

    # 统一校验工具参数；返回错误文本，无错误时返回 None
    def validate_params(self, params: dict[str, object]) -> str | None:
        if self.params_model is not None:
            try:
                self.params_model.model_validate(params)
            except ValidationError as exc:
                return str(exc)
        return None

    # 执行工具调用，返回结果或错误
    @abstractmethod
    async def invoke(self, params: dict[str, object]) -> ToolResult: ...
