from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from kama_claude.core.mcp.tool import McpTool


# 归一化查询大小写与空白，并保留英文 token 边界
def _normalize_query(q: str) -> str:
    return " ".join(q.casefold().split())


# 仅提取连续中文片段内部的相邻字符对
def _cjk_bigrams(s: str) -> frozenset[str]:
    bigrams: set[str] = set()
    previous: str | None = None
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            if previous is not None:
                bigrams.add(previous + ch)
            previous = ch
        else:
            previous = None
    return frozenset(bigrams)


@dataclass
class ToolSearchMatch:
    name: str
    server: str
    description: str
    activated: bool


class McpToolExposure:
    """管理一次 run 内 MCP 工具的搜索、激活和 schema 预算。"""

    # 构建独立目录和激活状态；常驻工具必须能够容纳在 lazy 预算中
    def __init__(
        self,
        catalog: list[McpTool],
        *,
        mode: Literal["lazy", "eager"],
        max_active_tools: int = 8,
        max_schema_chars: int = 12000,
        search_max_results: int = 5,
        always_expose: list[str] | None = None,
        blocked_tools: list[str] | None = None,
        reserved_names: set[str] | frozenset[str] = frozenset({"tool_search"}),
    ) -> None:
        if mode not in ("lazy", "eager"):
            raise ValueError("MCP exposure mode must be lazy or eager")
        if min(max_active_tools, max_schema_chars, search_max_results) <= 0:
            raise ValueError("MCP exposure budgets must be positive")
        self.mode = mode
        self.max_active_tools = max_active_tools
        self.max_schema_chars = max_schema_chars
        self.search_max_results = search_max_results
        self.always_expose = set(always_expose or [])
        self.blocked_tools = set(blocked_tools or [])
        self.reserved_names = set(reserved_names)
        self._tools = {t.name: t for t in catalog if t.name not in self.blocked_tools}
        self.catalog = list(self._tools.values())
        self._schema_sizes = {
            n: len(json.dumps(t.input_schema)) for n, t in self._tools.items()
        }
        self._active: set[str] = (
            set(self._tools) if mode == "eager" else self.always_expose & set(self._tools)
        )
        self._last_used: dict[str, float] = {n: time.monotonic() for n in self._active}
        self._relevance: dict[str, float] = {}
        self.activation_failures: dict[str, str] = {}
        self._search_history: list[tuple[str, frozenset[str]]] = []
        self._last_result: frozenset[str] | None = None
        self._repeat_count = 0
        self._blocked_results: set[frozenset[str]] = set()
        self.search_blocked = False
        self.last_search_repeated = False
        self.last_query_repeated = False
        self._desc_bigrams: dict[str, frozenset[str]] = {
            n: _cjk_bigrams(t.description) for n, t in self._tools.items()
        }
        if mode == "lazy" and self._budget_error(self._active) is not None:
            raise ValueError("MCP always_expose tools exceed the lazy exposure budget")

    # 搜索名称和描述，更新重复检测状态并返回简短匹配结果
    def search(self, query: str, limit: int = 5) -> list[ToolSearchMatch]:
        self.search_blocked = False
        self.last_search_repeated = False
        self.last_query_repeated = False
        tokens = [t.lower() for t in query.split() if t]
        if not tokens:
            return []
        query_bigrams = _cjk_bigrams(query)
        scored: list[tuple[float, McpTool]] = []
        for tool in self._tools.values():
            name = tool.name.lower()
            description = tool.description.lower()
            score = 0.0
            for token in tokens:
                if token in name:
                    score = max(score, 3.0 if name.startswith(token) else 2.0)
                elif token in description:
                    score = max(score, 1.0)
            if query_bigrams and query_bigrams & self._desc_bigrams[tool.name]:
                score = max(score, 0.5)
            if score:
                scored.append((score, tool))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        selected = scored[: min(max(1, limit), self.search_max_results)]

        result_set = frozenset(t.name for _, t in selected)
        norm = _normalize_query(query)
        self.last_query_repeated = any(hq == norm for hq, _ in self._search_history)
        repeated = self._last_result is not None and result_set == self._last_result
        if repeated:
            self._repeat_count += 1
            self.last_search_repeated = True
        else:
            self._repeat_count = 0
        self._last_result = result_set
        self._search_history.append((norm, result_set))
        if len(self._search_history) > 8:
            self._search_history.pop(0)
        if self._repeat_count >= 3:
            self._blocked_results.add(result_set)
        if result_set in self._blocked_results:
            self.search_blocked = True
            return []

        now = time.monotonic()
        matches = []
        for score, tool in selected:
            self._relevance[tool.name] = score
            self._last_used[tool.name] = now
            matches.append(ToolSearchMatch(
                name=tool.name,
                server=tool.name.split("__", 1)[0],
                description=tool.description[:200],
                activated=tool.name in self._active,
            ))
        return matches

    # 激活工具并在超出预算时淘汰低相关性工具
    def activate(self, names: Iterable[str]) -> list[str]:
        activated: list[str] = []
        self.activation_failures = {}
        for name in names:
            if name not in self._tools:
                continue
            if name in self._active:
                self._last_used[name] = time.monotonic()
                continue
            proposed = self._active | {name}
            protected = self.always_expose | self.reserved_names | {name}
            candidates = sorted(
                (n for n in self._active if n not in protected),
                key=lambda n: (self._relevance.get(n, 0.0), self._last_used.get(n, 0.0), n),
            )
            while candidates and self._budget_error(proposed) is not None:
                proposed.remove(candidates.pop(0))
            reason = self._budget_error(proposed)
            if reason is not None:
                self.activation_failures[name] = reason
                continue
            self._active = proposed
            activated.append(name)
            self._last_used[name] = time.monotonic()
        return [n for n in activated if n in self._active]

    # 返回指定集合超出的预算名称
    def _budget_error(self, names: set[str]) -> str | None:
        if len(names) > self.max_active_tools:
            return "max_active_tools budget"
        if sum(self._schema_sizes[n] for n in names) > self.max_schema_chars:
            return "max_schema_chars budget"
        return None

    # 统计当前 MCP 参数 schema 字符数
    @property
    def schema_chars(self) -> int:
        return sum(self._schema_sizes[n] for n in self._active)

    # 判断工具是否对本次 run 可见
    def is_visible(self, name: str) -> bool:
        return name in self.visible_names()

    # 返回独立快照，避免激活动作改变已发给模型的本轮集合
    def visible_names(self) -> set[str]:
        return self.reserved_names | set(self._active)

    # 按目录顺序生成已激活 MCP 工具的 schema
    def visible_tool_schemas(self) -> list[dict[str, object]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.catalog if t.name in self._active
        ]
