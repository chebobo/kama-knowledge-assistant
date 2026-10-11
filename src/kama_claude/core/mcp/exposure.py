from __future__ import annotations

import json
import math
import re
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


# 英文停用词：冠词/介词等没有区分能力的词，避免 a 匹配 append 产生噪声
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "to", "for", "in", "on", "at",
    "and", "or", "is", "are", "be", "it", "this", "that",
})

# 匹配类型基础分（同一查询词在多个字段命中时取最佳）
#   完整工具名 > 工具名词完整词 > 词前缀 > 工具名子串 > 描述完整词 > 描述子串 > 服务器名
_SCORE_FULL_NAME = 5.0
_SCORE_NAME_WORD = 3.0
_SCORE_NAME_PREFIX = 2.5
_SCORE_NAME_SUBSTR = 2.0
_SCORE_DESC_WORD = 1.5
_SCORE_DESC_SUBSTR = 1.0
_SCORE_SERVER = 0.8

# bigram 弱匹配：单个共享 bigram 的基础贡献与封顶（保持弱匹配等级低于主匹配）
_BIGRAM_BASE = 0.15
_BIGRAM_CAP = 0.5


def _tool_name_parts(tool_name: str) -> tuple[str, ...]:
    """write_file -> ("write", "file")，词边界拆分。"""
    return tuple(p for p in re.split(r"[\W_]+", tool_name) if p)


def _desc_words(description: str) -> frozenset[str]:
    """描述中的英文/数字词表（按词边界提取，中文不在此表）。"""
    return frozenset(re.findall(r"[a-z0-9]+", description.casefold()))


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
        # 搜索评分缓存：工具名词词表、描述原文/英文词表、token 与 bigram 文档频率
        self._name_parts: dict[str, tuple[str, ...]] = {}
        self._desc_lower: dict[str, str] = {}
        self._desc_words: dict[str, frozenset[str]] = {}
        self._token_df: dict[str, int] = {}
        self._bigram_df: dict[str, int] = {}
        for tool in self._tools.values():
            server, _, name = tool.name.partition("__")
            self._name_parts[name] = _tool_name_parts(name)
            self._desc_lower[tool.name] = tool.description.casefold()
            self._desc_words[tool.name] = _desc_words(tool.description)
            word_tokens = set(self._name_parts[name]) | {server.casefold()} | set(
                self._desc_words[tool.name]
            )
            for token in word_tokens:
                self._token_df[token] = self._token_df.get(token, 0) + 1
        self._desc_bigrams: dict[str, frozenset[str]] = {
            n: _cjk_bigrams(t.description) for n, t in self._tools.items()
        }
        for bigrams in self._desc_bigrams.values():
            for b in bigrams:
                self._bigram_df[b] = self._bigram_df.get(b, 0) + 1
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
        if mode == "lazy" and self._budget_error(self._active) is not None:
            raise ValueError("MCP always_expose tools exceed the lazy exposure budget")

    # ------------------------------------------------------------------
    # 评分函数
    # ------------------------------------------------------------------

    # 平滑 IDF：词命中工具越多越不具区分力；权重恒为正，目录大小自适应
    def _idf(self, token: str) -> float:
        df = self._token_df.get(token, 0)
        return 1.0 + 1.5 * math.log10((len(self._tools) + 1) / (df + 1))

    # 单个查询词对单个工具的最佳字段命中分；服务器名只作为弱线索
    def _token_score(self, token: str, tool: McpTool) -> float:
        server, _, name = tool.name.partition("__")
        parts = self._name_parts[name]
        if token == tool.name.casefold():
            return _SCORE_FULL_NAME
        if token in parts:
            return _SCORE_NAME_WORD
        if any(p.startswith(token) for p in parts if len(p) > len(token)):
            return _SCORE_NAME_PREFIX
        if token in name.casefold():
            return _SCORE_NAME_SUBSTR
        if token in self._desc_words[tool.name]:
            return _SCORE_DESC_WORD
        if token in self._desc_lower[tool.name]:
            return _SCORE_DESC_SUBSTR
        if token == server.casefold() or server.casefold().startswith(token):
            return _SCORE_SERVER
        return 0.0

    # 中文 bigram 弱匹配：共享更多、更少见的字符对得分更高，但封顶在弱匹配等级
    def _bigram_score(self, tool: McpTool, query_bigrams: frozenset[str]) -> float:
        shared = query_bigrams & self._desc_bigrams[tool.name]
        if not shared:
            return 0.0
        n = len(self._tools)
        total = 0.0
        for b in shared:
            df_b = self._bigram_df.get(b, 0)
            rarity = 1.0 + math.log((n + 1) / (df_b + 1)) / math.log(n + 1)
            total += _BIGRAM_BASE * rarity
        return min(_BIGRAM_CAP, total)

    # 多查询词共同评分：每个词取字段最佳分乘 IDF 后汇总，重复词不重复奖励
    def _tool_score(self, tool: McpTool, tokens: set[str], query_bigrams: frozenset[str]) -> float:
        total = 0.0
        for token in tokens:
            s = self._token_score(token, tool)
            if s > 0:
                total += s * self._idf(token)
        total += self._bigram_score(tool, query_bigrams)
        return total

    # 搜索名称和描述，更新重复检测状态并返回简短匹配结果
    def search(self, query: str, limit: int = 5) -> list[ToolSearchMatch]:
        self.search_blocked = False
        self.last_search_repeated = False
        self.last_query_repeated = False
        tokens = {
            t.casefold() for t in query.split()
            if t and t.casefold() not in _STOP_WORDS
        }
        if not tokens:
            return []
        query_bigrams = _cjk_bigrams(query)
        scored: list[tuple[float, McpTool]] = []
        for tool in self._tools.values():
            score = self._tool_score(tool, tokens, query_bigrams)
            if score > 0:
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
