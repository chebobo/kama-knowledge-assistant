from __future__ import annotations

import json
from itertools import count
from unittest.mock import AsyncMock

import pytest

from kama_claude.core.mcp.client import McpClient, McpToolDef
from kama_claude.core.mcp.exposure import McpToolExposure, _cjk_bigrams
from kama_claude.core.mcp.tool import McpTool


# 构造不依赖真实服务器的 MCP 工具
def make_tool(name: str, description: str = "Read data", schema: dict | None = None,
              server: str = "files") -> McpTool:
    return McpTool(AsyncMock(spec=McpClient), server, McpToolDef(
        name, description, schema if schema is not None else {"type": "object"},
    ))


# 功能：lazy 仅暴露常驻工具，eager 全量暴露，blocked 优先于常驻
# 设计：使用同一目录对比两种模式，并验证未知常驻名称不会泄漏
@pytest.mark.parametrize("mode", ["lazy", "eager"])
def test_modes_and_blocked_precedence(mode: str) -> None:
    tools = [make_tool(n) for n in ("read", "write", "delete")]
    exposure = McpToolExposure(
        tools, mode=mode, always_expose=["files__read", "files__delete", "missing"],
        blocked_tools=["files__delete"],
    )
    expected = {"tool_search", "files__read"}
    if mode == "eager":
        expected.add("files__write")
    assert exposure.visible_names() == expected
    assert exposure.activate(["files__delete", "missing"]) == []
    assert exposure.search("delete") == []
    assert [t.name for t in exposure.catalog] == ["files__read", "files__write"]


# 功能：搜索按名称前缀、名称子串、描述排序，限制结果并省略 schema
# 设计：三个工具分别命中不同字段，同时覆盖大小写、空查询和多 token
def test_search_ranking_and_cap() -> None:
    exposure = McpToolExposure([
        make_tool("read"), make_tool("files_backup", server="archive"),
        make_tool("plain", "files", server="archive"),
    ], mode="lazy", search_max_results=2)
    matches = exposure.search("FILES", limit=99)
    assert [m.name for m in matches] == ["files__read", "archive__files_backup"]
    assert all(not m.activated for m in matches)
    assert exposure.search("unmatched") == []
    assert exposure.search("   ") == []
    assert len(exposure.search("files__read unmatched")) == 1


# 功能：搜索和激活状态按 run 隔离，快照不会随激活改变
# 设计：两个 exposure 共享工具实例，仅修改其中一个的激活集合
def test_run_isolation_and_snapshot() -> None:
    tools = [make_tool("read")]
    a = McpToolExposure(tools, mode="lazy")
    b = McpToolExposure(tools, mode="lazy")
    snapshot = a.visible_names()
    assert a.activate(iter(["files__read"])) == ["files__read"]
    assert a.is_visible("files__read")
    assert not b.is_visible("files__read")
    assert "files__read" not in snapshot
    assert a.search("read")[0].activated
    assert [s["name"] for s in a.visible_tool_schemas()] == ["files__read"]


# 功能：数量预算按相关性优先淘汰，相关性相同按使用时间淘汰
# 设计：控制时钟，搜索提高旧工具相关度，再加入新工具观察淘汰结果
def test_count_budget_relevance_then_age(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = count(1)
    monkeypatch.setattr("kama_claude.core.mcp.exposure.time.monotonic", lambda: next(clock))
    tools = [make_tool(n) for n in ("old", "relevant", "new", "last")]
    exposure = McpToolExposure(tools, mode="lazy", max_active_tools=2)
    exposure.activate(["files__old", "files__relevant"])
    exposure.search("relevant")
    exposure.activate(["files__new"])
    assert exposure.visible_names() == {"tool_search", "files__relevant", "files__new"}
    # 新实例所有工具同相关度，旧工具优先被淘汰。
    other = McpToolExposure(tools, mode="lazy", max_active_tools=2)
    other.activate(["files__old", "files__new", "files__last"])
    assert other.visible_names() == {"tool_search", "files__new", "files__last"}


# 功能：schema 预算能淘汰普通工具，但不会淘汰常驻工具或破坏拒绝前状态
# 设计：单个工具预算只能容纳一个 schema，再尝试无法放入的超大 schema
def test_schema_budget_and_atomic_rejection() -> None:
    tools = [make_tool("a"), make_tool("b"), make_tool("big", schema={"description": "x" * 500})]
    budget = len(json.dumps(tools[0].input_schema))
    exposure = McpToolExposure(tools, mode="lazy", max_schema_chars=budget)
    exposure.activate(["files__a"])
    assert exposure.activate(["files__b"]) == ["files__b"]
    assert not exposure.is_visible("files__a")
    before = exposure.visible_names()
    assert exposure.activate(["files__big"]) == []
    assert exposure.visible_names() == before
    assert exposure.schema_chars <= budget
    assert exposure.activation_failures["files__big"] == "max_schema_chars budget"
    pinned = McpToolExposure(tools, mode="lazy", max_active_tools=1, always_expose=["files__a"])
    assert pinned.activate(["files__b"]) == []
    assert pinned.is_visible("files__a")
    assert pinned.activation_failures["files__b"] == "max_active_tools budget"


# 功能：常驻工具预算冲突明确失败，eager 保持全量兼容
# 设计：构造两个常驻工具与一个工具数量上限，覆盖初始状态边界
def test_pinned_budget_overflow_and_eager_compatibility() -> None:
    tools = [make_tool("a"), make_tool("b")]
    with pytest.raises(ValueError, match="always_expose"):
        McpToolExposure(tools, mode="lazy", max_active_tools=1, always_expose=[t.name for t in tools])
    eager = McpToolExposure(tools, mode="eager", max_active_tools=1, max_schema_chars=1)
    assert all(eager.is_visible(t.name) for t in tools)


# 功能：连续三次重复结果后在第四次搜索内拦截
# 设计：重复调用同一 query，逐次检查提示标志、计数和硬拦截时点
def test_search_repeat_guard_blocks_on_fourth_call() -> None:
    exposure = McpToolExposure([make_tool("read")], mode="lazy")
    first = exposure.search("read")
    assert not exposure.last_search_repeated and not exposure.last_query_repeated
    second = exposure.search("read")
    assert exposure.last_search_repeated and exposure.last_query_repeated
    assert second == first and not exposure.search_blocked
    third = exposure.search("read")
    assert exposure.last_search_repeated and not exposure.search_blocked
    fourth = exposure.search("read")
    assert first and second and third
    assert exposure.last_search_repeated
    assert fourth == []
    assert exposure.search_blocked
    assert exposure._repeat_count == 3


# 功能：不同 query 命中相同结果集合时仍计入连续重复
# 设计：让名称和描述分别提供两个 query，验证判定依据是集合而不是 query 文本
def test_same_result_set_from_different_queries_is_repeated() -> None:
    exposure = McpToolExposure([make_tool("read", "read data")], mode="lazy")
    for query in ("read", "data", "read", "data"):
        exposure.search(query)
    assert exposure.search_blocked
    assert exposure._repeat_count == 3


# 功能：连续空结果也会在第四次被拦截
# 设计：使用四个不同的不存在词，覆盖语义近似词刷不存在工具的状态机边界
def test_empty_result_repeat_guard() -> None:
    exposure = McpToolExposure([make_tool("read")], mode="lazy")
    for query in ("missing-a", "missing-b", "missing-c", "missing-d"):
        result = exposure.search(query)
    assert result == []
    assert exposure.search_blocked
    assert exposure._last_result == frozenset()


# 功能：换到新结果集合后重置重复计数，并保留被禁集合
# 设计：先达到 A 阈值，再验证 B 可搜索且最终回到 A 仍被拦截
def test_multiple_blocked_result_sets_are_independent() -> None:
    exposure = McpToolExposure([make_tool("a"), make_tool("b")], mode="lazy")
    for _ in range(4):
        exposure.search("files__a")
    assert exposure.search_blocked
    for _ in range(4):
        exposure.search("files__b")
    assert exposure.search_blocked
    assert exposure._blocked_results == {frozenset({"files__a"}), frozenset({"files__b"})}
    assert exposure.search("files__a") == []
    assert exposure.search_blocked


# 功能：非空结果切换为空时不把首次空结果算作重复
# 设计：分别断言 A、空、空 三次调用后的内部计数为 0、0、1
def test_nonempty_to_empty_starts_new_repeat_sequence() -> None:
    exposure = McpToolExposure([make_tool("read")], mode="lazy")
    assert exposure.search("read")
    assert exposure._repeat_count == 0
    assert exposure.search("missing") == []
    assert exposure._repeat_count == 0
    exposure.search("missing-again")
    assert exposure._repeat_count == 1
    assert exposure.last_search_repeated


# 功能：纯空白 query 只复位公开标志，不写入搜索状态
# 设计：在两次有效搜索之间插入空白调用，比较历史长度和重复计数保持不变
def test_blank_query_does_not_change_search_state() -> None:
    exposure = McpToolExposure([make_tool("read")], mode="lazy")
    exposure.search("read")
    exposure.search("read")
    before = (len(exposure._search_history), exposure._repeat_count, exposure._last_result)
    assert exposure.search("   \t") == []
    assert not exposure.search_blocked
    assert not exposure.last_search_repeated
    assert not exposure.last_query_repeated
    assert (len(exposure._search_history), exposure._repeat_count, exposure._last_result) == before


# 功能：query 历史排除当前调用并限制为最近八次有效搜索
# 设计：验证首次不误报、A-B-A 独立提示、超出窗口后旧 query 可重新出现
def test_query_history_window_and_current_call_exclusion() -> None:
    exposure = McpToolExposure([make_tool("a"), make_tool("b")], mode="lazy")
    exposure.search("files__a")
    assert not exposure.last_query_repeated
    exposure.search("files__b")
    exposure.search("files__a")
    assert exposure.last_query_repeated
    for index in range(8):
        exposure.search(f"new-{index}")
    assert len(exposure._search_history) == 8
    exposure.search("files__a")
    assert not exposure.last_query_repeated


# 功能：中文近似措辞能通过 bigram 弱匹配命中
# 设计：在描述中增加一个助词破坏完整子串，直接检查 0.5 分弱匹配
def test_cjk_bigram_weak_match_and_ranking() -> None:
    exposure = McpToolExposure([
        make_tool("a_correct", "读取文件的内容"),
    ], mode="lazy")
    matches = exposure.search("读取文件内容", limit=1)
    assert [match.name for match in matches] == ["files__a_correct"]
    assert exposure._relevance["files__a_correct"] == 0.5


# 功能：bigram 不改变原英文子串匹配，中文边界仍能提供正常弱匹配
# 设计：保留 xy 对 xyz 的既有 1 分命中，同时检查三类中文边界查询的弱匹配
def test_cjk_bigram_boundaries() -> None:
    exposure = McpToolExposure([
        make_tool("target", "读取文件内容"),
        make_tool("unrelated", "下载 xyz 数据"),
    ], mode="lazy")
    # 规格此例的“不命中”与保留原子串规则冲突，原始失败见交接日志。
    assert [m.name for m in exposure.search("查找 xy")] == ["files__unrelated"]
    assert exposure._relevance["files__unrelated"] == 1.0
    assert exposure.search("读取，文件", limit=1)[0].name == "files__target"
    assert exposure.search("读取123文件", limit=1)[0].name == "files__target"
    assert exposure.search("读取PDF文件", limit=1)[0].name == "files__target"


# 功能：英文搜索保持原有匹配行为，中文片段不足两个字符时不启用 n-gram
# 设计：分别检查英文名称排序和无相邻中文字符时不凭跨边界字符召回
def test_english_behavior_and_short_cjk_query() -> None:
    exposure = McpToolExposure([
        make_tool("read"),
        make_tool("reader"),
        make_tool("plain", "档案"),
    ], mode="lazy")
    assert [match.name for match in exposure.search("read")] == ["files__read", "files__reader"]
    assert exposure.search("文 x 件") == []


# 功能：新结果集合重置计数，既有禁集不会影响其他集合
# 设计：先累积两次重复再换方向，并在旧集合被禁后验证第三个集合仍可返回
def test_new_result_resets_count_and_remains_searchable() -> None:
    exposure = McpToolExposure([make_tool(n) for n in ("a", "b", "c")], mode="lazy")
    for _ in range(3):
        exposure.search("files__a")
    assert exposure._repeat_count == 2
    assert exposure.search("files__b")
    assert exposure._repeat_count == 0 and not exposure.search_blocked
    for _ in range(3):
        exposure.search("files__b")
    assert exposure.search_blocked
    assert exposure.search("files__c")
    assert exposure._repeat_count == 0 and not exposure.search_blocked


# 功能：query 归一化保留 token 边界，query 与结果重复独立判断
# 设计：大小写和空白变体视为同一 query，合并 token 后则是新 query
def test_normalization_and_independent_repeat_flags() -> None:
    exposure = McpToolExposure([make_tool("readfile", "read file")], mode="lazy")
    exposure.search(" READ   FILE ")
    assert not exposure.last_query_repeated and not exposure.last_search_repeated
    exposure.search("read file")
    assert exposure.last_query_repeated and exposure.last_search_repeated
    exposure.search("readfile")
    assert not exposure.last_query_repeated and exposure.last_search_repeated


# 功能：达到阈值和命中禁集的调用也更新历史、基准且历史淘汰不会解禁
# 设计：禁用 A 后穿插八个新 query，验证窗口外 A 不触发 query 提示但仍被拦截
def test_blocked_calls_update_history_and_outlive_query_window() -> None:
    exposure = McpToolExposure([make_tool("a"), make_tool("b")], mode="lazy")
    for _ in range(4):
        exposure.search("files__a")
    assert len(exposure._search_history) == 4
    exposure.search("files__b")
    exposure.search("files__a")
    assert exposure.search_blocked and not exposure.last_search_repeated
    assert exposure._search_history[-1] == ("files__a", frozenset({"files__a"}))
    assert exposure._last_result == frozenset({"files__a"})
    for index in range(8):
        exposure.search(f"missing-{index}")
    assert len(exposure._search_history) == 8
    assert exposure.search("files__a") == []
    assert exposure.search_blocked and not exposure.last_query_repeated
    assert exposure._last_result == frozenset({"files__a"})


# 功能：每次 run 的搜索计数、禁集和历史互不共享
# 设计：两个实例共享目录对象，仅对第一个达到阈值再检查第二个首次搜索
def test_search_guard_run_isolation() -> None:
    tools = [make_tool("read")]
    a = McpToolExposure(tools, mode="lazy")
    b = McpToolExposure(tools, mode="lazy")
    for _ in range(4):
        a.search("read")
    assert a.search_blocked
    assert b.search("read")
    assert not b.search_blocked and not b.last_query_repeated and not b.last_search_repeated


# 功能：中文 bigram 不跨任何非中文边界，也不生成英文 bigram
# 设计：直接断言辅助函数输出，并用只有跨边界字符对的工具检验未误召回
@pytest.mark.parametrize("query", ["读取，文件", "读取123文件", "读取PDF文件", "读取 文件"])
def test_cjk_bigram_exact_boundaries(query: str) -> None:
    assert _cjk_bigrams(query) == frozenset({"读取", "文件"})
    assert _cjk_bigrams("查找 xy") == frozenset({"查找"})
    assert _cjk_bigrams("文x件") == frozenset()
    exposure = McpToolExposure([make_tool("cross_only", "取文")], mode="lazy")
    assert exposure.search(query) == []


# 功能：完整描述主匹配优先于中文弱匹配，同分仍按工具名排序
# 设计：让噪声名称排在正确工具之前，limit=1 时必须选择主匹配
def test_complete_description_outranks_bigram_noise() -> None:
    exposure = McpToolExposure([
        make_tool("a_noise", "查找文件内容"),
        make_tool("z_correct", "查找股票行情"),
    ], mode="lazy")
    assert [m.name for m in exposure.search("查找股票行情", limit=1)] == ["files__z_correct"]
    assert exposure._relevance["files__z_correct"] == 1.0
    exposure.search("查找股票行情", limit=2)
    assert exposure._relevance["files__a_noise"] == 0.5


# 功能：名称和描述主匹配与 bigram 同时命中时使用 max 而不加分
# 设计：构造 3、2、1 三个等级并检查分数及确定性顺序完全保留
def test_primary_scores_are_not_added_to_bigram_score() -> None:
    exposure = McpToolExposure([
        make_tool("查找", "查找内容", server="查找"),
        make_tool("查找", "查找内容"),
        make_tool("plain", "查找内容"),
    ], mode="lazy")
    assert [m.name for m in exposure.search("查找")] == [
        "查找__查找", "files__查找", "files__plain",
    ]
    assert exposure._relevance == {"查找__查找": 3.0, "files__查找": 2.0, "files__plain": 1.0}


# 功能：混合语言只通过原有英文 token 匹配，不从英文部分生成弱匹配
# 设计：英文片段只是无关描述的前缀，完整混合 token 不匹配且中文 bigram 不相交
def test_mixed_language_does_not_generate_english_bigrams() -> None:
    exposure = McpToolExposure([make_tool("unrelated", "下载 xyz 数据")], mode="lazy")
    assert exposure.search("查找xy") == []


# 功能：结果集合忽略排序但只覆盖最终 Top-K，目录在构造时缓存
# 设计：交换两工具排名不改变集合，缩小 limit 改变集合并验证后加工具不进入目录
def test_result_set_ignores_order_and_uses_top_k_snapshot() -> None:
    tools = [make_tool("alpha", "beta"), make_tool("beta", "alpha")]
    exposure = McpToolExposure(tools, mode="lazy")
    exposure.search("alpha", limit=2)
    exposure.search("beta", limit=2)
    assert exposure.last_search_repeated
    assert exposure.search("alpha", limit=1)
    assert not exposure.last_search_repeated
    tools.append(make_tool("new", "新增中文"))
    assert exposure.search("新增中文") == []


# 功能：拦截后的空白调用复位本次标志但不解除禁集或增加历史
# 设计：先触发三种公开标志，再插入空白请求并重新验证原集合依然被禁
def test_blank_after_block_keeps_guard_state() -> None:
    exposure = McpToolExposure([make_tool("read")], mode="lazy")
    for _ in range(4):
        exposure.search("read")
    assert exposure.search_blocked and exposure.last_query_repeated and exposure.last_search_repeated
    history = list(exposure._search_history)
    blocked = set(exposure._blocked_results)
    assert exposure.search(" \t\n") == []
    assert not exposure.search_blocked and not exposure.last_query_repeated
    assert not exposure.last_search_repeated
    assert exposure._search_history == history and exposure._blocked_results == blocked
    assert exposure._repeat_count == 3
    assert exposure.search("read") == []
    assert exposure.search_blocked
