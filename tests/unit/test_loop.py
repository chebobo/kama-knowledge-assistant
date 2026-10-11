from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from kama_claude.core.loop import AgentLoop
from kama_claude.core.mcp.client import McpClient, McpToolDef
from kama_claude.core.mcp.exposure import McpToolExposure
from kama_claude.core.mcp.tool import McpTool
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.tool_search import ToolSearchTool
from kama_claude.core.tools.registry import ToolRegistry

# --- stubs -------------------------------------------------------------------


class _MockProvider:
    """Returns canned responses in order; raises exc immediately if given."""

    def __init__(
        self,
        responses: list[LlmResponse],
        exc: BaseException | None = None,
    ) -> None:
        self._responses = iter(responses)
        self._exc = exc

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if self._exc is not None:
            raise self._exc
        return next(self._responses)


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes msg"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


class _FailTool(BaseTool):
    name = "fail"
    description = "Always raises"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("tool error")


# --- helpers -----------------------------------------------------------------


def _ctx(max_steps: int = 5) -> ExecutionContext:
    return ExecutionContext(run_id="r1", goal="test goal", max_steps=max_steps)


def _tc(name: str = "echo", inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {"msg": "hi"})


def _make_loop(
    provider: _MockProvider,
    registry: ToolRegistry | None = None,
    bus: EventBus | None = None,
) -> tuple[AgentLoop, EventBus]:
    b = bus or EventBus()
    return AgentLoop(provider, registry or ToolRegistry(), b), b  # type: ignore[arg-type]


async def _events(bus: EventBus) -> list[BaseModel]:
    collected: list[BaseModel] = []

    async def _h(e: BaseModel) -> None:
        collected.append(e)

    bus.subscribe(_h)
    return collected


# --- MCP exposure 相关 stub 与 helper ------------------------------------------


class _RecordingProvider:
    """记录每轮收到的 tool_schemas 与 messages，按序返回固定响应。"""

    def __init__(self, responses: list[LlmResponse]) -> None:
        self._responses = iter(responses)
        self.seen_schemas: list[list[str]] = []
        self.seen_messages: list[list[dict[str, object]]] = []

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self.seen_schemas.append([s["name"] for s in tool_schemas])
        self.seen_messages.append(messages)
        return next(self._responses)


def _make_mcp_tool() -> tuple[McpTool, AsyncMock]:
    client = AsyncMock(spec=McpClient)
    tool = McpTool(
        client,
        "fs",
        McpToolDef(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    )
    return tool, client


MCP_TOOL_NAME = "fs__read_file"


def _make_mcp_loop(
    responses: list[LlmResponse],
    *,
    mode: str = "lazy",
) -> tuple[AgentLoop, _RecordingProvider, AsyncMock, McpTool, McpToolExposure]:
    mcp_tool, client = _make_mcp_tool()
    exposure = McpToolExposure([mcp_tool], mode=mode)  # type: ignore[arg-type]
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(mcp_tool)
    registry.register(ToolSearchTool(exposure))
    provider = _RecordingProvider(responses)
    loop = AgentLoop(provider, registry, EventBus(), exposure=exposure)
    return loop, provider, client, mcp_tool, exposure


# --- tests -------------------------------------------------------------------


# 功能：验证 LLM 返回 end_turn 时 loop 将 context 标记为 success
# 设计：单步 provider 直接返回 end_turn，最简正常路径，确认 loop 的基本终止逻辑
async def test_end_turn_marks_success() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="done")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 1


# 功能：验证达到 max_steps 时 loop 以 exceeded_max_steps 原因将 context 标记为 failed
# 设计：设置 max_steps=2 + 无限 tool_use provider，同时验证 step 数量和失败原因，确认计数器与终止逻辑联动正确
async def test_max_steps_marks_failed() -> None:
    tc = _tc("unknown", {})
    provider = _MockProvider([LlmResponse(stop_reason="tool_use", tool_calls=[tc])] * 10)
    loop, _ = _make_loop(provider)
    ctx = _ctx(max_steps=2)
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "exceeded_max_steps"
    assert ctx.step == 2


# 功能：验证"调工具 → end_turn"的两步路径最终标记为 success
# 设计：provider 返回 [tool_use, end_turn] 序列，注册真实 EchoTool，覆盖最常见的正常工作路径
async def test_tool_use_then_end_turn_marks_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn", text="summary"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具结果按 Anthropic 格式（tool_result user 消息）追加到消息历史
# 设计：检查 messages[2]（tool_result 所在位置），断言 tool_use_id 和 content，确认 loop 正确调用了 context.add_tool_result
async def test_tool_result_appended_to_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc(inp={"msg": "hello"})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    # messages: [goal, assistant(tool_use), user(tool_result), assistant(end_turn)]
    tool_result_msg = ctx.messages[2]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "hello"


# 功能：验证工具失败时 loop 不终止，而是将错误追加上下文让 LLM 重新决策
# 设计：工具始终 raise + provider 第二步返回 end_turn，确认 loop 最终到达 success；这是 agent 区别于普通脚本的核心特性
async def test_tool_failure_loop_continues_to_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn", text="handled error"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具失败的错误信息以 is_error=True 追加进上下文，让 LLM 能感知工具调用失败
# 设计：检查 tool_result block 中的 is_error 标记，与 test_tool_failure_loop_continues_to_success 互补
async def test_tool_failure_result_is_error_in_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    tool_result_msg = ctx.messages[2]
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block.get("is_error") is True


# 功能：验证收到 CancelledError 时 loop 将 context 标记为 cancelled 后继续上抛 CancelledError
# 设计：用 pytest.raises 捕获 CancelledError，同时检查 context.status，确认优雅退出行为：先记录状态，再传播取消信号
async def test_cancelled_error_marks_failed_and_reraises() -> None:
    provider = _MockProvider([], exc=asyncio.CancelledError())
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    with pytest.raises(asyncio.CancelledError):
        await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "cancelled"


# 功能：验证 LLM 调用异常被捕获并标记为 llm_error，不向上传播
# 设计：provider 抛 RuntimeError，确认 loop 不崩溃、context 状态为 failed/llm_error，异常被正确吸收
async def test_llm_api_error_marks_failed() -> None:
    provider = _MockProvider([], exc=RuntimeError("api error"))
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "llm_error"


# 功能：验证每个步骤都发布 step.started 和 step.finished 事件
# 设计：注入 bus + 事件收集器，检查事件类型集合，确认步骤级事件的可观测性（S2 TUI 依赖这两个事件显示进度）
async def test_step_started_and_finished_events_published() -> None:
    bus = EventBus()
    events = await _events(bus)
    provider = _MockProvider([LlmResponse(stop_reason="end_turn")])
    loop, _ = _make_loop(provider, bus=bus)
    ctx = _ctx()
    await loop.run(ctx)
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "step.started" in types
    assert "step.finished" in types


# 功能：验证多步执行后 step 计数器正确累积到步数总量
# 设计：三步序列 [tool_use, tool_use, end_turn]，确认 step==3，排除计数器初始化错误或某步未递增的情况
async def test_step_counter_increments_across_steps() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx(max_steps=10)
    await loop.run(ctx)
    assert ctx.step == 3
    assert ctx.status == "success"


# 功能：验证 LLM 文本响应以正确的 content block 格式追加到消息历史
# 设计：检查 messages[1] 的 role 和 content block 结构，确认 loop 构造的 assistant 消息符合 Anthropic 格式
async def test_assistant_message_blocks_added_to_context() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="answer")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assistant_msg = ctx.messages[1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    assert blocks[0]["type"] == "text"  # type: ignore[index]
    assert blocks[0]["text"] == "answer"  # type: ignore[index]


# --- MCP 按需暴露：可见子集、隐藏拒绝、搜索激活、截断、压缩 ---------------------


# 功能：lazy 模式下 Provider 首轮只收到内置工具与 tool_search，不含未激活 MCP 工具
# 设计：exposure 目录含一个 MCP 工具但未激活，断言 provider 首轮 schema 名集合
async def test_provider_receives_visible_subset_only() -> None:
    loop, provider, client, mcp_tool, _ = _make_mcp_loop(
        [LlmResponse(stop_reason="end_turn")]
    )
    ctx = _ctx()
    await loop.run(ctx)
    first = provider.seen_schemas[0]
    assert "echo" in first
    assert "tool_search" in first
    assert mcp_tool.name not in first


# 功能：模型猜测隐藏 MCP 工具名时返回 tool_not_exposed，不进入权限审批、不调用 MCP
# 设计：第一轮模型直接调用未激活 MCP 工具，断言权限 mock 与 client 均未被调用，错误进上下文
async def test_hidden_tool_rejected_without_permission_or_mcp() -> None:
    loop, provider, client, mcp_tool, _ = _make_mcp_loop([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc(MCP_TOOL_NAME, {"path": "/x"})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    permission = MagicMock()
    loop._permission_manager = permission  # type: ignore[attr-defined]
    ctx = _ctx()
    await loop.run(ctx)
    permission.check_and_wait.assert_not_called()
    client.call_tool.assert_not_called()
    block = ctx.messages[2]["content"][0]  # type: ignore[index]
    assert block["is_error"] is True
    assert "not exposed" in block["content"]


# 功能：tool_search 激活的工具必须下一轮才可见、可执行
# 设计：第一轮模型调 tool_search 搜索 "read"，第二轮模型调 filesystem__read_file，
#       断言首轮 schema 不含该工具、第二轮含，且 MCP 仅在第二轮被真正调用
async def test_tool_search_activation_visible_next_turn() -> None:
    search_tc = ToolCallBlock(id="s1", name="tool_search", input={"query": "read"})
    read_tc = ToolCallBlock(id="r1", name=MCP_TOOL_NAME, input={"path": "/tmp/x"})
    loop, provider, client, mcp_tool, _ = _make_mcp_loop([
        LlmResponse(stop_reason="tool_use", tool_calls=[search_tc]),
        LlmResponse(stop_reason="tool_use", tool_calls=[read_tc]),
        LlmResponse(stop_reason="end_turn"),
    ])
    ctx = _ctx(max_steps=10)
    await loop.run(ctx)
    assert mcp_tool.name not in provider.seen_schemas[0]
    assert mcp_tool.name in provider.seen_schemas[1]
    client.call_tool.assert_awaited_once()
    assert ctx.status == "success"


# 功能：同一轮同时返回 tool_search 与隐藏 MCP 工具时，隐藏工具仍被拒绝
# 设计：第一轮 provider 同时给出搜索和隐藏调用；搜索正常执行，隐藏调用返回 not_exposed
async def test_same_turn_search_and_hidden_tool_rejected() -> None:
    search_tc = ToolCallBlock(id="s1", name="tool_search", input={"query": "read"})
    read_tc = ToolCallBlock(id="r1", name=MCP_TOOL_NAME, input={"path": "/x"})
    loop, provider, client, mcp_tool, _ = _make_mcp_loop([
        LlmResponse(stop_reason="tool_use", tool_calls=[search_tc, read_tc]),
        LlmResponse(stop_reason="end_turn"),
    ])
    ctx = _ctx()
    await loop.run(ctx)
    client.call_tool.assert_not_called()
    blocks = ctx.messages[2]["content"]  # type: ignore[index]
    assert blocks[0].get("is_error") is not True  # tool_search 正常
    assert "found" in blocks[0]["content"].lower() or "mcp tools" in blocks[0]["content"]
    assert blocks[1]["is_error"] is True  # 隐藏工具被拒
    assert "not exposed" in blocks[1]["content"]


# 功能：超长工具结果只截断发给 Provider 的副本，context 保留完整内容
# 设计：echo 返回 10000 字符结果，loop 用小预算截断；断言 provider 收到截断版且含标记，context 完整
async def test_tool_result_truncated_for_provider_but_kept_in_context() -> None:
    long_msg = "x" * 10_000
    mcp_tool, client = _make_mcp_tool()
    exposure = McpToolExposure([mcp_tool], mode="lazy")  # type: ignore[arg-type]
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(mcp_tool)
    provider = _RecordingProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc(inp={"msg": long_msg})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    loop = AgentLoop(
        provider, registry, EventBus(),
        exposure=exposure,
        tool_result_limit=100,
        tool_result_keep=50,
    )
    ctx = _ctx()
    await loop.run(ctx)
    sent = provider.seen_messages[1]
    sent_block = sent[2]["content"][0]  # type: ignore[index]
    assert len(sent_block["content"]) < 1000
    assert "Full output is available in run events" in sent_block["content"]
    ctx_block = ctx.messages[2]["content"][0]  # type: ignore[index]
    assert ctx_block["content"] == long_msg


# 功能：缓存 token 使 context_pct 达到阈值时 Compactor 被触发
# 设计：provider 返回带 usage.context_pct=0.9 的 tool_use 响应，mock compactor，断言 compact 被调用
async def test_compactor_triggered_by_cache_token_context_pct() -> None:
    mcp_tool, _ = _make_mcp_tool()
    exposure = McpToolExposure([mcp_tool], mode="lazy")  # type: ignore[arg-type]
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(mcp_tool)
    usage = UsageStats(
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=30,
        context_pct=0.9,
    )
    provider = _RecordingProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()], usage=usage),
        LlmResponse(stop_reason="end_turn"),
    ])
    compactor = MagicMock()
    compactor.compact = AsyncMock()
    loop = AgentLoop(
        provider, registry, EventBus(),
        exposure=exposure,
        compactor=compactor,  # type: ignore[arg-type]
        compact_threshold=0.8,
    )
    ctx = _ctx()
    await loop.run(ctx)
    compactor.compact.assert_awaited_once()


# 功能：100 个 MCP 工具时 lazy 模式首轮只暴露基础工具、tool_search 与常驻工具
# 设计：构造 100 个未激活 MCP 工具 + 1 个 always_expose，断言 provider 首轮 schema 集合
async def test_hundred_tools_lazy_first_turn_baseline() -> None:
    tools = []
    for i in range(100):
        client = AsyncMock(spec=McpClient)
        tools.append(
            McpTool(
                client,
                "srv",
                McpToolDef(
                    name=f"tool{i}",
                    description=f"Tool {i}",
                    input_schema={"type": "object", "properties": {}},
                ),
            )
        )
    exposure = McpToolExposure(
        tools,  # type: ignore[arg-type]
        mode="lazy",
        always_expose=["srv__tool0"],
    )
    registry = ToolRegistry()
    registry.register(_EchoTool())
    for t in tools:
        registry.register(t)
    registry.register(ToolSearchTool(exposure))
    provider = _RecordingProvider([LlmResponse(stop_reason="end_turn")])
    loop = AgentLoop(provider, registry, EventBus(), exposure=exposure)
    ctx = _ctx()
    await loop.run(ctx)
    first = provider.seen_schemas[0]
    assert "echo" in first
    assert "tool_search" in first
    assert "srv__tool0" in first  # always_expose 常驻
    assert "srv__tool1" not in first
    assert len([n for n in first if n.startswith("srv__")]) == 1
