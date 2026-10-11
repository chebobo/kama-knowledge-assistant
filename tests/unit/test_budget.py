from __future__ import annotations

from kama_claude.core.compact.budget import truncate_tool_results


def _make_tool_result_msg(content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "id1", "content": content}],
    }


# 功能：验证 tool_result 内容未超过阈值时原文不变
# 设计：构造 7999 字符内容（刚好低于 8000），断言消息原样返回
def test_short_tool_result_untouched() -> None:
    text = "x" * 7999
    msgs = [_make_tool_result_msg(text)]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    assert result[0]["content"][0]["content"] == text


# 功能：验证 tool_result 内容超过阈值时被截断并附加省略标记
# 设计：构造 10000 字符内容，断言截断后长度 < 原始，且包含省略标记字符串
def test_long_tool_result_truncated() -> None:
    text = "y" * 10_000
    msgs = [_make_tool_result_msg(text)]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    truncated = result[0]["content"][0]["content"]
    assert len(truncated) < len(text)
    assert "chars omitted" in truncated
    assert truncated.startswith("y" * 4000)


# 功能：验证 tool_result 内容恰好等于阈值时不截断
# 设计：构造恰好 8000 字符内容，断言原文保持不变
def test_exact_limit_untouched() -> None:
    text = "z" * 8000
    msgs = [_make_tool_result_msg(text)]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    assert result[0]["content"][0]["content"] == text


# 功能：验证 text 类型 block 不受截断影响
# 设计：构造含 text block 的 user 消息，内容超过阈值，断言内容原样返回
def test_non_tool_result_block_untouched() -> None:
    long_text = "a" * 20_000
    msgs = [{"role": "user", "content": [{"type": "text", "text": long_text}]}]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    assert result[0]["content"][0]["text"] == long_text


# 功能：验证同一 user 消息含多个 tool_result 时各自独立判断截断
# 设计：构造一条消息含两个 tool_result，一短一长，断言只有长的被截断
def test_multiple_tool_results_independent() -> None:
    short = "s" * 100
    long = "l" * 10_000
    msgs = [{
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": short},
            {"type": "tool_result", "tool_use_id": "b", "content": long},
        ],
    }]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    blocks = result[0]["content"]
    assert blocks[0]["content"] == short
    assert "chars omitted" in blocks[1]["content"]


# 功能：验证 assistant 消息不被截断处理
# 设计：构造超长内容的 assistant 消息，断言原样返回
def test_assistant_message_untouched() -> None:
    text = "a" * 20_000
    msgs = [{"role": "assistant", "content": text}]
    result = truncate_tool_results(msgs, limit=8000, keep=4000)
    assert result[0]["content"] == text


# 功能：已截断的 tool_result 再次截断时不重复处理（幂等）
# 设计：先截断一次，对结果再截断，断言标记只出现一次且内容不变
def test_already_truncated_result_not_re_truncated() -> None:
    text = "y" * 10_000
    once = truncate_tool_results([_make_tool_result_msg(text)], limit=8000, keep=4000)
    twice = truncate_tool_results(once, limit=8000, keep=4000)
    assert once[0]["content"][0]["content"] == twice[0]["content"][0]["content"]
    assert twice[0]["content"][0]["content"].count("Full output is available") == 1


# 功能：截断不修改原始消息对象（返回新列表与新 block 副本）
# 设计：输入超长结果，断言原 msg dict 与 block 内容保持不变
def test_truncation_does_not_mutate_original() -> None:
    text = "z" * 10_000
    msg = _make_tool_result_msg(text)
    original_block = msg["content"][0]
    result = truncate_tool_results([msg], limit=8000, keep=4000)
    assert result[0] is not msg
    assert result[0]["content"][0] is not original_block
    assert original_block["content"] == text
    assert msg["content"][0]["content"] == text
