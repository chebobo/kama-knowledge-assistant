from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import get_config


def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")# 功能：验证 .env 文件中的值被正确加载并覆盖内建默认值
# 设计：写 .env 到临时目录并 chdir 进去，清除同名系统环境变量排除干扰，确认 .env 加载路径有效
def test_dotenv_base_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 9999


# 功能：验证系统环境变量的优先级高于 .env 文件中的值
# 设计：.env 写 9999，系统环境变量写 8888，确认最终值为 8888，对应四级优先链的顶层约束
def test_system_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_PORT", "8888")

    cfg = get_config()

    assert cfg.port == 8888


# 功能：验证 .env 文件不存在时静默跳过，使用内建默认值（不抛异常）
# 设计：chdir 到空目录，清除系统环境变量，确认 get_config() 不因 .env 缺失而崩溃，默认端口为 7437
def test_missing_env_file_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 7437


# 功能：验证 .env 中设置的 KAMA_CONFIG 能正确影响 TOML 配置文件的加载路径
# 设计：.env 指向自定义 TOML 文件，TOML 中写入不同端口，确认 .env 在 TOML 加载前被读取（优先级链的正确顺序）
def test_dotenv_before_toml_kama_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "custom.toml"
    toml_path.write_bytes(b'[core]\nport = 5555\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, f"KAMA_CONFIG={toml_path}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_CONFIG", raising=False)
    monkeypatch.delenv("KAMA_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 5555


# 功能：验证同一变量经过完整四级优先链后，最终值为最高优先级来源（系统环境变量）
# 设计：同时设置默认值(7437)/TOML(6000)/.env(7000)/系统环境变量(8000)，确认最终值为 8000，是优先级链的综合正确性验证
def test_priority_chain_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 默认值：7437
    # TOML：6000
    # .env：7000
    # 系统环境变量：8000（最高）
    toml_path = tmp_path / "kama.toml"
    toml_path.write_bytes(b'[core]\nport = 6000\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, "KAMA_PORT=7000\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_CONFIG", str(toml_path))
    monkeypatch.setenv("KAMA_PORT", "8000")

    cfg = get_config()

    assert cfg.port == 8000


# --- MCP 配置项：exposure_mode、预算、数组、未知键、环境变量 ---------------------


def _load_toml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml: str) -> object:
    toml_path = tmp_path / "kama_mcp.toml"
    toml_path.write_bytes(toml.encode("utf-8"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_CONFIG", str(toml_path))
    for key in (
        "KAMA_MCP_EXPOSURE_MODE",
        "KAMA_MCP_MAX_ACTIVE_TOOLS",
        "KAMA_MCP_MAX_SCHEMA_CHARS",
        "KAMA_MCP_SEARCH_MAX_RESULTS",
        "KAMA_MCP_ALWAYS_EXPOSE",
        "KAMA_MCP_BLOCKED_TOOLS",
    ):
        monkeypatch.delenv(key, raising=False)
    return get_config()


# 功能：显式 lazy / eager 被正确解析
# 设计：分别写两种 exposure_mode，断言解析值
def test_mcp_exposure_mode_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("lazy", "eager"):
        cfg = _load_toml_config(tmp_path, monkeypatch, f'[mcp]\nexposure_mode = "{mode}"\n')
        assert cfg.mcp.exposure_mode == mode


# 功能：未写 exposure_mode 时保持 None（由 runner 按 servers 推断）
# 设计：只写 servers 空表，断言 exposure_mode 为 None
def test_mcp_exposure_mode_unset_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_toml_config(tmp_path, monkeypatch, "[mcp]\nservers = []\n")
    assert cfg.mcp.exposure_mode is None


# 功能：非法 exposure_mode 报配置错误
# 设计：写入 "foo"，断言 SystemExit
def test_mcp_exposure_mode_invalid_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _load_toml_config(tmp_path, monkeypatch, '[mcp]\nexposure_mode = "foo"\n')


# 功能：预算字段拒绝布尔、负数、零与非整数
# 设计：对每个预算键用非法值触发 SystemExit
@pytest.mark.parametrize("key", ["max_active_tools", "max_schema_chars", "search_max_results"])
def test_mcp_budget_invalid_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    for bad in ("true", "-1", "0", '"x"'):
        with pytest.raises(SystemExit):
            _load_toml_config(tmp_path, monkeypatch, f"[mcp]\n{key} = {bad}\n")


# 功能：预算字段合法值被解析
# 设计：写 3/9000/4，断言精确值
def test_mcp_budget_valid_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_toml_config(
        tmp_path,
        monkeypatch,
        "[mcp]\nmax_active_tools = 3\nmax_schema_chars = 9000\nsearch_max_results = 4\n",
    )
    assert cfg.mcp.max_active_tools == 3
    assert cfg.mcp.max_schema_chars == 9000
    assert cfg.mcp.search_max_results == 4


# 功能：always_expose / blocked_tools 数组校验（非数组、空串、非字符串均拒绝）
# 设计：对两个键分别用非法值触发 SystemExit
@pytest.mark.parametrize("key", ["always_expose", "blocked_tools"])
def test_mcp_string_array_invalid_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    for bad in ('"not-an-array"', '["ok", ""]', "[1, 2]"):
        with pytest.raises(SystemExit):
            _load_toml_config(tmp_path, monkeypatch, f"[mcp]\n{key} = {bad}\n")


# 功能：always_expose / blocked_tools 合法数组被解析
# 设计：写含 server 前缀的完整工具名，断言精确列表
def test_mcp_string_array_valid_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_toml_config(
        tmp_path,
        monkeypatch,
        '[mcp]\nalways_expose = ["filesystem__read_file"]\nblocked_tools = []\n',
    )
    assert cfg.mcp.always_expose == ["filesystem__read_file"]
    assert cfg.mcp.blocked_tools == []


# 功能：未知 [mcp] 键仍报错
# 设计：写入不存在的键，断言 SystemExit
def test_mcp_unknown_key_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _load_toml_config(tmp_path, monkeypatch, '[mcp]\nmystery_key = 1\n')


# 功能：六组 KAMA_MCP_* 环境变量生效且覆盖 TOML 值
# 设计：TOML 写一组值，系统环境变量写另一组，断言环境变量优先
def test_mcp_env_vars_override_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_toml_config(
        tmp_path,
        monkeypatch,
        '[mcp]\nexposure_mode = "lazy"\nmax_active_tools = 8\nsearch_max_results = 5\n',
    )
    monkeypatch.setenv("KAMA_MCP_EXPOSURE_MODE", "eager")
    monkeypatch.setenv("KAMA_MCP_MAX_ACTIVE_TOOLS", "12")
    monkeypatch.setenv("KAMA_MCP_SEARCH_MAX_RESULTS", "7")
    monkeypatch.setenv("KAMA_MCP_ALWAYS_EXPOSE", "fs__read,fs__write")
    monkeypatch.setenv("KAMA_MCP_BLOCKED_TOOLS", "fs__rm")
    cfg = get_config()
    assert cfg.mcp.exposure_mode == "eager"
    assert cfg.mcp.max_active_tools == 12
    assert cfg.mcp.search_max_results == 7
    assert cfg.mcp.always_expose == ["fs__read", "fs__write"]
    assert cfg.mcp.blocked_tools == ["fs__rm"]


# 功能：环境变量非法值报错（负数、非 lazy/eager）
# 设计：分别设置非法环境变量，断言 SystemExit
@pytest.mark.parametrize(
    "env_key, value",
    [
        ("KAMA_MCP_EXPOSURE_MODE", "foo"),
        ("KAMA_MCP_MAX_ACTIVE_TOOLS", "-3"),
        ("KAMA_MCP_SEARCH_MAX_RESULTS", "0"),
    ],
)
def test_mcp_env_var_invalid_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_key: str, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAMA_CONFIG", raising=False)
    monkeypatch.setenv(env_key, value)
    with pytest.raises(SystemExit):
        get_config()
