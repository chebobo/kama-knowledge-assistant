from pathlib import Path

import pytest

from kama_claude.core.config import get_config


# 功能：验证 Knowledge 配置默认值
# 设计：默认关闭且使用 V1 预算和后端参数
def test_knowledge_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAMA_CONFIG", raising=False)
    config = get_config()
    assert config.knowledge.enabled is False
    assert config.knowledge.top_k == 8
    assert config.knowledge.max_retrieved_tokens == 4000


# 功能：验证 TOML 和环境变量配置覆盖顺序
# 设计：环境变量优先于 TOML，嵌套配置完整解析
def test_knowledge_toml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[knowledge]\nenabled = true\ntop_k = 5\n[knowledge.embedding]\nmodel = 'model-a'\n[knowledge.store]\npath = 'store-a'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAMA_CONFIG", str(path))
    monkeypatch.setenv("KAMA_KNOWLEDGE_TOP_K", "9")
    config = get_config()
    assert config.knowledge.enabled is True
    assert config.knowledge.top_k == 9
    assert config.knowledge.embedding.model == "model-a"
    assert config.knowledge.store.path == "store-a"


# 功能：验证非法 Knowledge 配置被拒绝
# 设计：保持现有配置的 fail-fast 语义
def test_knowledge_invalid_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[knowledge]\ntop_k = 0\n", encoding="utf-8")
    monkeypatch.setenv("KAMA_CONFIG", str(path))
    with pytest.raises(SystemExit, match="knowledge.top_k"):
        get_config()
