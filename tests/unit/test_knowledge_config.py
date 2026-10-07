from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig, _apply_toml, get_config


# 功能：验证 Knowledge 配置默认值
# 设计：默认关闭且使用 V1 预算和后端参数
def test_knowledge_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAMA_CONFIG", raising=False)
    config = get_config()
    assert config.knowledge.enabled is False
    assert config.knowledge.top_k == 8
    assert config.knowledge.max_retrieved_tokens == 4000
    assert config.knowledge.hybrid_enabled is False
    assert config.knowledge.candidate_multiplier == 2
    assert config.knowledge.rrf_k == 60
    assert config.knowledge.document_aggregation is None


# 功能：验证 Hybrid 与 lexical 配置能从 TOML 严格解析
# 设计：直接应用内存配置表，覆盖启用开关、索引路径和融合参数而不依赖文件系统
def test_hybrid_config_toml_values() -> None:
    config = KamaConfig()

    _apply_toml(
        config,
        {
            "knowledge": {
                "hybrid_enabled": True,
                "lexical_path": "knowledge/lexical.db",
                "candidate_multiplier": 3,
                "rrf_k": 45,
                "document_aggregation": "sum",
            }
        },
    )

    assert config.knowledge.hybrid_enabled is True
    assert config.knowledge.lexical_path == "knowledge/lexical.db"
    assert config.knowledge.candidate_multiplier == 3
    assert config.knowledge.rrf_k == 45
    assert config.knowledge.document_aggregation == "sum"


# 功能：验证无效 Hybrid 参数会在配置加载时被拒绝
# 设计：直接提供小于最小值的 candidate multiplier，确认 fail-fast 错误包含字段名
def test_hybrid_config_rejects_invalid_candidate_multiplier() -> None:
    with pytest.raises(SystemExit, match="knowledge.candidate_multiplier"):
        _apply_toml(KamaConfig(), {"knowledge": {"candidate_multiplier": 0}})


# 功能：验证未知文档聚合函数不会静默改变检索排序
def test_hybrid_config_rejects_invalid_document_aggregation() -> None:
    with pytest.raises(SystemExit, match="document_aggregation"):
        _apply_toml(KamaConfig(), {"knowledge": {"document_aggregation": "max"}})


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
