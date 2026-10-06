# KamaClaude

## Personal Assistant Knowledge (RAG V1)

The optional local knowledge subsystem uses Qdrant Local for vector search, SQLite for the document catalog, and the local `bge-small-zh-v1.5` model for embeddings and token counting. It is disabled by default and does not require a Qdrant server.

Enable it in `~/.kama/config.toml`:

```toml
[knowledge]
enabled = true
auto_retrieve = true
top_k = 8
max_retrieved_tokens = 4000

[knowledge.embedding]
provider = "local"
model = "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"

[knowledge.store]
provider = "qdrant"
path = "~/.kama/knowledge/qdrant"
collection = "kama_knowledge"
```

V1 supports TXT, Markdown, and common code files. Root Agent runs perform one automatic retrieval when `auto_retrieve` is enabled. Root Agents and Subagents can also use `knowledge_search`, which inherits the current user and project scope. Initialization or retrieval failures are isolated from normal Agent execution.

本地 AI Agent 系统。`kama-core` 作为常驻守护进程处理所有任务，`kama`（CLI）和 `kama-tui`（TUI）通过 TCP loopback 与之通信。

## 环境要求

| 依赖 | 版本 |
|------|------|
| 操作系统 | macOS / Linux |
| Python | 3.12.x |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.4 |

安装 uv（若尚未安装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Python 3.12 由 uv 自动管理，无需手动安装。

## 快速开始

```bash
git clone <repo> && cd KamaClaude
uv sync
cp .env.example .env        # 按需修改

uv run kama-core &          # 启动守护进程（后台）
uv run kama ping            # 验证连通：应返回 pong
uv run kama --version       # 应输出 0.0.1
```

## 文档

- **[RUNBOOK.md](./RUNBOOK.md)** — 完整操作参考：配置、开发命令、故障排查
- **[WIRE_PROTOCOL.md](./WIRE_PROTOCOL.md)** — IPC 协议定义（由代码生成，勿手动编辑）
