# Kama Knowledge Assistant

> 基于开源 [KamaClaude](https://github.com/youngyangyang04/KamaClaude)（MIT）逐行学习并二次开发的**本地 AI Agent + 个人知识助手**。
> `kama-core` 常驻守护进程处理 Agent / Session / Tool / Permission / MCP / 事件流；`kama` CLI 与 `kama-tui`（Textual）通过 TCP loopback 通信。
> 在原作者 Agent 框架之上，从零实现了一套完整的 RAG 知识库子系统（Qdrant + Hybrid Search + Document Aggregation），带可复现的评测基线。

## 项目来源与贡献边界

本项目 fork 自开源项目 [KamaClaude](https://github.com/youngyangyang04/KamaClaude)（MIT License，© 2026 程序员Carl）。作者以学习为目的逐行阅读并改造原项目，原项目提供了 Agent 框架基础（daemon / CLI / TUI / Session / Tool / Permission / MCP / 事件流）。本仓库新增内容（原项目不包含）：

- **RAG 知识库子系统**（核心增量）：Qdrant Local 向量检索 + 本地 `bge-small-zh-v1.5` Embedding + SQLite FTS5 trigram BM25 词法检索 + RRF 混合融合 + sum 文档级聚合；
- **量化评测体系**：20 问 golden set、single/multi 双口径、Hit / Recall / MRR / NDCG、固定候选池受控实验；
- **Agent 集成**：Root Agent 自动检索注入、`knowledge_search` 工具（scope 隔离）、token 预算控制、RAG 故障降级；
- **工程修复与验证**：Windows 信号处理兼容、真实 Agent 端到端验证（Q9 / Q18 实跑凭证）。

原项目 LICENSE 完整保留（见 [LICENSE](./LICENSE)），新增代码同样遵循 MIT。

## Highlights

- **本地优先**：向量检索（Qdrant Local）、Embedding（`bge-small-zh-v1.5`）、文档索引全部本地运行，不上传数据，无需外部服务。
- **完整 Agent 框架**：Session 状态机、Tool/Permission 体系、MCP（stdio/TCP）、Subagent（spawn/result）、事件总线（events.jsonl + IPC 实时流 + trace）。
- **RAG 四阶段演进，全部有数值评测**：V1 基础检索 → V2 Chunker 改良 → V3 Hybrid Search（中文 FTS5 trigram BM25 + RRF）→ V4 Document Aggregation（sum，opt-in）。
- **严谨评测方法论**：20 问 golden set、single/multi 双口径、Hit/Recall/MRR/NDCG、固定候选池受控实验、token 预算复核。

## 架构总览

```text
kama-core (daemon)
├── SocketServer (JSON-RPC 2.0 + NDJSON over TCP loopback)
├── SessionManager
├── AgentRunner → AgentLoop (Plan → Act → Observe)
│   ├── ToolRegistry (builtin / MCP / subagent)
│   ├── PermissionManager (审批 + always-allow 持久化)
│   └── KnowledgeService (可选，RAG)
│       └── HybridRetriever
│           ├── DenseRetriever → Qdrant Local
│           ├── BM25Retriever  → SQLite FTS5 (trigram)
│           └── RRF Fusion → Top-K → Token Budget (4000)
├── EventBus → EventWriter / IpcBroadcaster / TraceWriter
└── MCP Manager (stdio / TCP)
```

## RAG 技术演进与评测结果

知识库：187 篇中文 AI/Agent 技术文档，2989 chunks，Qdrant（512 维 cosine）+ FTS5 双索引同步。

| 阶段 | 关键改进 | single Recall@5 | MRR@5 | NDCG@5 |
|---|---|---|---:|---:|---:|
| V2 Dense（诚实基线）| 挤掉 V1 空壳/截断虚高 | 0.800 | 0.713 | 0.735 |
| Hybrid (k=8) | FTS5 trigram BM25 + RRF，救回精确术语题 | 0.900 | 0.765 | 0.797 |
| **Aggregation sum (k=8)** | 文档级信号，消除大部头 chunk 霸榜 | **0.950** | **0.863** | **0.885** |

- **双口径评测**：single（主）+ multi（辅助，`related_docs` 多标签，6 题），严格区分 Query Hit Rate 与 Recall。
- **受控实验**：固定候选池（2×k）、完整 RRF、三种聚合函数（sum/max/first-seen）A/B，控制组与 baseline 完全一致，证明收益来自文档级排序而非去重本身。
- **P1 预算复核**（4000 token 真实 bge tokenizer 计数）：sum 聚合的 Budget-Hit 0.90→0.95（Q9 目标文档首次进入上下文）、平均进 context 文档数 2.9→5.9、平均 token 消耗 3106→2638。
- 完整报告：`knowledge_docs/` 下 `RAG_V2_HYBRID_实现总结与改进方向.md`、`eval_baseline.md`、`eval_aggregation.md`（含逐问验证与对比图）。

## 快速开始

Windows（PowerShell）或 macOS/Linux（bash）：

```powershell
git clone https://github.com/chebobo/kama-knowledge-assistant.git && cd kama-knowledge-assistant
uv sync
cp .env.example .env        # 按需修改（LLM API key 等）

uv run kama-core &          # 启动守护进程（后台）
uv run kama ping            # 验证连通：应返回 pong
uv run kama-tui             # 或启动 TUI
```

## 启用知识库（RAG）

`~/.kama/config.toml`：

```toml
[knowledge]
enabled = true
auto_retrieve = true        # Root Agent run 开始时自动检索一次
top_k = 8
max_retrieved_tokens = 4000

[knowledge.embedding]
provider = "local"
# 必须指向 snapshot 子目录（hub 缓存父目录无法直接加载）：
model = "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"

[knowledge.store]
provider = "qdrant"
path = "~/.kama/knowledge/qdrant"
collection = "kama_knowledge"

[knowledge.hybrid]
hybrid_enabled = true
lexical_path = "~/.kama/knowledge/lexical.db"
candidate_multiplier = 2
rrf_k = 60
document_aggregation = "sum"   # 可选：None（默认）| "sum"
```

- Root Agent 自动检索；Root/Subagent 均可手动调用 `knowledge_search`，scope（global + current project）由系统注入，LLM 无法越权指定 project。
- RAG 初始化/检索失败自动降级，不影响普通 Agent Run。

## 环境要求

| 依赖 | 版本 |
|------|------|
| 操作系统 | Windows 10+ / macOS / Linux |
| Python | 3.12.x |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.4 |

## 文档

- **[RUNBOOK.md](./RUNBOOK.md)** — 完整操作参考：配置、开发命令、故障排查
- **[WIRE_PROTOCOL.md](./WIRE_PROTOCOL.md)** — IPC 协议定义（由代码生成，勿手动编辑）
- `knowledge_docs/` — RAG 评测脚本、golden set、评测报告与交接文档
