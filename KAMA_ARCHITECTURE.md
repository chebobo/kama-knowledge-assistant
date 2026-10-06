# KamaClaude 项目架构与技术选型

> 本文件记录当前仓库实际实现的系统框架、核心数据流、主要技术和采用这些技术的原因。
>
> 描述以源码为准。若 README、旧设计文档和源码不一致，以当前源码和测试行为为准。

## 1. 系统定位

KamaClaude 是一个本地 AI Agent 系统。核心进程 kama-core 作为常驻 daemon 处理 Agent、Session、Tool、权限、MCP 和事件；kama CLI 与 kama-tui TUI 作为客户端通过本机 TCP loopback 连接 daemon。

整体结构：

```text
kama-core
  ├── Config / Logging
  ├── SocketServer
  ├── SessionManager
  ├── AgentRunner
  ├── PermissionManager
  ├── MCP Server Manager
  ├── EventBus / Trace
  └── AgentLoop

kama CLI  ───────┐
kama TUI  ───────┴── JSON-RPC 2.0 + NDJSON over TCP loopback
```

当前默认地址：127.0.0.1:7437。

注意：部分旧文档把传输描述为 Unix domain socket，但当前实现使用 asyncio.start_server() 和 asyncio.open_connection() 提供 TCP loopback。

## 2. 运行入口

入口由 pyproject.toml 的 [project.scripts] 定义：

```text
kama       -> kama_claude.cli.main:main
kama-core  -> kama_claude.core.app:run
kama-tui   -> kama_claude.tui.__main__:main
```

### kama-core

主要入口是 src/kama_claude/core/app.py。CoreApp.run() 负责加载配置、初始化日志和可选 TraceWriter、初始化 PermissionManager/EventBus/IpcEventBroadcaster、启动 MCP Server、创建 SessionManager 和 AgentRunner 工厂、注册 RPC handlers，并等待信号优雅退出。

采用常驻 daemon 的原因：

- 复用模型外的系统依赖和 MCP 连接。
- 让 CLI/TUI 共享 Session、Permission 和事件流。
- 将长任务生命周期与短命令行进程分开。
- 让权限审批等双向交互保持可用。

### kama

CLI 位于 src/kama_claude/cli/，适合 ping、run、chat、trace 和 core 生命周期管理。CLI 不直接执行 Agent，只通过 SocketClient 调用 daemon，从而与 TUI 共享同一套服务行为。

### kama-tui

TUI 位于 src/kama_claude/tui/app.py，使用 Textual 构建，是主要用户界面。它负责创建 chat session、订阅事件、流式显示 LLM 输出、展示工具调用、权限控件、skill 补全和上下文压缩操作。

Textual 适合当前终端工作台，因为它提供异步 UI、布局、焦点、键盘事件和可组合 widget。

## 3. 分层架构

### 3.1 Transport 层

文件：core/transport/socket_server.py、socket_client.py、ipc_broadcaster.py。

SocketServer 使用 asyncio TCP server 读取 NDJSON，每行解析为 JSON-RPC 请求。每条请求单独创建 task，保证长时间运行的 Agent 请求不会阻塞 permission.respond 等后续请求。

SocketClient 维护 request id 到 Future 的映射，并把服务端消息分为 RPC response/error 和 server push event。

选择 JSON-RPC 2.0 + NDJSON 的原因：请求响应有明确 id，适合并发；错误结构统一；NDJSON 易于流式传输、调试和回放；事件推送可以复用同一连接。

### 3.2 Protocol 层

文件：core/bus/envelope.py、commands.py、events.py。

Pydantic v2 模型定义命令、事件和 JSON-RPC envelope。命令和事件通过 type 字段构成 discriminated union。

Pydantic 用于 IPC 边界校验、JSON 序列化和显式协议建模。WIRE_PROTOCOL.md 由 scripts/gen_protocol_doc.py 生成，修改协议模型后必须重新生成并检查同步状态。

### 3.3 Core Application 层

CoreApp 是系统组装根，持有全局共享的 EventBus、IpcEventBroadcaster、TraceWriter、PermissionManager、MCP Manager 和 SessionManager。具体 Agent run 的局部状态由 AgentRunner 创建，因此系统资源和 run 级状态分离。

### 3.4 Session 层

文件：core/session/model.py、store.py、manager.py。

Session 支持 one_shot 和 chat，默认保存于 ~/.kama/sessions/<session_id>/：

```text
meta.json
thread.jsonl
notes.md
runs/<run_id>/events.jsonl
```

SessionManager 负责 session 状态机、同一 session 的并发锁、消息追加、run 调度、skill 解析和手动压缩。JSONL 适合追加、诊断、恢复以及保存 Anthropic tool_use/tool_result 结构。

### 3.5 AgentRunner

文件：core/runner.py。

AgentRunner 是一次 run 的装配层，负责读取历史和记忆、创建 ExecutionContext、TaskManager、ToolRegistry、Compactor、AgentLoop，并写入 events.jsonl。

Runner 作为装配层，可以让 AgentLoop 不知道 Session、MCP、权限和持久化细节，也方便测试注入 provider、EventBus、运行目录和事件处理器。

### 3.6 ExecutionContext

文件：core/context.py。

ExecutionContext 保存一次 run 的 goal、消息、global/project/session context、步骤、状态、失败原因和最终结果，并负责追加 assistant/tool messages 和组合 system prompt。

它是一次 run 的状态对象，不是独立 Context 服务。因此新增 RAG 时应在此文件增加 retrieved_context 等字段，不应新建 core/context/ 包。

### 3.7 AgentLoop

文件：core/loop.py。

AgentLoop 实现 Plan -> Act -> Observe：调用 LLM、保存 assistant blocks、执行工具、保存 tool results、判断是否继续、触发压缩并发布 step 事件。

它不负责 Session 持久化、RAG 检索、Parser、Embedding、Vector DB 或权限策略定义。保持这个边界可以让核心循环简单且容易用 fake provider 测试。

### 3.8 LLM 层

文件：core/llm/base.py、provider.py、types.py。

LLMProvider 是 Protocol，当前生产实现是 Anthropic provider。统一返回 LlmResponse，包括 stop reason、文本、tool calls、thinking blocks 和 usage。Anthropic provider 使用流式响应发布 token，并处理重试和 usage 统计。

Provider 抽象让测试不依赖真实 API，也让 AgentLoop 不依赖具体 SDK，后续可以替换模型服务。

### 3.9 Tool 层

文件：core/tools/base.py、registry.py、invocation.py 和 builtin/。

所有工具实现 BaseTool，提供 name、description、input_schema、可选 params_model 和异步 invoke。Registry 生成 Anthropic tool schema；invoke_tool() 统一处理参数校验、权限、超时、重试、错误分类和工具事件。

这种设计让内置工具、MCP 工具和 Subagent 工具走同一执行链，AgentLoop 不需要为不同工具写特殊分支。

### 3.10 Permission 层

文件：core/permissions/policy.py、manager.py、storage.py。

权限系统包含静态策略、session 缓存、持久化策略、客户端审批 Future、超时和断连处理。需要审批的工具发布 permission.requested，客户端通过 permission.respond 恢复 Future。

该设计可以让 daemon 等待用户决定，同时让 CLI 和 TUI 共享审批协议，并持久化 always allow/deny。

### 3.11 EventBus、EventWriter 和 Trace

文件：core/events/bus.py、events/writer.py、trace/writer.py、trace/provider.py。

事件可同时流向：

```text
EventWriter      -> runs/<run_id>/events.jsonl
IpcBroadcaster   -> 客户端实时事件
TraceWriter      -> daemon trace
```

EventBus 使用异步 subscriber 顺序发布。EventWriter 记录单次 run 事件，TraceWriter 通过队列和后台 drain task 降低文件写入对主流程的影响。

### 3.12 MCP 层

文件：core/mcp/client.py、server.py、tool.py。

支持 stdio 和 TCP MCP Server。daemon 启动时完成 initialize、工具发现，并把 MCP 工具包装为 BaseTool 注册到 ToolRegistry。

MCP 表示外部系统连接和能力；Knowledge/RAG 表示本地文档解析、索引和检索。两者保持独立，未来可将 MCP 数据源接入 Knowledge ingestion。

### 3.13 Memory 层

文件：core/memory/loader.py 和 core/tools/builtin/note_save.py。

Memory 通过 ~/.kama/context.md、项目 .kama/context.md 和 session notes.md 保存用户明确要长期记住的事实和偏好。它不等于 Knowledge/RAG，不应自动 embedding 全部聊天记录或文档。

### 3.14 Skill 层

文件：core/skills/loader.py 和 core/skills/builtin/。

Skill 通过 /name arguments 触发，可覆盖 system prompt 并限制工具白名单。项目 skill 优先于用户 skill，用户 skill 优先于内置 skill。Markdown frontmatter 使用户可以直接创建和修改 skill。

## 4. 主要数据流

### 4.1 普通 Agent Run

```text
客户端
  ↓ session.send_message / agent.run
SocketServer
  ↓
CoreApp handler
  ↓
SessionManager
  ↓
AgentRunner
  ↓
ExecutionContext
  ↓
AgentLoop
  ↓
LLMProvider
  ↓
ToolRegistry / invoke_tool
  ↓
ExecutionContext
  ↓
AgentLoop ...
```

### 4.2 事件流

```text
AgentLoop / Tool / Session
          ↓
       EventBus
       ├── EventWriter
       ├── IpcEventBroadcaster
       └── TraceWriter
```

### 4.3 Session 持久化

```text
SessionManager
  ├── meta.json
  ├── thread.jsonl
  ├── notes.md
  └── runs/<run_id>/events.jsonl
```

## 5. 配置和依赖

配置位于 core/config.py，优先级为：

```text
内置默认值 -> ~/.kama/config.toml -> .kama/config.toml -> .env -> 系统环境变量
```

项目用 dataclass 表达运行时配置，用 tomllib 解析 TOML，用 python-dotenv 加载 .env。手工严格解析可以及时发现未知配置键和类型错误，且不需要引入复杂配置框架。

项目要求 Python >=3.12,<3.13，使用 uv 管理环境和 lockfile。

主要运行时依赖及原因：

| 技术 | 用途 | 采用原因 |
|---|---|---|
| Python 3.12 | 主语言 | 支持当前 asyncio、typing 和 dataclass 能力 |
| asyncio | daemon、Socket、Agent 并发 | 适合长任务、流式输出和权限等待 |
| Pydantic v2 | 协议和参数模型 | 边界校验、序列化、判别联合 |
| Anthropic SDK | LLM 流式调用 | 支持当前模型、tool use、thinking 和 usage |
| Textual | TUI | 异步终端 UI、布局、键盘和 widget |
| httpx | HTTP/MCP 相关网络 | 异步 HTTP 和 SOCKS 支持 |
| python-dotenv | .env | 保持本地配置和密钥管理简单 |
| uv | 依赖和环境 | 快速同步、锁定版本、可复现开发环境 |
| pytest/pytest-asyncio | 测试 | 覆盖同步和异步生命周期 |
| Ruff | lint | 快速统一风格和 import 检查 |
| mypy strict | 类型检查 | 约束协议、事件、工具边界的一致性 |

## 6. 测试策略

```text
tests/unit/
tests/integration/
```

Unit 测试覆盖 Context、Loop、Runner、Session、Socket、Tool、Permission、MCP、Trace、Config 和 Compaction。Integration 测试启动真实 daemon 子进程，验证双进程通信、Session IPC、权限流程和端到端 run。

新增代码还必须遵守 AGENT.md 的注释约定：生产函数前有一行中文功能注释，测试函数前有“功能”和“设计”两行中文注释。

## 7. 持久化和可观测性

主要持久化位置：

```text
~/.kama/sessions/    Session 数据
~/.kama/policy.toml  权限策略
~/.kama/traces/      daemon trace
runs/                非 session run 数据
```

日志用于诊断，events 用于单次 run 回放，trace 用于记录 IPC、事件和 LLM 层操作。三层记录分别服务于实时界面、回放和问题排查。

## 8. Personal Assistant RAG 的接入位置

RAG 应作为新增 Knowledge 能力接入，而不是新的 Agent 或新的 Context 系统：

```text
CoreApp
  ↓ 创建/注入可选 KnowledgeService
AgentRunner
  ├── Root Run 开始时调用 retrieve_context()
  └── 构造 knowledge_search Tool
ExecutionContext
  └── 保存 retrieved_context、user_id、project_id
AgentLoop
  └── 保持原有执行循环不变
```

Knowledge 内部边界：

```text
KnowledgeService
  ├── Parser
  ├── Chunker
  ├── EmbeddingProvider
  ├── VectorStore
  ├── Retriever
  └── SQLite Document Catalog
```

这样可以让 AgentLoop 继续只负责 LLM/tool execution，让 AgentRunner 只负责依赖组装，并支持未来从 Qdrant Local 切换到 Qdrant Server。

## 9. 重要边界

不要：

```text
新建 core/context/ 包
把 RAG 查询硬编码到 AgentLoop
让 AgentRunner 直接调用 Qdrant
让 LLM 自由指定 project_id
把 Knowledge 和 Memory 合并
把 MCP 和 Knowledge 写成一个模块
让 RAG 初始化失败阻止 daemon 启动
在 V1 引入 RPC、Hybrid、Reranker 或 Query Rewrite
```

应当：

```text
通过 ExecutionContext 传递 scope
通过 KnowledgeService 暴露 RAG 能力
通过 EmbeddingProvider 抽象 embedding
通过 VectorStore 抽象 Qdrant
在 worker thread 执行同步 Qdrant/embedding
让 RAG 故障降级为空上下文或结构化 Tool 错误
```
