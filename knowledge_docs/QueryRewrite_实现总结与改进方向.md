# KamaClaude 个人知识助手 RAG — Query Rewrite 实现总结与改进方向

> 状态：**v2.2 定稿（11/11 验收通过）→ 已接入生产（默认关闭）→ chunk token 超长截断修复（>512 归零）→ 新索引重测通过 → C 组双路融合验证完成（multi 全面反超基线）**
> 更新：2026-10-10（晚间：token 感知切分重建 + A/B/C 三组重测）
> 数据来源：`eval_rag_rewrite.py` / `rewrite_results_v2_2/`（字符切分索引 2989 chunks 时代）与 `rewrite_results_v3_token*/`（token 切分索引 5265 chunks 时代），summary.json + per_question.jsonl + 索引快照均可复现
> 承接：`RAG_V2_HYBRID_实现总结与改进方向.md`（Q18 框架对比盲区 = 本模块的靶子）

> **重要索引说明**：本文第三节 3.1–3.5 的数据来自**字符切分索引（chunk_size=1600/overlap=200，2989 chunks）**，是 v2.2 Prompt 定稿时的证据；2026-10-10 晚间完成 **token 感知切分重建（max_tokens=480/overlap=48，5265 chunks）** 后，全部指标已在**新索引**上重测（3.6–3.8），并新增 **C 组双路融合**（3.7）。当前生产索引为 token 切分版本，简历/对外数据以新索引为准。

---

## 一、项目背景与总体目标

V2 Hybrid 阶段把检索链路推到 `Agg-sum k=8 = 0.950/0.863/0.885`，但留下最后一个硬缺陷：

- **Q18「主流 Agent 框架有什么异同」三层 MISS**：Dense 语义近邻、FTS5 trigram、文档聚合全部召不回 `agent_framework_comparison.md`（标题「OpenClaw、Hermes Agent、Claude Code三框架横评」）；
- 诊断矩阵（`diag_q18_variants.py`）实证：**不是知识库没内容，而是"查询表达"问题**——原问泛词「主流/框架/异同」与文档专名（OpenClaw/Hermes Agent/Claude Code）在 Dense 与 BM25 候选池零重叠；文档专名词与纯主题维度词（框架对比/架构差异/设计特点/记忆机制/工具调用/上下文管理）均双路 rank1。

**本模块目标**：用受约束的查询改写解决 Q18 及同类「概念模糊 / 术语型」查询，同时保证其余 19 题**零回归**（比总分不降更严格：逐题保住命中）。

**评测口径**（与 Hybrid 阶段一致）：
- 20 问 golden 集，Q2/Q3/Q9/Q16/Q18/Q19 标注多个相关文档；
- single（主口径）= 命中 `expected_doc`；multi（辅助口径）= 命中任一 `related_docs`；
- document 级匹配，指标在 **top-5** 计算（未舍入值比较）；
- A/B 全部跑在同一索引快照（Qdrant 复制 + SQLite backup，前后指纹校验），检索参数与生产一致（每路 2×k=16 候选、RRF=60、sum 聚合、top_k=8）。

---

## 二、更改思路：四版 Prompt 的演进脉络

### 阶段 1：v1 补实体（失败，未跑全量）

**做了什么**：改写时向问题中补充 LangChain、LlamaIndex、AutoGPT 等"主流框架"常识实体。

**暴露的问题**（两个独立缺陷）：
1. **空输出**：deepseek-v4-pro 兼容层默认输出 thinking 块（实测 300~970 字符），`max_tokens=256/512` 时 thinking 吃光预算 → `stop_reason=max_tokens`、文本为空；
2. **常识 ≠ 知识库事实**：补的实体在知识库里根本没有对应文档，改写把检索引向歧途。

**教训**：改写不能依赖模型的常识；预算必须按模型真实输出习惯划定。

### 阶段 2：空输出修复 + v2 禁实体删泛词

**修复空输出**：`MAX_TOKENS 256→1024`；`RewriteResult` 增加 `stop_reason / thinking_chars / text_chars` 诊断字段；`provider.complete_text` 增加 thinking 块解析——每次失败都能回答"是想太久吃光预算，还是真没输出"。

**v2 改写规则**：禁止补充实体；删除「主流」等检索价值低的泛词；**具体题也统一补充维度词**。

**效果**：Q18 首次命中（改写 `Agent框架 架构设计 工具调用 记忆机制 上下文管理 安全机制 适用场景 异同对比`），但 **Q2/Q5/Q9 回归**——具体题被强加的「上下文管理/架构设计/计算流程」等泛词挤掉了目标文档，4 项验收失败（preserve_baseline_hits / single MRR / single NDCG / multi 逐题）。

**教训**：对具体题加维度词 = 用无关词污染精确查询。

### 阶段 3：v2.1 分支规则雏形（仍留缝）

**规则**：先判断问题类型——具体题保留关键词、模糊题维度化；但保留"可补充同义表达/英文全称"的余地。

**效果**：Q18 保持命中，但 **Q3 回归**——改写加入 `Retrieval-Augmented Generation` 等英文全称后，目标文档被挤到 rank6 掉出 top-5；另有 Q1/Q6/Q9 改写异常回退。preserve_baseline_hits 与 multi 逐题仍挂。

**教训**："可以补充同义/全称"就是给回归留缝——凡是"可加可不加"的权限，模型大概率会加。

### 阶段 4：v2.2 恒等 + 分支（定稿）

**最终规则**（`PROMPT_VERSION=query-rewrite-v2.2`，temperature=0.0）：

| 问题类型 | 判定方式 | 改写动作 |
|---|---|---|
| 具体题（含明确实体/概念） | 交 LLM 判定 | **恒等改写**：保留全部关键词，仅删「是什么/如何/为什么」等提问词；**不得添加**英文全称、同义表达或任何新词 |
| 模糊题（类别比较/概念提问） | 交 LLM 判定 | **删泛词 + 维度化**：删除「主流/框架/异同」等低价值词，用主题维度词补充（架构设计、记忆机制、工具调用、上下文管理、安全机制、适用场景、优劣对比） |
| 公共约束 | — | 禁止引入原问题未提及的任何实体/产品/框架名；只输出一条查询，不带标题引号解释 |

**为什么把"模糊/具体"判定交给 LLM**：用正则硬判（如"含多个名词=模糊"）覆盖不了"主流 Agent 框架"这类看似具体实则模糊的表达；v2.2 让模型用 temperature=0 判定，并把两种规则写成**互斥二选一**，不给"既可加也可不加"的模糊地带。

**效果**：11/11 验收全过，20/20 命中，零回归（见第三节）。

---

## 三、结果提升对比（数据）

### 3.1 主指标：A（基线） vs B（v2.2 改写）

| 口径 | 组 | Hit Rate@5 | Recall@5 | MRR@5 | NDCG@5 | micro Recall |
|---|---:|---:|---:|---:|---:|---:|
| single | A | 0.950 | 0.950 | 0.8625 | 0.8846 | 0.950 |
| single | **B** | **1.000** | **1.000** | **0.9083** | **0.9315** | **1.000** |
| multi | A | 0.950 | 0.8583 | 0.9250 | 0.8628 | 0.7778 |
| multi | **B** | **1.000** | **0.9083** | **0.9667** | **0.9034** | **0.8519** |

> 口径说明：A 为**同环境同快照重跑**的未舍入基线（0.8625/0.8846），与 Hybrid 文档的历史 `single@8 0.863/0.885` 展示值不同——验收一律用本环境 A 组原始值比较，避免"显示相同实际下降"。

**读表要点**：B 组在 single/multi 双口径下四项指标全部上升；multi micro Recall 0.7778→0.8519，多文档题覆盖显著改善。

### 3.2 故障注入对照：D1 / D2 与 A 完全一致

| 组 | 含义 | 结果 |
|---|---|---|
| D1 | 关闭改写（rewrite_enabled=False）| 0 次 LLM 调用；20 题 retrieved_documents 与 A **逐题完全相等** |
| D2_timeout | 注入 60s 卡死 + 10ms 预算 | 全部回退原 query；状态 `fallback/timeout`；结果 = A |
| D2_exception | 注入 RuntimeError | 全部回退原 query；状态 `fallback/error:RuntimeError`；结果 = A |
| D2_empty | 注入空文本返回 | 全部回退原 query；状态 `empty/empty_output`；结果 = A |

三类故障 × 20 题 = 60 项检查全 true，且 `rewritten_query is None` + `effective_query == 原问题` + 检索结果与 A 相等——**失败即无害**。

### 3.3 Q18 案例：双路 MISS → 双路 rank1

| 组 | 送入检索的 query | Dense 候选池含目标 | BM25 候选池含目标 | top-5 |
|---|---|---|---|---|
| A | `主流 Agent 框架有什么异同` | ❌ | ❌ | agent_vs_workflow / rag_interview / chapter1 / agent_skill_interview / agent_interview |
| **B** | `Agent 架构设计 记忆机制 工具调用 上下文管理 安全机制 适用场景 优劣对比` | ✅ | ✅ | **agent_framework_comparison.md** / claude_code_deep_dive / chapter10 / production_agent_architecture_harness / chapter3 |

目标文档从双路候选池零重叠 → 双路候选池命中 + top-5 rank1。

### 3.4 回归控制：B 组 3 题回退，不构成回归

B 组 20 题：**17 success + 3 fallback（Q11/Q15/Q19）**。回退意味着用原 query 检索（= A 组基线），因此：

- Q11 `MCP 协议是什么，解决什么问题` → 回退原问题 → 结果与 A 完全一致（不改写也能命中）；
- Q15 `Agent 的性能应该如何评估`、Q19 `工作流和自主 Agent 有什么区别` → 同上；
- 验收项 `preserve_baseline_hits`（A 已命中题逐题不降）保证这 3 题没有被"半截改写"破坏。

**结论：回退 ≠ 失败，是"LLM 可用时改写、不可用时退化为基线"的自愈路径。**

### 3.5 改写性能（rewrite_statistics）

| 指标 | 值 |
|---|---:|
| success_rate | 0.85（17/20）|
| empty_rate | 0.0 |
| fallback_rate | 0.15（3/20）|
| 平均改写延迟（live）| 8300 ms |
| LLM tokens（input/output）| 1725 / 4565 |

> 延迟 8.3s 来自 deepseek 的 thinking 推理；**生产接入后每次改写最多占 15s 预算（含重试），超时自动回退原 query**——对检索主链路是"可接受的等待 + 自愈"，不阻塞。

### 3.6 chunk token 超长截断修复（2026-10-10，独立于改写）

**背景**：生产 daemon 出现 `[transformers] Token indices sequence length is longer than the specified maximum sequence length (1070 > 512)` 警告——`BasicChunker` 用 1600 字符固定窗口纯字符切分，`bge-small-zh-v1.5`（max_seq_length=512）对超长 chunk 静默截断，尾部信息被丢弃且无提示。

**改造**：`BasicChunker` 增加 token 模式（`max_tokens=480, overlap_tokens=48`，给 512 留余量）；`_split_tokens` 按 tokenizer encode 全文 → 窗口 decode → 段落边界回退（`\n\n` → `\n`）→ 末段兜底（残余 <50 字符时整段重取覆盖，防止被 `MIN_CHUNK_CHARS` 过滤静默丢尾）；signature 变化触发 content_hash 版本变化 → 全量重建索引。真实 tokenizer 下 decode 重建的 re-encode 有 ±几 token 抖动（最大 483），仍在 512 上限内。

**修复前后对比（scan_chunk_tokens.py 复跑）**：

| 项目 | 旧索引（chars 1600/200） | 新索引（token 480/48） |
|---|---:|---:|
| chunk 总数 | 2,989 | 5,265（+76%，窗口变细）|
| >512 token | 733（24.5%）| **0（0.0%）** |
| 最大 token | 1,500 | 483 |
| 平均 token | 387.7 | 293.8 |

> 中间产物清理：重建时 11 篇评测报告/方案（QueryRewrite_*、rewrite_results*/report.md 等）被当知识入库，扩展 `index_knowledge.py` 的 EXCLUDE 后按真实相对路径补删（`scripts/clean_residue_docs.py`）；旧索引备份在 `knowledge_docs/index_snapshot_chars_v1600/`（勿动）。

### 3.7 新索引重测：A/B/C 三组（当前权威数据）

20 题 golden、同一 token 切分索引快照、同参数（每路 16 候选、RRF=60、sum 聚合、top_k=8），A/B/C 三组结果并存于 `rewrite_results_v3_token/`（A+D1/D2）、`rewrite_results_v3_token_b/`（+B）、`rewrite_results_v3_token_c/`（+C，复用同批改写缓存，严格可比）。

| 口径 | 组 | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | micro Recall |
|---|---:|---:|---:|---:|---:|---:|
| single | A（原问题）| 0.950 | 0.950 | 0.8875 | 0.9031 | 0.950 |
| single | B（单改写）| **1.000** | **1.000** | **0.9375** | **0.9531** | **1.000** |
| single | **C（双路融合）** | **1.000** | **1.000** | **0.9375** | **0.9531** | **1.000** |
| multi | A | 0.950 | 0.9083 | 0.9500 | 0.8992 | 0.8519 |
| multi | B | 1.000 | 0.8833 | 1.0000 | 0.9048 | 0.8148 |
| multi | **C** | **1.000** | **0.9333** | **1.0000** | **0.9285** | **0.8889** |

**对比要点**：
- **A 组（生产默认路径）**：比字符切分索引提升——MRR 0.8625→0.8875、NDCG 0.8846→0.9031、Hit 0.95 持平；切分变细后相关片段排名更靠前；
- **B 组（仅改写）**：single 全绿（20/20、Q18 命中），但 **multi 出现 1 题回归**——Q2 的 react_reflection_planning.md 被 claude_code_deep_dive 挤出 top5（关键词堆叠式改写在小 chunk 索引上更容易被"词频密集但不相关"的 chunk 抢位）；
- **C 组（原查询 + 改写查询双路检索，文档级 RRF 分数相加融合）**：single 保持 B 的水平，**multi 全面反超 A**（recall 0.9333、micro 0.8889 为三组最高）——原查询路把 B 组丢的 Q2/Q19 补回，改写路解决 Q18。

### 3.8 C 组验收与逐题（多文档题）

C 组验收五项全 true：single_hit ≥0.95 / preserve_baseline_hits / multi 每题不降 / multi micro 不降 / Q18 进 top5。

| Q | A recall_m | B recall_m | C recall_m | 说明 |
|---|---:|---:|---:|---|
| Q2 | 1.0 | 0.5 | **1.0** | react_reflection_planning 被原查询路补回（top5 第 5 位）|
| Q3 | 1.0 | 1.0 | 1.0 | 持平 |
| Q9 | 0.667 | 0.667 | 0.667 | 持平 |
| Q16 | 0.5 | 0.5 | 0.5 | 两路都召不回 related 文档，非融合可解 |
| Q18 | 0.0 | 0.5 | 0.5 | 改写路命中主文档，原查询路兜底 agent_vs_workflow |
| Q19 | 1.0 | 0.5 | **1.0** | dynamic_workflows 被原查询路补回 |

**融合实现**（`eval_rag_rewrite.py` `--c-enabled`）：原查询、改写查询各走一遍生产 `HybridRetriever`（RRF+sum 聚合），两路 top-8 文档按 `document_id` 分数相加（两路同为 Hybrid 输出，量纲一致），按 `(-score, first_seen)` 重排取 top-8。改写失败/空输出时 C 组退化为原问题单路（= A 组），不影响主链路。

---

## 四、问题解决情况对照

| 此前记录的问题 | 阶段归属 | 现状 |
|---|---|---|
| Q18 框架对比盲区（Dense/trigram/聚合三层 MISS）| Query Rewrite v2.2 | ✅ 双路 rank1（原问题双路 MISS）；新索引重测仍命中 |
| 改写空输出（thinking 吃光预算）| v2 修复 | ✅ stop_reason/thinking_chars 诊断 + MAX_TOKENS 1024 |
| 具体题被泛词污染（Q2/Q5/Q9）| v2.2 恒等改写 | ✅ 零回归（preserve_baseline_hits 通过）|
| "补充同义/全称"留缝（Q3 回归）| v2.2 二选一规则 | ✅ 具体题禁止添加任何新词 |
| 失败路径安全性 | D1/D2 故障注入 | ✅ 60 项检查全 true，结果与 A 一致 |
| chunk 超长静默截断（1070>512 丢尾部）| token 感知切分 v3（2026-10-10）| ✅ >512 从 733→0；新索引 A/B/C 重测通过 |
| 单改写 multi 回归（Q2/Q19）| C 组双路融合（2026-10-10）| ✅ multi recall 0.9333 > A 组 0.9083，验收五项全绿 |

---

## 五、当前生产状态与配置

| 项目 | 状态 |
|---|---|
| 生产接入 | ✅ 已接入 `KnowledgeService.search` 编排层（2026-10-10）|
| 默认行为 | **关闭**：`rewrite_enabled=false` 时不创建 QueryRewriter、0 次 LLM 调用、0 额外延迟，行为与接入前完全一致 |
| 失败回退 | 非 success（empty/fallback/超时/异常）一律用原问题检索 |
| 懒加载 | `rewriter` 在第一次启用改写时才创建；超时预算由构造参数注入 |
| 边界保持 | `HybridRetriever` 零修改；`AgentLoop` 零修改；KnowledgeService 只做编排不做检索实现 |
| 当前索引 | token 感知切分 v3（480/48），5265 chunks，`>512=0`；生产检索链路 = A 组路径（B/C 需显式开启改写）|
| C 组双路融合 | **实验已验证（multi 最优），暂不接入生产**——与单改写共用 `rewrite_enabled` 开关，默认关闭保持零成本；是否作为"改写开启"时的检索策略，待生产端到端验证后决策 |

启用方式（默认关闭，无需改动）：

```toml
[knowledge]
rewrite_enabled = true
rewrite_timeout_budget_s = 15.0
```

或环境变量 `KAMA_KNOWLEDGE_REWRITE_ENABLED=true` / `KAMA_KNOWLEDGE_REWRITE_TIMEOUT_BUDGET_S=15`。

**重要边界**：
1. D1/D2 目前验证的是**评测编排路径**（`eval_rag_rewrite.py` 的 select_query）；生产 Service 接入后的开关与回退行为，需要真实对话端到端验证（下一步 P0）；
2. 改写结果**不落缓存到生产路径**（评测缓存只用于离线复现）；生产每次调用都会真调 LLM，延迟约 8s——上线前需评估是否引入缓存层；
3. 不改写原问题元数据，失败信息只记录 status/reason，不含 LLM 异常正文（防凭据泄漏）。

---

## 六、结论与后续方向

**阶段结论**：
1. **查询表达是 RAG 召回盲区的真实成因**：Q18 在 Dense/trigram/聚合三层都召不回，但改写为维度词后双路 rank1——检索器已到天花板，改写是正确发力点；
2. **受约束改写优于自由改写**：v1 补实体（常识≠知识库）、v2 全量加维度词（污染具体题）、v2.1 留缝（回归）三次失败收敛出 v2.2 的"恒等 + 分支"——**不给模型任何"可加可不加"的权限**是零回归的关键；
3. **评测体系先于生产**：A/B 证明效果、D1 证明透明、D2 证明自愈，三类证据齐了才允许接入，且默认关闭；
4. **chunk 上限必须按 embedding 的 max_seq_length 设计**：字符切分保证不了 token 上限（1 字符≠1 token），token 感知切分 + 段落边界回退 + 末段兜底后 `>512=0`，且细粒度索引让 A 组 MRR/NDCG 不降反升；
5. **双路融合是改写副作用的对冲**：单改写（B）在细粒度索引上可能被"词频密集不相关"的 chunk 抢位（Q2/Q19），原查询路 + 改写路双路融合（C）后 multi 全面反超基线，single 保持满分——**C 组是"改写开启"时的最优检索策略**（实验证据齐备，生产接入待 P0 后决策）。

**后续优先级**：

| 优先级 | 改进项 | 目标 | 验证方式 | 状态 |
|---|---|---|---|---|
| P0 | 生产端到端验证 | 真实对话确认改写开关/回退生效 | 开启 rewrite_enabled 后真实提问 Q18 类问题，对照 daemon trace | 待做 |
| P1 | C 组双路融合生产接入 | 把已验证的融合策略接入"改写开启"路径 | 在 KnowledgeService 编排层加 `rewrite_dual=true`（默认关），复用评测验证数据 | **实验已验证，接入待决策** |
| P1 | 生产缓存改写结果 | 消除 8s 延迟 | 按 query hash + prompt 版本缓存，TTL 可控 | 待评估 |
| P2 | Reranker（cross-encoder 精排）| 位次靠后题目 | ragas 四项指标上线评测后决定 | 待 P0/P1 后 |
| P2 | 扩大 golden 集 | 20 题不足以代表真实查询分布 | 收集真实用户查询标注 | 待定 |

---

## 附录：复跑命令与文件清单

```powershell
# 全量评测（默认关闭 = A/D1/D2；--rewrite-enabled 追加 B 组；--c-enabled 追加 C 组双路融合）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag_rewrite.py --output <新目录>
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag_rewrite.py --rewrite-enabled --output <新目录>
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag_rewrite.py --rewrite-enabled --c-enabled --output <新目录>
# 复用已有改写缓存（同一模型/prompt/参数），避免重复调 LLM
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag_rewrite.py --rewrite-enabled --c-enabled --cache <旧目录>\rewrite_cache.json --output <新目录>
# prompt/模型参数变更后必须显式重生成缓存
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag_rewrite.py --rewrite-enabled --refresh-cache --output <新目录>

# chunk token 分布扫描（改造前后对比的权威脚本）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\scripts\scan_chunk_tokens.py

# 全量入库（token 切分 v3；EXCLUDE 已扩展排除中间产物）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\index_knowledge.py

# 单测（rewrite 模块 + service 接入 + chunker token 模式）
& D:\tinyclaude\.venv\Scripts\python.exe -m pytest tests/unit/test_knowledge_rewrite.py tests/unit/test_knowledge_service_rewrite.py tests/unit/test_knowledge_chunker.py -q
```

| 文件 | 用途 |
|---|---|
| `src/kama_claude/core/knowledge/rewrite.py` | QueryRewriter：结构化结果 + 15s 总预算 + 三层校验（v2.2）|
| `src/kama_claude/core/llm/provider.py` | `complete_text` 非流式入口 + thinking 块解析 |
| `src/kama_claude/core/knowledge/service.py` | `search()` 编排层接入（`_maybe_rewrite`，默认关闭懒加载）|
| `src/kama_claude/core/config.py` / `app.py` | `rewrite_enabled` / `rewrite_timeout_budget_s` 配置与注入 |
| `src/kama_claude/core/knowledge/chunker/basic.py` | **token 感知切分 v3**：`max_tokens=480/overlap=48`、段落回退、末段兜底（signature v3 触发重建）|
| `knowledge_docs/index_knowledge.py` | 全量入库：chunker 注入 + EXCLUDE 扩展（排除中间产物）|
| `knowledge_docs/eval_rag_rewrite.py` | 评测编排：A/B/C/D1/D2 + 缓存 + 验收（`--c-enabled` 双路融合）|
| `knowledge_docs/golden_set.json` | 20 问 ground truth（Q2/Q3/Q9/Q16/Q18/Q19 多相关）|
| `knowledge_docs/rewrite_results_v2_2/` | v2.2 定稿全量输出（字符切分索引时代，勿覆盖）|
| `knowledge_docs/rewrite_results_v3_token/` | token 切分索引 A/D1/D2（新基线）|
| `knowledge_docs/rewrite_results_v3_token_b/` | token 切分 + B 组（单改写）|
| `knowledge_docs/rewrite_results_v3_token_c/` | token 切分 + C 组（双路融合，multi 最优，验收全绿）|
| `knowledge_docs/index_snapshot_chars_v1600/` | 字符切分旧索引备份（qdrant+catalog+lexical，勿动）|
| `scripts/scan_chunk_tokens.py` / `debug_chunk_over.py` / `clean_residue_docs.py` | token 分布扫描 / 假警报诊断 / 中间产物清理 |
| `knowledge_docs/rewrite_results_v2/` `rewrite_results_v2_1/` | 中间失败版本数据（保留对比）|
| `knowledge_docs/rewrite_checkpoint_20261010/` | 首轮失败检查点（空输出 + 网络异常，未动）|
| `knowledge_docs/diag_q18_variants.py` / `diag_target_doc.py` / `test_q18_rewrite.py` / `test_4q_regression.py` | 诊断与回归证据脚本 |
| `tests/unit/test_knowledge_rewrite.py` / `test_knowledge_service_rewrite.py` / `test_knowledge_chunker.py` | 模块 + 生产接入 + chunker token 模式单测 |

运行时数据：`~/.kama/knowledge/`（qdrant **5265 向量** 512 维 + lexical.db，FTS5 trigram，bm25 权重 1.0/1.2/1.5）；embedding `bge-small-zh-v1.5`（snapshot `7999e1d3…`，max_seq_length=512）；改写模型 `deepseek-v4-pro`（temperature=0，max_tokens=1024，timeout 15s）。
