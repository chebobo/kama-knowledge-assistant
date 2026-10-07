# RAG 指标化评估报告（V1 → V2 对比 · golden 集 · 双口径）

> 生成：V1 2026-10-06 / V2 2026-10-07 / 双口径 2026-10-07
> 评估集：`golden_set.json`（20 问，每问人工标注预期文档 + 预期 section；6 题标注 related_docs）
> 匹配：document 级（chunk 的 source 文件名 == 目标文档，同文档重复 chunk 只计首次）
> 口径：**single（主指标）**= 命中 expected_doc 才 rel=1；**multi（辅助）**= 命中任一 related_docs 即 rel=1（仅 Q2/Q3/Q9/Q16/Q18/Q19 多答案，其余两口径一致）
> 链路：真实 Qdrant Local + bge-small-zh-v1.5
> 索引：V1 = 187 篇 / 3237 chunks（含 frontmatter/空壳）；V2 = 187 篇 / 2989 chunks（跳 frontmatter/滤空壳/段落边界 + 标题/描述注入）

## 数值对比（k=5）

| 指标 | V1 | V2 | 变化 |
|---|---|---|---|
| Recall@5 | 0.900 (18/20) | 0.800 (16/20) | **-0.100** |
| MRR@5 | 0.709 | **0.713** | **+0.004（超 V1）** |
| NDCG@5 | 0.755 | 0.735 | -0.020 |

## 逐问对比（k=5）

| Q | 问题 | 预期文档 | V1 rank | V2 rank | 变化 |
|---|---|---|---|---|---|
| 1 | 混合路由 | agent_hybrid_routing_interview | 1 | 1 | = |
| 2 | ReAct 循环 | chapter1（李博杰） | 1 | 1 | = |
| 3 | RAG 核心思路 | rag_interview | 1 | **MISS** | **退化** |
| 4 | Agent 记忆分类 | agent_memory | 1 | 1 | = |
| 5 | 多头注意力计算 | mha | 5 | **1** | **救回** |
| 6 | 模型蒸馏 | model_distillation | 1 | 1 | = |
| 7 | Claude Code 上下文 | claude_code_context_window_interview | 1 | 1 | = |
| 8 | Function Calling | function_calling | 1 | 1 | = |
| 9 | 多 Agent 通信 | multi_agent_communication_interview | 1 | **MISS** | **退化** |
| 10 | 工具设计 | agent_tool_design | 1 | 1 | = |
| 11 | MCP 协议 | mcp_protocol | 4 | **1** | **救回** |
| 12 | 向量数据库 | vector_database | 1 | 1 | = |
| 13 | RAG 评估 | rag_evaluation | 1 | 1 | = |
| 14 | 上下文工程 | context_engineering | 3 | 2 | 前移 |
| 15 | Agent 评估 | agent_evaluation | 1 | 1 | = |
| 16 | 幻觉治理 | agent_hallucination_control_interview | MISS | **MISS** | 未解（硬缺陷） |
| 17 | 微调 vs RAG | finetuning_vs_rag | 5 | 4 | 前移 |
| 18 | Agent 框架对比 | agent_framework_comparison | MISS | **MISS** | 未解（硬缺陷） |
| 19 | Workflow vs Agent | agent_vs_workflow | 5 | 2 | 前移 |
| 20 | GraphRAG | graphrag_interview | 1 | 1 | = |

## 结论：V2 改良的实际效果

**改善面（Q5/Q11 救回，Q14/Q17/Q19 前移）**：
- 根源是 **frontmatter 中文 title/description 注入**：Q5 的 mha.md 靠 description 中"多头注意力"中文信号从 rank5 → rank1；Q11 mcp 从 rank4 → rank1。
- 段落边界切分让 chunk 语义完整，排序整体前移。

**退化面（Q3/Q9 新 MISS）——已被双口径评测正名**：
- Q3：`agentic_rag.md` 与"RAG 核心思路"语义高度重合，Top-5 全占；multi 口径下 **agentic_rag 是相关文档 → 命中（rank=1）**。**属 golden 单标签口径局限，非检索缺陷**。
- Q9：李博杰 `chapter10.md`（多智能体整章，无 frontmatter，51 chunks）正文主题密度高，淹没 33 chunks 的面试文档；multi 口径下 **chapter10 是相关文档 → 命中（rank=1）**。**属大部头章节淹没 + 单标签局限**。
- 注意：Q3/Q9 的 Top-5 被**同一个文档的多个 chunk 霸榜**（agentic_rag ×5、chapter10 ×4），说明即使多标签，检索也只找回了"一个"相关文档——**文档频率偏置/结果多样性不足**是新暴露的问题，与 Q9 大部头淹没同源，指向后续 Document Aggregation / Hybrid 多样性处理。

**遗留硬缺陷（Q16/Q18，双口径下均仍 MISS，确认真缺陷）**：
- Q16：Top-5 = how_llm_trained / llm_keywords / why_rag，**连 related_docs（agent_drift_hallucination_interview）都没进 Top-5**。
- Q18：Top-5 = chapter1 / agent_vs_workflow，**连 related_docs（harness_interview）都没进 Top-5**。
- 结论：多标签口径无法救回这两题——不是评测判错，而是**检索确实没有返回任何合理相关的文档**。这是纯向量 top-k 的语义近邻竞争 + 专名/文体不匹配，正是 P2 Hybrid Search（BM25）的目标。

## 双口径评测结果（2026-10-07，真实索引重跑）

**k=5：**

| 口径 | Query Hit Rate@5 | Recall@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| single（主） | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| multi（辅助） | **0.900 (18/20)** | **0.792** | **0.812** | **0.757** |

**k=8（与 k=5 一致，相关文档均在 top-5 内或未进 top-8）：**

| 口径 | Query Hit Rate@8 | Recall@8 | MRR@8 | NDCG@8 |
|---|---:|---:|---:|---:|
| single（主） | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| multi（辅助） | **0.900 (18/20)** | **0.792** | **0.812** | **0.757** |

- 差异来源：Q3、Q9 两题（multi 命中、single MISS）——正是先前归因"单标签口径局限"的两题，现已实证。
- `Query Hit Rate` 表示每题是否至少命中一个相关文档；严格 `Recall@k` 表示命中的相关文档数占该题 related_docs 总数。
- **Hit Rate (0.900) ≠ Recall (0.792)**：多标签题普遍只召回 1 个相关文档——Q3 仅召回 agentic_rag（漏 rag_interview）、Q9 仅召回 chapter10（3 个 related 只中 1 个）、Q19 仅召回 agent_vs_workflow（漏 dynamic_workflows）。这是**结果多样性不足**的量化证据，与单文档霸榜同源。
- Q16/Q18 两口径均 MISS → **真缺陷确认**，评测修正不会掩盖真实问题，这是双口径设计的价值。
- Q2/Q19 的 related 文档未改变判定（expected 本身已命中）。

## 对 V2 的量化评价

| 维度 | 结论 |
|---|---|
| chunk 质量 | 显著提升：空壳清零、满窗口 172→1、均值更均匀 |
| 文档级召回 | Recall 持平偏降（0.900→0.800），但 2 题属口径局限/大部头淹没 |
| 命中排序 | MRR 超 V1（0.713 vs 0.709）、NDCG 接近（0.735 vs 0.755） |
| 系统能力 | content_hash 版本因子 + 增量重建正确工作（153 重建 / 34 跳过） |

**下一步**：Hybrid 已完成实现与评测；根据 Q9 单文档 chunk 霸榜仍存在，建议进入独立 Document Aggregation 评估。Query Rewrite 和 Reranker 继续后置。

## 复跑命令

```powershell
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5 hybrid
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8 hybrid
```

## V2 Dense 与 Hybrid 受控对比（2026-10-07）

索引版本：V2 chunker + `bge-small-zh-v1.5` + FTS5 trigram v1。全量同步后 Catalog 为 187 篇 / 2,989 chunks，FTS5 为 187 documents / 2,989 rows；两份 Hybrid 执行/设计计划不属于 Golden Set 数据集，已从索引排除。Hybrid 候选倍率为 2，RRF `k=60`。评测时间：UTC 2026-10-07 05:05-05:09。

| k | Retriever | 口径 | Query Hit Rate | 严格 Recall | MRR | NDCG |
|---:|---|---|---:|---:|---:|---:|
| 5 | Dense | single | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| 5 | Hybrid | single | 0.850 (17/20) | 0.850 | 0.738 | 0.766 |
| 5 | Dense | multi | 0.900 (18/20) | 0.792 | 0.812 | 0.757 |
| 5 | Hybrid | multi | 0.900 (18/20) | 0.792 | 0.812 | 0.761 |
| 8 | Dense | single | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| 8 | Hybrid | single | 0.900 (18/20) | 0.900 | 0.765 | 0.797 |
| 8 | Dense | multi | 0.900 (18/20) | 0.792 | 0.812 | 0.757 |
| 8 | Hybrid | multi | 0.950 (19/20) | 0.842 | 0.823 | 0.772 |

Dense/Hybrid 命令分别为 `uv run python knowledge_docs/eval_rag.py 5 dense|hybrid` 和 `uv run python knowledge_docs/eval_rag.py 8 dense|hybrid`。评测器输出每题 rank 和 top-k 文档；Dense 与 Hybrid 使用同一个 Qdrant 索引、scope、embedding 和 Golden Set。

| Query | Dense k=5 | Hybrid k=5 | Hybrid k=8 | 观察 |
|---|---:|---:|---:|---|
| Q3 RAG 核心思路 | single MISS / multi 1 | single 2 / multi 1 | single 3 / multi 2 | single 标签救回，相关文档靠后 |
| Q5 多头注意力 | 1 | 2 | 3 | Hybrid 排名退后 |
| Q9 多 Agent 通信 | single MISS / multi 1 | single MISS / multi 1 | single MISS / multi 1 | 仍被 `chapter10.md` 多 chunks 占位 |
| Q11 MCP 协议 | 1 | 2 | 2 | Hybrid 排名退后 |
| Q14 上下文工程 | 2 | 1 | 1 | 提升 |
| Q16 幻觉治理 | MISS | 4 | 1 | Hybrid 在 k=5/k=8 均救回 |
| Q17 微调 vs RAG | 4 | MISS | 8 | k=5 退化，k=8 恢复但靠后 |
| Q18 Agent 框架对比 | MISS | MISS | MISS | 仍未解决 |
| Q19 Workflow vs Agent | 2 | 1 | 1 | 提升 |

结论：Hybrid 改善 single Recall、MRR 和 NDCG，且 Q16 的精确概念漏召回被救回；Q5/Q11/Q17 出现局部退化，multi Recall@5 没有提升，说明 BM25 trigram + RRF 并未普遍改善多文档覆盖。Q9 的单文档霸榜仍在，支持将 Document Aggregation 作为下一独立实验；评测未证明 aggregation 会增加 Recall，需保持单独对照。暂不进入 Query Rewrite 或 Reranker。
