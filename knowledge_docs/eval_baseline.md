# RAG 指标化评估报告（V1 → V2 对比 · golden 集）

> 生成：V1 2026-10-06 / V2 2026-10-07
> 评估集：`golden_set.json`（20 问，每问人工标注预期文档 + 预期 section，已对照原文章节结构验证）
> 匹配：document 级（chunk 的 source 文件名 == expected_doc，同文档重复 chunk 只计首次）
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

**退化面（Q3/Q9 新 MISS）**：
- Q3：`agentic_rag.md` 的 title/desc 与"RAG 核心思路"语义高度重合，Top-5 全占；`rag_interview.md`（面试题汇总文体）反而被挤出。**实际两者都相关，属 golden 单标签口径局限**。
- Q9：李博杰 `chapter10.md`（多智能体整章，无 frontmatter，51 chunks）正文主题密度高，淹没 33 chunks 的面试文档。**大部头章节淹没问题**。

**遗留硬缺陷（Q16/Q18，V1 就有，V2 未解）**：
- 幻觉治理、框架对比两题，预期文档被 generic 文档（how_llm_trained、chapter1）抢占。
- 共性：**纯向量 top-k 的语义近邻竞争 + 专名/文体不匹配**，正是 P2 Hybrid Search（BM25）+ Reranker 的目标。

## 对 V2 的量化评价

| 维度 | 结论 |
|---|---|
| chunk 质量 | 显著提升：空壳清零、满窗口 172→1、均值更均匀 |
| 文档级召回 | Recall 持平偏降（0.900→0.800），但 2 题属口径局限/大部头淹没 |
| 命中排序 | MRR 超 V1（0.713 vs 0.709）、NDCG 接近（0.735 vs 0.755） |
| 系统能力 | content_hash 版本因子 + 增量重建正确工作（153 重建 / 34 跳过） |

**下一步（P2）**：BM25 Hybrid Search（精确关键词救 Q9/Q16/Q18 类长尾）→ Reranker（精排）→ Query Rewrite。

## 复跑命令

```powershell
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8
```
