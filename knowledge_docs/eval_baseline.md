# RAG 指标化评估报告（V1 基线 · golden 集）

> 生成：2026-10-06
> 评估集：`golden_set.json`（20 问，每问人工标注预期文档 + 预期 section，已对照原文章节结构验证）
> 匹配：document 级（chunk 的 source 文件名 == expected_doc，同文档重复 chunk 只计首次）
> 链路：真实 Qdrant Local + bge-small-zh-v1.5，187 篇 / 3237 chunks

## V1 数值基线

| 指标 | k=5 | k=8 |
|---|---|---|
| Recall@k | 0.900 (18/20) | 0.900 (18/20) |
| MRR@k | 0.709 | 0.709 |
| NDCG@k | 0.755 | 0.755 |

Top-8 相比 Top-5 无新增命中 → 2 个 MISS 在 k=8 仍未召回，是**硬缺陷**而非排序问题。

## 逐问明细（k=5）

| Q | 问题 | 预期文档 | rank | 备注 |
|---|---|---|---|---|
| 1 | 混合路由 | agent_hybrid_routing_interview | 1 | |
| 2 | ReAct 循环 | chapter1（李博杰） | 1 | |
| 3 | RAG 核心思路 | rag_interview | 1 | |
| 4 | Agent 记忆分类 | agent_memory | 1 | |
| 5 | 多头注意力计算 | mha | **5** | 差一位，NDCG 0.387 |
| 6 | 模型蒸馏 | model_distillation | 1 | |
| 7 | Claude Code 上下文 | claude_code_context_window_interview | 1 | |
| 8 | Function Calling | function_calling | 1 | |
| 9 | 多 Agent 通信 | multi_agent_communication_interview | 1 | |
| 10 | 工具设计 | agent_tool_design | 1 | |
| 11 | MCP 协议 | mcp_protocol | **4** | NDCG 0.431 |
| 12 | 向量数据库 | vector_database | 1 | |
| 13 | RAG 评估 | rag_evaluation | 1 | |
| 14 | 上下文工程 | context_engineering | **3** | NDCG 0.500 |
| 15 | Agent 评估 | agent_evaluation | 1 | |
| 16 | 幻觉治理 | agent_hallucination_control_interview | **MISS** | Top-5 全不相关 |
| 17 | 微调 vs RAG | finetuning_vs_rag | **5** | |
| 18 | Agent 框架对比 | agent_framework_comparison | **MISS** | Top-5 全不相关 |
| 19 | Workflow vs Agent | agent_vs_workflow | **5** | |
| 20 | GraphRAG | graphrag_interview | 1 | |

## 2 个硬缺陷分析（V2 改良目标）

### Q16 幻觉治理 → agent_hallucination_control_interview.md MISS
实际 Top-5：how_llm_trained / 20260506bytedance / llm_keywords / why_rag
- "幻觉"是 LLM 训练/词条类文档的常见词，向量空间被 generic 文档抢占
- 预期文档标题带"Agent系统如何约束"，query 中"治理手段"的语义未被 bge 充分匹配
- **候选改良**：Hybrid Search（BM25 关键词命中"幻觉+治理"）或 query 重写（"幻觉治理" → "Agent 约束幻觉"）

### Q18 主流 Agent 框架对比 → agent_framework_comparison.md MISS
实际 Top-5：chapter1 ×3 / multi_agent_communication / chapter10
- "框架"与"Agent"高频出现在入门书籍章节，预期文档（三框架横评）被稀释
- **候选改良**：同上（混合检索 / 重写）；或文档标题更精准（OpenClaw/Hermes 专名检索时能命中，但 query 用"主流框架"太泛）

### 次要缺陷（rank 5 档，3 题）
Q5 mha、Q17 finetuning_vs_rag、Q19 agent_vs_workflow 均排在 Top-5 末位——NDCG 贡献仅 0.387。
这些是**排序问题**：相关文档能被召回但排位靠后，改良后有望前移。

## 复跑命令

```powershell
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8
```

## 与早前 10 问快照的关系
- 早前 10 问矩阵是"主观相关三分法"快照（Top-5 全相关、预期 Top-1 9/10）
- 本报告是**同一批数据上的硬指标**：20 问、document 级 golden、可复现数值
- 两套口径不冲突：快照看"命中质量"，本报告看"能否在固定窗口内找到预期文档"
