# Query Rewrite 第一阶段评测报告

运行时间（UTC）：2026-10-10T10:58:53.395043+00:00

参数：每路 16 个 chunk，RRF=60，sum 聚合后最多 8 个文档；主验收在文档 top-5。
历史 0.950/0.863/0.885 为 single @8；本报告 @5 使用同环境 A 组未舍入值。
D1/D2 只验证本评测入口编排，不代表生产 Service 已接入或验证。

## 指标（未舍入）

|组|口径|Hit|macro Recall|micro Recall|MRR|NDCG|
|---|---|---:|---:|---:|---:|---:|
|A|single@5|0.95|0.95|0.95|0.8875|0.9030803155822426|
|A|multi@5|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|A|single@8|0.95|0.95|0.95|0.8875|0.9030803155822426|
|A|multi@8|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D1|single@5|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D1|multi@5|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D1|single@8|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D1|multi@8|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_timeout|single@5|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_timeout|multi@5|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_timeout|single@8|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_timeout|multi@8|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_exception|single@5|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_exception|multi@5|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_exception|single@8|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_exception|multi@8|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_empty|single@5|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_empty|multi@5|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|D2_empty|single@8|0.95|0.95|0.95|0.8875|0.9030803155822426|
|D2_empty|multi@8|0.95|0.9083333333333334|0.8518518518518519|0.95|0.8991641682982466|
|B|single@5|1.0|1.0|1.0|0.9375|0.9530803155822426|
|B|multi@5|1.0|0.8833333333333334|0.8148148148148148|1.0|0.9047582361694252|
|B|single@8|1.0|1.0|1.0|0.9375|0.9530803155822426|
|B|multi@8|1.0|0.9333333333333333|0.8888888888888888|1.0|0.9253499281911969|

## 验收

A/B 状态：completed；全部通过：False

- D1_zero_llm_calls: True
- D1_equals_A: True
- D2_fallback_status_query_results: True
- Q18_success_top5: True
- single_hit_at_least_095: True
- preserve_baseline_hits: True
- single_mrr_non_decreasing: True
- single_ndcg_non_decreasing: True
- multi_each_question_non_decreasing: False
- multi_micro_recall_non_decreasing: False
- fixed_rewrite_retrieval_stable: True

## Q18 候选池及结果

- A: status=disabled; effective_query=主流 Agent 框架有什么异同
  - 目标文档进入候选池：{'dense': False, 'bm25': False}
  - top-5：['agent_vs_workflow.md', 'chapter1.md', 'rag_interview.md', 'agent_interview.md', 'agent_skill_interview.md']
- B: status=success; effective_query=Agent 架构设计 记忆机制 工具调用 上下文管理 安全机制 适用场景 优劣对比
  - 目标文档进入候选池：{'dense': True, 'bm25': True}
  - top-5：['agent_framework_comparison.md', 'claude_code_deep_dive.md', 'introduction.md', 'chapter3.md', 'chapter5.md']

## 逐题 A/B 对比

|Q|原问题|改写|状态|A Hit/MRR|B Hit/MRR|A multi Recall|B multi Recall|
|---|---|---|---|---|---|---|---|
|1|为什么单模型走不通？Agent 混合路由优化是什么|单模型走不通 Agent 混合路由优化|success|1.0/1.0|1.0/1.0|1.0|1.0|
|2|ReAct 循环是如何工作的，LLM、上下文和工具如何协同|ReAct 循环 工作 LLM 上下文 工具 协同|success|1.0/1.0|1.0/1.0|1.0|0.5|
|3|RAG 检索增强生成的核心思路是什么|RAG 检索增强生成 核心思路|success|1.0/0.25|1.0/0.25|1.0|1.0|
|4|Agent 的记忆系统分为哪几类||empty|1.0/1.0|1.0/1.0|1.0|1.0|
|5|Transformer 的多头自注意力机制是如何计算的|Transformer 多头自注意力机制 计算|success|1.0/1.0|1.0/1.0|1.0|1.0|
|6|什么是模型蒸馏？它到底在蒸馏什么|模型蒸馏 蒸馏|success|1.0/1.0|1.0/1.0|1.0|1.0|
|7|Claude Code 是怎么管理上下文窗口的（Auto-Compact）|Claude Code 上下文窗口管理 Auto-Compact|success|1.0/1.0|1.0/1.0|1.0|1.0|
|8|Function Calling 的原理是什么，大模型如何调用工具|Function Calling 原理 大模型 调用工具|success|1.0/1.0|1.0/1.0|1.0|1.0|
|9|多 Agent 之间如何通信协作|多 Agent 之间通信协作|success|1.0/0.5|1.0/0.5|0.6666666666666666|0.6666666666666666|
|10|Agent 的工具应该怎么设计|Agent 工具设计|success|1.0/1.0|1.0/1.0|1.0|1.0|
|11|MCP 协议是什么，解决什么问题|MCP协议 解决问题|success|1.0/1.0|1.0/1.0|1.0|1.0|
|12|向量数据库的核心概念和选型考虑|向量数据库 核心概念 选型考虑|success|1.0/1.0|1.0/1.0|1.0|1.0|
|13|RAG 系统应该如何评估|RAG 系统 评估|success|1.0/1.0|1.0/1.0|1.0|1.0|
|14|什么是上下文工程（Context Engineering）|上下文工程（Context Engineering）|success|1.0/1.0|1.0/1.0|1.0|1.0|
|15|Agent 的性能应该如何评估|Agent 性能 评估|success|1.0/1.0|1.0/1.0|1.0|1.0|
|16|大模型幻觉产生的原因和治理手段|大模型幻觉 产生原因 治理手段|success|1.0/1.0|1.0/1.0|0.5|0.5|
|17|微调（Fine-tuning）和 RAG 有什么区别，怎么选|微调 Fine-tuning RAG 区别 选择|success|1.0/1.0|1.0/1.0|1.0|1.0|
|18|主流 Agent 框架有什么异同|Agent 架构设计 记忆机制 工具调用 上下文管理 安全机制 适用场景 优劣对比|success|0.0/0.0|1.0/1.0|0.0|0.5|
|19|工作流（Workflow）和自主 Agent 有什么区别|工作流 Workflow 自主 Agent 区别|success|1.0/1.0|1.0/1.0|1.0|0.5|
|20|GraphRAG 是什么，和普通 RAG 有什么不同|GraphRAG 和普通 RAG 不同|success|1.0/1.0|1.0/1.0|1.0|1.0|

任一核心逐题指标下降题目：[2, 19]

## 改写性能

```json
{
  "success_rate": 0.95,
  "empty_rate": 0.05,
  "fallback_rate": 0.0,
  "mean_generation_latency_ms": 5703.362400000333,
  "live_mean_latency_ms": 5703.362400000333,
  "llm_tokens": {
    "input": 2025,
    "output": 6674
  },
  "tokens_note": "completed calls only; timed-out or retried usage may be unknown"
}
```

完整候选、document_id、状态、耗时、token、逐题指标见 per_question.jsonl；
模型、prompt/参数、快照指纹、原始验收结果见 summary.json。

限于当前 20 题与固定索引；不更新简历数据。
