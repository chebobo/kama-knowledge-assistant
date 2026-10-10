# Query Rewrite 第一阶段评测报告

运行时间（UTC）：2026-10-10T05:48:58.674977+00:00

参数：每路 16 个 chunk，RRF=60，sum 聚合后最多 8 个文档；主验收在文档 top-5。
历史 0.950/0.863/0.885 为 single @8；本报告 @5 使用同环境 A 组未舍入值。
D1/D2 只验证本评测入口编排，不代表生产 Service 已接入或验证。

## 指标（未舍入）

|组|口径|Hit|macro Recall|micro Recall|MRR|NDCG|
|---|---|---:|---:|---:|---:|---:|
|A|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|A|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|A|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|A|multi@8|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D1|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D1|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D1|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D1|multi@8|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_timeout|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_timeout|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_timeout|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_timeout|multi@8|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_exception|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_exception|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_exception|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_exception|multi@8|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_empty|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_empty|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|D2_empty|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|D2_empty|multi@8|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|B|single@5|0.95|0.95|0.95|0.8625|0.8846268032608154|
|B|multi@5|0.95|0.8583333333333334|0.7777777777777778|0.925|0.8627861572546065|
|B|single@8|0.95|0.95|0.95|0.8625|0.8846268032608154|
|B|multi@8|0.95|0.8833333333333334|0.8148148148148148|0.925|0.87245747743547|

## 验收

A/B 状态：completed；全部通过：False

- D1_zero_llm_calls: True
- D1_equals_A: True
- D2_fallback_status_query_results: True
- Q18_success_top5: False
- single_hit_at_least_095: True
- preserve_baseline_hits: True
- single_mrr_non_decreasing: True
- single_ndcg_non_decreasing: True
- multi_each_question_non_decreasing: True
- multi_micro_recall_non_decreasing: True
- fixed_rewrite_retrieval_stable: True

## Q18 候选池及结果

- A: status=disabled; effective_query=主流 Agent 框架有什么异同
  - 目标文档进入候选池：{'dense': False, 'bm25': False}
  - top-5：['agent_vs_workflow.md', 'rag_interview.md', 'chapter1.md', 'agent_skill_interview.md', 'agent_interview.md']
- B: status=empty; effective_query=主流 Agent 框架有什么异同
  - 目标文档进入候选池：{'dense': False, 'bm25': False}
  - top-5：['agent_vs_workflow.md', 'rag_interview.md', 'chapter1.md', 'agent_skill_interview.md', 'agent_interview.md']

## 逐题 A/B 对比

|Q|原问题|改写|状态|A Hit/MRR|B Hit/MRR|A multi Recall|B multi Recall|
|---|---|---|---|---|---|---|---|
|1|为什么单模型走不通？Agent 混合路由优化是什么||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|2|ReAct 循环是如何工作的，LLM、上下文和工具如何协同||fallback|1.0/0.5|1.0/0.5|0.5|0.5|
|3|RAG 检索增强生成的核心思路是什么||fallback|1.0/0.25|1.0/0.25|1.0|1.0|
|4|Agent 的记忆系统分为哪几类||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|5|Transformer 的多头自注意力机制是如何计算的||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|6|什么是模型蒸馏？它到底在蒸馏什么||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|7|Claude Code 是怎么管理上下文窗口的（Auto-Compact）||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|8|Function Calling 的原理是什么，大模型如何调用工具||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|9|多 Agent 之间如何通信协作||fallback|1.0/0.5|1.0/0.5|0.6666666666666666|0.6666666666666666|
|10|Agent 的工具应该怎么设计||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|11|MCP 协议是什么，解决什么问题||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|12|向量数据库的核心概念和选型考虑||empty|1.0/1.0|1.0/1.0|1.0|1.0|
|13|RAG 系统应该如何评估|RAG 检索增强生成 系统 评估方法 评估指标 检索质量 生成质量 忠实度 答案相关性|success|1.0/1.0|1.0/1.0|1.0|1.0|
|14|什么是上下文工程（Context Engineering）||empty|1.0/1.0|1.0/1.0|1.0|1.0|
|15|Agent 的性能应该如何评估||empty|1.0/1.0|1.0/1.0|1.0|1.0|
|16|大模型幻觉产生的原因和治理手段|大模型幻觉 产生原因 治理手段 大语言模型 LLM hallucination 成因 缓解策略 事实一致性 RAG 检索增强生成 RLHF 强化学习人类反馈 解码策略|success|1.0/1.0|1.0/1.0|0.5|0.5|
|17|微调（Fine-tuning）和 RAG 有什么区别，怎么选|微调 Fine-tuning 与 RAG（Retrieval-Augmented Generation）的区别对比、技术选型、适用场景与优缺点|success|1.0/1.0|1.0/1.0|1.0|1.0|
|18|主流 Agent 框架有什么异同||empty|0.0/0.0|0.0/0.0|0.0|0.0|
|19|工作流（Workflow）和自主 Agent 有什么区别|工作流（Workflow）与自主 Agent（Autonomous Agent/智能体）的区别、对比、架构差异及设计特点|success|1.0/1.0|1.0/1.0|0.5|0.5|
|20|GraphRAG 是什么，和普通 RAG 有什么不同|GraphRAG（Graph Retrieval-Augmented Generation）是什么，与普通 RAG（Retrieval-Augmented Generation）有什么不同，对比架构差异、技术特点与适用场景|success|1.0/1.0|1.0/1.0|1.0|1.0|

任一核心逐题指标下降题目：[]

## 改写性能

```json
{
  "success_rate": 0.25,
  "empty_rate": 0.2,
  "fallback_rate": 0.55,
  "mean_generation_latency_ms": 2821.1141349998798,
  "live_mean_latency_ms": 2821.1141349998798,
  "llm_tokens": {
    "input": 622,
    "output": 2112
  },
  "tokens_note": "completed calls only; timed-out or retried usage may be unknown"
}
```

完整候选、document_id、状态、耗时、token、逐题指标见 per_question.jsonl；
模型、prompt/参数、快照指纹、原始验收结果见 summary.json。

限于当前 20 题与固定索引；不更新简历数据。
