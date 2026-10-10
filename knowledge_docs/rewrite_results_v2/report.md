# Query Rewrite 第一阶段评测报告

运行时间（UTC）：2026-10-10T07:27:28.133002+00:00

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
|B|single@5|0.95|0.95|0.95|0.8391666666666667|0.8658764682653967|
|B|multi@5|1.0|0.8916666666666666|0.8148148148148148|0.9266666666666667|0.8580165522597886|
|B|single@8|1.0|1.0|1.0|0.8474999999999999|0.8836868276207979|
|B|multi@8|1.0|0.9333333333333333|0.8888888888888888|0.9266666666666667|0.8760458951889623|

## 验收

A/B 状态：completed；全部通过：False

- D1_zero_llm_calls: True
- D1_equals_A: True
- D2_fallback_status_query_results: True
- Q18_success_top5: True
- single_hit_at_least_095: True
- preserve_baseline_hits: False
- single_mrr_non_decreasing: False
- single_ndcg_non_decreasing: False
- multi_each_question_non_decreasing: False
- multi_micro_recall_non_decreasing: True
- fixed_rewrite_retrieval_stable: True

## Q18 候选池及结果

- A: status=disabled; effective_query=主流 Agent 框架有什么异同
  - 目标文档进入候选池：{'dense': False, 'bm25': False}
  - top-5：['agent_vs_workflow.md', 'rag_interview.md', 'chapter1.md', 'agent_skill_interview.md', 'agent_interview.md']
- B: status=success; effective_query=Agent框架 架构设计 工具调用 记忆机制 上下文管理 安全机制 适用场景 异同对比
  - 目标文档进入候选池：{'dense': True, 'bm25': True}
  - top-5：['agent_framework_comparison.md', 'claude_code_deep_dive.md', 'chapter3.md', 'production_agent_architecture_harness_org_talent.md', 'chapter10.md']

## 逐题 A/B 对比

|Q|原问题|改写|状态|A Hit/MRR|B Hit/MRR|A multi Recall|B multi Recall|
|---|---|---|---|---|---|---|---|
|1|为什么单模型走不通？Agent 混合路由优化是什么|单模型 局限性 原因 Agent 混合路由优化 定义 路由策略 架构设计 适用场景 优劣对比|success|1.0/1.0|1.0/1.0|1.0|1.0|
|2|ReAct 循环是如何工作的，LLM、上下文和工具如何协同|ReAct 循环 工作机制 LLM 上下文管理 工具调用 协同|success|1.0/0.5|1.0/0.2|0.5|0.5|
|3|RAG 检索增强生成的核心思路是什么|RAG 检索增强生成 核心思路 架构设计 检索机制 生成流程|success|1.0/0.25|1.0/0.25|1.0|1.0|
|4|Agent 的记忆系统分为哪几类|Agent 记忆系统 分类 记忆机制 类别|success|1.0/1.0|1.0/1.0|1.0|1.0|
|5|Transformer 的多头自注意力机制是如何计算的|Transformer 多头自注意力机制 计算流程 计算公式 计算步骤|success|1.0/1.0|1.0/0.3333333333333333|1.0|1.0|
|6|什么是模型蒸馏？它到底在蒸馏什么|模型蒸馏 概念 蒸馏对象 知识迁移机制|success|1.0/1.0|1.0/1.0|1.0|1.0|
|7|Claude Code 是怎么管理上下文窗口的（Auto-Compact）|Claude Code 上下文窗口管理 Auto-Compact 机制|success|1.0/1.0|1.0/1.0|1.0|1.0|
|8|Function Calling 的原理是什么，大模型如何调用工具|Function Calling 原理 大模型 工具调用 机制|success|1.0/1.0|1.0/1.0|1.0|1.0|
|9|多 Agent 之间如何通信协作|多 Agent 通信协作 消息传递 任务协调 上下文管理 架构设计|success|1.0/0.5|0.0/0.0|0.6666666666666666|0.3333333333333333|
|10|Agent 的工具应该怎么设计|Agent 工具设计 架构设计 工具调用 参数定义 接口规范 错误处理 安全机制|success|1.0/1.0|1.0/1.0|1.0|1.0|
|11|MCP 协议是什么，解决什么问题|MCP协议 定义 解决什么问题 设计目标 应用场景|success|1.0/1.0|1.0/1.0|1.0|1.0|
|12|向量数据库的核心概念和选型考虑||fallback|1.0/1.0|1.0/1.0|1.0|1.0|
|13|RAG 系统应该如何评估|RAG 系统 评估方法 评估指标 检索质量 生成质量 忠实度 相关性|success|1.0/1.0|1.0/1.0|1.0|1.0|
|14|什么是上下文工程（Context Engineering）|上下文工程 Context Engineering 概念 定义 核心思想|success|1.0/1.0|1.0/1.0|1.0|1.0|
|15|Agent 的性能应该如何评估|Agent 性能评估 评估指标 评估方法 任务成功率 响应质量 工具调用效率|success|1.0/1.0|1.0/1.0|1.0|1.0|
|16|大模型幻觉产生的原因和治理手段|大模型幻觉 成因 治理手段 缓解策略|success|1.0/1.0|1.0/1.0|0.5|0.5|
|17|微调（Fine-tuning）和 RAG 有什么区别，怎么选|微调 Fine-tuning RAG 区别 优劣对比 适用场景 选择依据|success|1.0/1.0|1.0/1.0|1.0|1.0|
|18|主流 Agent 框架有什么异同|Agent框架 架构设计 工具调用 记忆机制 上下文管理 安全机制 适用场景 异同对比|success|0.0/0.0|1.0/1.0|0.0|0.5|
|19|工作流（Workflow）和自主 Agent 有什么区别|工作流 Workflow 自主 Agent 区别 架构设计 工具调用 上下文管理 适用场景 优劣对比|success|1.0/1.0|1.0/1.0|0.5|1.0|
|20|GraphRAG 是什么，和普通 RAG 有什么不同|GraphRAG 概念 架构设计 记忆机制 上下文管理 与普通 RAG 区别 优劣对比 适用场景|success|1.0/1.0|1.0/1.0|1.0|1.0|

任一核心逐题指标下降题目：[2, 5, 9]

## 改写性能

```json
{
  "success_rate": 0.95,
  "empty_rate": 0.0,
  "fallback_rate": 0.05,
  "mean_generation_latency_ms": 7199.801279999701,
  "live_mean_latency_ms": 7199.801279999701,
  "llm_tokens": {
    "input": 1336,
    "output": 4628
  },
  "tokens_note": "completed calls only; timed-out or retried usage may be unknown"
}
```

完整候选、document_id、状态、耗时、token、逐题指标见 per_question.jsonl；
模型、prompt/参数、快照指纹、原始验收结果见 summary.json。

限于当前 20 题与固定索引；不更新简历数据。
