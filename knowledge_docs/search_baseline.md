# 检索基线报告（V1 真实 RAG 链路）

> 生成时间：2026-10-06
> 链路：`QdrantVectorStore.search` + `bge-small-zh-v1.5`（真实 Qdrant Local，187 篇文档 / 3237 chunks 已入库）
> 检索参数：top_k=5，filter user_id="local"（global scope）
> 用途：改良 chunker 后同问题集重跑，对比检索质量。

## 总体结论

| 指标 | 结果 |
|---|---|
| 问题数 | 10 |
| Top-5 全相关的题数 | 10 / 10 |
| 预期文档进入 Top-5 | 10 / 10 |
| 预期文档排 Top-1 | 9 / 10 |
| Top-5 内 frontmatter 噪音 | 2 处（Q8 第 2 位、Q9 第 4 位）|

**V1 检索质量良好**：bge + Qdrant 的语义检索对这批知识库素材命中率很高。主要缺陷是 frontmatter 元数据 chunk 混入 Top-5（Q8 的 frontmatter score=0.802 甚至排第 2，挤掉了正文）。

## 逐题明细

| # | 问题 | 预期文档 | Top-1 | 相关命中 | 噪音 |
|---|---|---|---|---|---|
| 1 | Agent 混合路由优化是什么 | hybrid_routing | ✓ hybrid_routing | 5/5 | 无 |
| 2 | ReAct 循环是怎么工作的 | chapter1 | ✗ loop_engineering（ReAct 相关） | 5/5 | 无 |
| 3 | RAG 检索增强生成的核心思路 | rag | ✓ rag_interview | 5/5 | 无 |
| 4 | 级联降级怎么做 | hybrid_routing | ✓ hybrid_routing | 5/5 | 无 |
| 5 | Agent 记忆系统类型 | memory | ✓ agent_memory | 5/5 | 无 |
| 6 | Transformer 注意力如何计算 | attention | ✓ transformer_interview | 5/5 | 无 |
| 7 | 什么是模型蒸馏 | distillation | ✓ model_distillation | 5/5 | 无 |
| 8 | Claude Code 上下文窗口管理 | context | ✓ claude_code_context | 5/5 | **frontmatter 第 2 位** |
| 9 | function calling 是什么 | function_calling | ✓ agent_interview | 5/5 | **frontmatter 第 4 位** |
| 10 | 多 Agent 如何通信 | multi_agent | ✓ multi_agent_communication | 5/5 | 无 |

## 观察到的 V1 缺陷（V2 对比目标）

1. **frontmatter 抢位**：153 篇带 frontmatter 的文档，其 YAML（title/description 含大量关键词）embedding 后相似度很高，会挤占正文章节。Q8 中 frontmatter score=0.802 > 正文第 3 位 score=0.790。V2 跳过 frontmatter 后，这些位置应让给正文。
2. **空壳 section 影响有限**：109 个 <50 字符 chunk 未出现在本次 Top-5 中（分低），但会稀释整体检索质量。
3. **半截句子未见明显影响**：本次命中内容开头语义完整（多为标题+正文起始），固定窗口切断的 chunk 未显著伤害相关性——但这是"标题恰好对齐"的运气，长文档中仍可能踩中。

## 复跑命令

```powershell
# 检索基线（问题集在 search_baseline.py 中可编辑）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\search_baseline.py
```
