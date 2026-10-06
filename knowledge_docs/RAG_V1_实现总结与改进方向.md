# KamaClaude 个人知识助手 RAG V1 — 实现总结与改进方向

> 状态：**V1 基本实现完成** · 187 篇知识文档全量入库 · 检索链路可用 · 已建立可复现的数值评估基线
> 更新：2026-10-06

---

## 一、基本实现（已完成）

| 能力 | 状态 | 说明 |
|---|---|---|
| 向量存储 | ✅ | Qdrant Local（嵌入式，不部署服务），collection `kama_knowledge`，512 维 cosine |
| Embedding | ✅ | 本地模型 bge-small-zh-v1.5，经 EmbeddingProvider 抽象，不依赖外部 API |
| 文档档案 | ✅ | SQLite DocumentCatalog（document_id 主键、status 状态机、content_hash 变更检测）|
| 入库/更新 | ✅ | index / delete / reindex，content_hash 相同自动跳过，断点续传 |
| 切分 | ✅ | BasicChunker：按 Markdown 标题分 section + 1600 字符窗口 / 200 重叠 |
| 检索服务 | ✅ | KnowledgeService → Retriever → VectorStore，全部 Qdrant 层 metadata filter 隔离 |
| 作用域隔离 | ✅ | Global + 当前 Project，跨 Project 物理不可见 |
| Agent 接入 | ✅ | Root Agent 自动检索注入 `retrieved_context`；Subagent 不自动检索 |
| 检索工具 | ✅ | `knowledge_search`（Root/Subagent 可用，继承 user/project scope，默认 ALLOW）|
| 上下文预算 | ✅ | top_k=8，max_retrieved_tokens=4000 |
| 故障降级 | ✅ | 初始化失败自动 disabled，不阻塞 daemon，Agent 照常运行 |
| 配置 | ✅ | `[knowledge]` TOML 配置（enabled / top_k / embedding / store 路径等）|

## 二、测试结果

### 2.1 入库与切分基线（187 篇全量）

| 指标 | 数值 |
|---|---|
| 文档数 | 187 篇（0 失败）|
| 总字符 | 1,581,107 |
| sections | 3,066 |
| chunks | 3,237（与切分基线一致）|
| 带 frontmatter 的文档 | 153 篇 |
| 空壳 chunk（<50 字符）| 109 个 |
| 超长 section（>1600 触发切分）| 126 个 |
| chunk 长度 | 均值 497 / 中位 395 |

入库耗时：并发优化后 **184.8s** 全量完成（content_hash 跳过 100 篇已入库 + 新增 87 篇）。

### 2.2 检索快照（10 问 × Top-5）

![10 问检索矩阵](./eval_matrix_10q.png)

| # | 问题 | Top-1 | 相关命中 | 噪音 |
|---|---|---|---|---|
| 1 | Agent 混合路由优化 | ✓ 预期文档 | 5/5 | 无 |
| 2 | ReAct 循环 | 相关文档（预期排第 5）| 5/5 | 无 |
| 3 | RAG 核心思路 | ✓ 预期文档 | 5/5 | 无 |
| 4 | 级联降级 | ✓ 预期文档 | 5/5 | 无 |
| 5 | Agent 记忆 | ✓ 预期文档 | 5/5 | 无 |
| 6 | Transformer 注意力 | ✓ 预期文档 | 5/5 | 无 |
| 7 | 模型蒸馏 | ✓ 预期文档 | 5/5 | 无 |
| 8 | Claude Code 上下文 | ✓ 预期文档 | 5/5 | **frontmatter 第 2 位** |
| 9 | function calling | ✓ 预期文档 | 5/5 | **frontmatter 第 4 位** |
| 10 | 多 Agent 通信 | ✓ 预期文档 | 5/5 | 无 |

快照结论：10/10 问题 Top-5 全相关，9/10 预期文档排第 1；发现 **2 处 frontmatter 噪音**（纯 YAML 元数据混入 Top-5，Q8 的 frontmatter score=0.802 甚至压过正文）。

### 2.3 golden 集硬指标（20 问，业界标准）

![20 问命中位次](./eval_rank_20q.png)

| 指标 | k=5 | k=8 |
|---|---|---|
| Recall@k | **0.900** (18/20) | 0.900 (18/20) |
| MRR@k | **0.709** | 0.709 |
| NDCG@k | **0.755** | 0.755 |

- 匹配口径：document 级（chunk 的 source 文件名 == golden 预期文档，同文档重复 chunk 只计首次）
- 20 问 golden 已逐篇对照原文标题结构人工验证（`golden_set.json`）

## 三、结论：基本实现可用

1. **主链路打通且质量达标**：187 篇文档全量入库 0 失败；检索 18/20 预期文档可召回，13/20 排第 1；Root Agent 自动检索 + knowledge_search 工具已接入。
2. **可复现评估体系已建立**：golden 集 + 指标脚本 + 基线报告，后续任何改动（chunker、检索策略）都能用同一套数据量化对比。
3. **已知缺陷 5 处**（V2 改良靶子）：
   - **2 个召回硬缺陷**：Q16 幻觉治理、Q18 框架对比——k=8 仍召不回，是向量检索语义盲区（"幻觉/框架"等泛词被训练类文档抢占），不是排序问题；
   - **3 个排序缺陷**：Q5 多头注意力、Q17 微调vsRAG、Q19 WorkflowvsAgent 均排第 5，能召回但位次靠后；
   - **切分噪音**：frontmatter chunk 和 109 个空壳 chunk 会稀释向量质量、挤占正文章节位置。

## 四、后续改进方向（按优先级）

| 优先级 | 改进项 | 目标问题 | 预期收益 |
|---|---|---|---|
| P1 | **Chunker 改良**：跳过 frontmatter、过滤 <50 字符空壳、可选句子/段落边界切分 | 噪音 chunk、半截句子 | 向量质量整体提升，正文位置前移 |
| P1 | **重跑同 golden 对比**：改良后 `index_knowledge.py`（content_hash 智能重建）→ `eval_rag.py` | 全部 20 问 | 量化 Recall/MRR/NDCG 提升幅度 |
| P2 | **Hybrid Search**（BM25 关键词 + 向量，RRF 合并）| Q16 / Q18 泛词盲区 | 关键词精确命中"幻觉治理/框架对比" |
| P2 | **Query Rewrite**（查询改写/扩展）| 同上 + 口语化查询 | 提升 bge 对抽象问题的匹配度 |
| P3 | **Reranker**（检索后精排）| Q5/Q17/Q19 位次靠后 | NDCG 提升 |
| P3 | **更细 section 级评估** | 评估粒度 | 定位到具体 section 而非整篇 |

**V2 明确不做**（保持边界）：RPC / 独立 Knowledge Server、OCR、PDF/DOCX parser、外部 Embedding API、Subagent 自动 RAG、Hybrid/Reranker 的完整工程化（仅上述优先级内渐进）。

## 附：复跑命令与文件清单

```powershell
# 全量入库（content_hash 断点续传）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\index_knowledge.py
# 检索快照（10 问）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\search_baseline.py
# golden 指标评估（k=5 / k=8）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8
```

| 文件 | 用途 |
|---|---|
| `golden_set.json` | 20 问 ground truth（预期文档 + 预期 section）|
| `eval_rag.py` | 指标化评估（Recall/MRR/NDCG）|
| `search_baseline.py` / `eval_baseline.md` | 10 问快照 + V1 指标报告 |
| `chunk_baseline.py` / `chunk_baseline.md` / `.csv` | 切分基线 |
| `index_knowledge.py` | 全量入库脚本 |
| `eval_matrix_10q.svg` / `eval_rank_20q.svg` | 本报告配图 |

运行时数据：`~/.kama/knowledge/`（qdrant 3237 向量 + catalog.db 187 条档案）
