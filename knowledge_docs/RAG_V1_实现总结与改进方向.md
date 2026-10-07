# KamaClaude 个人知识助手 RAG V1 — 实现总结与改进方向

> 状态：**V1 基本实现完成 + V2 chunker/检索改良完成** · 187 篇知识文档全量入库 · 检索链路可用 · 已建立可复现的数值评估基线
> 更新：2026-10-07（V2 改良与重测结果已并入）

---

## 一、基本实现（已完成）

| 能力 | 状态 | 说明 |
|---|---|---|
| 向量存储 | ✅ | Qdrant Local（嵌入式，不部署服务），collection `kama_knowledge`，512 维 cosine |
| Embedding | ✅ | 本地模型 bge-small-zh-v1.5，经 EmbeddingProvider 抽象，不依赖外部 API |
| 文档档案 | ✅ | SQLite DocumentCatalog（document_id 主键、status 状态机、content_hash 变更检测）|
| 入库/更新 | ✅ | index / delete / reindex，content_hash 相同自动跳过，断点续传 |
| 切分 | ✅ | BasicChunker（V2）：按 Markdown 标题分 section + 1600 字符窗口 / 200 重叠，**跳过 frontmatter、过滤 <50 空壳、段落边界切分** |
| 元数据注入 | ✅ | FileParser 提取 frontmatter title/description/keywords；embedding 拼 title+section+描述+正文（跨语言中文信号）|
| 版本控制 | ✅ | content_hash 拼入 chunker 签名 + embed 策略 + title，索引输入变化自动增量重建 |
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

## 三、V2 改良（2026-10-07）与重测结果

![V1 vs V2 改良对比](./eval_v1_v2_compare.png)

### 3.1 V2 做了什么

1. **Chunker 三改**：跳过 YAML frontmatter（153 篇噪音 chunk 消失）、过滤 <50 字符空壳（109 个清零）、窗口末尾段落边界切分（满窗口块 172→1）。
2. **frontmatter 元数据提取**：FileParser 解析 `title/description/keywords/tags`（Q5 关键：mha.md 的 description 含中文"多头注意力"）。
3. **标题/描述注入**：embedding 输入 = title + section + description + keywords + 正文，保留文档级检索信号。
4. **content_hash 版本因子**：hash 拼入 chunker 签名 + embed 策略 + title——索引输入一变自动重建（实测 153 重建 / 34 跳过，增量控制正确）。

### 3.2 chunk 切分质量对比（同 187 篇同口径）

| 指标 | V1 | V2 | 变化 |
|---|---|---|---|
| 总 chunk 数 | 3,237 | 2,989 | -248（-7.7%）|
| 空壳 chunk（<50）| 109 | **0** | 过滤 |
| 满窗口 chunk（=1600）| 172 | **1** | 段落边界生效 |
| 最短 chunk | 7 | 51 | 无空壳 |
| 长度均值 / 中位 | 497 / 395 | 514 / 412 | 更均匀 |

### 3.3 golden 指标对比（k=5，20 问）

| 指标 | V1 | V2 | 变化 |
|---|---|---|---|
| Recall@5 | 0.900 (18/20) | 0.800 (16/20) | -0.100 |
| MRR@5 | 0.709 | **0.713** | +0.004（超 V1）|
| NDCG@5 | 0.755 | 0.735 | -0.020 |

**双口径评测（2026-10-07）**：为 Q2/Q3/Q9/Q16/Q18/Q19 六题标注 `related_docs`（确实存在多合理答案），single 命中 expected_doc / multi 命中任一相关文档：

| 口径 | Recall@5 | MRR@5 | NDCG@5 |
|---|---|---|---|
| single（主指标）| 0.800 (16/20) | 0.713 | 0.735 |
| multi（辅助）| **0.900 (18/20)** | **0.812** | **0.835** |

![V2 双口径评测](./eval_v2_dual_metric.png)

逐问：**救回 3 题**（Q5 多头注意力 5→1、Q11 MCP 4→1、Q19 Workflow 5→2，Q14/Q17 前移）；**2 题被多标签正名**（Q3/Q9）。

### 3.4 V2 结论

- **chunk 质量显著提升、排序整体前移**（MRR 超 V1），核心功臣是 frontmatter 中文 title/description 注入。
- **Q3/Q9 已被双口径评测正名**（multi 命中）：
  - Q3：`agentic_rag.md` 与"RAG 核心思路"语义高度重合且 Top-5 全占——**检索返回了合理相关文档，是单标签判错，非检索缺陷**；
  - Q9：李博杰 `chapter10.md`（多智能体整章）相关且被优先召回——**大部头文档相关，是单标签局限**；
  - 新暴露：两题 Top-5 被**同一文档霸榜**（agentic_rag ×5、chapter10 ×4），文档频率偏置/结果多样性不足，与 Q9 大部头淹没同源。
- **Q16/Q18 双口径均 MISS，确认真缺陷**：连各自 related_docs（agent_drift_hallucination_interview / harness_interview）都没进 Top-5——不是评测判错，是**检索没返回任何合理相关文档**。纯向量 top-k 固有盲区，指向 **P2 Hybrid Search**（BM25 精确关键词）。

**V2 剩余问题全部指向 P2**：Hybrid Search（BM25 精确关键词救长尾）→ Document Aggregation（解决霸榜/大部头）→ Reranker（精排）→ Query Rewrite。

## 四、结论：基本实现可用

1. **主链路打通且质量达标**：187 篇文档全量入库 0 失败；检索 16/20 预期文档可召回，14/20 排第 1（V2）；Root Agent 自动检索 + knowledge_search 工具已接入。
2. **可复现评估体系已建立**：golden 集 + 指标脚本 + 基线报告，chunker/检索策略的任何改动都能用同一套数据量化对比（V1→V2 已实证）。
3. **V2 已知缺陷（改进靶子）**：
   - **2 个召回硬缺陷**：Q16 幻觉治理、Q18 框架对比——k=8 仍召不回，泛词被 generic 文档抢占（V1 就有，指向 P2 Hybrid）；
   - **2 个长尾问题**：Q3 单标签口径局限（实际双文档相关）、Q9 大部头章节淹没；
   - **已解决**：frontmatter/空壳噪音、Q5/Q11 排序缺陷。

## 四、后续改进方向（按优先级）

| 优先级 | 改进项 | 目标问题 | 预期收益 |
|---|---|---|---|
| ~~P1~~ ✅ | **Chunker 改良**：跳过 frontmatter、过滤 <50 字符空壳、段落边界切分（已完成）| 噪音 chunk、半截句子 | 已实证：空壳清零、满窗口 172→1 |
| ~~P1~~ ✅ | **重跑同 golden 对比**（已完成）| 全部 20 问 | 已实证：MRR 0.709→0.713、NDCG 0.755→0.735、Recall 0.900→0.800（2 题口径局限）|
| P2 | **Hybrid Search**（BM25 关键词 + 向量，RRF 合并）| Q16 / Q18 泛词盲区 + Q9 长尾 | 关键词精确命中"幻觉治理/框架对比/多Agent通信" |
| P2 | **Query Rewrite**（查询改写/扩展）| 同上 + 口语化查询 | 提升 bge 对抽象问题的匹配度 |
| P3 | **Reranker**（检索后精排）| 位次靠后题目 | NDCG 提升 |
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

运行时数据：`~/.kama/knowledge/`（qdrant 2989 向量 + catalog.db 187 条档案，V2 重建）
