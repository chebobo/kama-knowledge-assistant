# KamaClaude 个人知识助手 RAG V2 — Hybrid Search 实现总结与改进方向

> 状态：**RAG V2 Hybrid Search 实现完成 + 受控评测通过** · Dense + BM25 双路检索 + RRF 融合 · 187 篇知识文档双索引同步 · 可复现数值对比已建立
> 更新：2026-10-07（Hybrid 实测与结论已并入）
> 接续：`RAG_V1_实现总结与改进方向.md`（V1 主链路 + V2 chunker/检索信号改良）

---

## 一、基本实现（已完成）

| 能力 | 状态 | 说明 |
|---|---|---|
| DenseRetriever | ✅ | 包装 Qdrant 向量检索（bge-small-zh-v1.5，512 维 cosine），结果带 `retriever="dense"`、`rank` |
| BM25Retriever | ✅ | 基于 SQLite FTS5 的 lexical 检索，结果带 `retriever="bm25"`、`rank` |
| 词法索引 | ✅ | `SqliteLexicalStore`：FTS5 虚表 + **trigram tokenizer**（中文三字切分，解决默认分词把整句当 1 token 的问题）；UNINDEXED 列承载 scope/project/user 过滤 |
| trigram OR 查询 | ✅ | 从 query 提取 3 字 trigram 用 OR 匹配，允许命中文档关键子串而非整句精确短语（修复"真实问题不与文档原文完全一致导致候选为空"）|
| HybridRetriever | ✅ | 双路并行检索（`asyncio.gather`），单路故障降级保留另一路；候选 = 2 × top_k，避免单路 Top-K 提前丢失 |
| RRF 融合 | ✅ | `reciprocal_rank_fusion`：按排名贡献 `1/(k+rank)` 聚合（k=60），不依赖 Dense/BM25 分数数值可比性 |
| 生命周期同步 | ✅ | index/delete/reindex 同时维护 Qdrant + FTS5；Qdrant chunk 集合 = FTS5 row 集合 |
| 版本控制 | ✅ | `_INDEX_STRATEGY` 拼入 `fts5:trigram:v1`，content_hash 变化自动触发全量重建 |
| 配置 | ✅ | `hybrid_enabled` / `lexical_path` / `candidate_multiplier` / `rrf_k`，TOML + env 严格解析 |
| 故障降级 | ✅ | FTS5 初始化失败 → lexical 置空、retriever 回退 Dense，daemon 与 Agent 不受影响 |
| 评估体系 | ✅ | `eval_rag.py` 支持 `dense`/`hybrid` 双模式，同一 golden 集、同一索引受控对比 |
| 边界保持 | ✅ | AgentLoop 零修改；KnowledgeService 不直接碰 FTS/RRF；Retriever Protocol 未破坏（`KnowledgeSearchResult` 仅加可选字段）|

## 二、实现要点

1. **中文 lexical 检索的核心坑与解法**：SQLite FTS5 默认 `unicode61` 会把整段中文当作 1 个 token，BM25 形同虚设；V2 采用 `tokenize=trigram`（本机 SQLite 3.53.1 支持，运行时探测并降级）。
2. **query 侧 trigram 化**：首次受控评测 Hybrid 与 Dense 完全相同，定位根因是 FTS 把整句当作精确短语、真实问题不与文档连续原文一致 → lexical 候选几乎为空；改为从 query 提取 3 字 trigram 用 OR 匹配后，lexical 候选开始生效。
3. **双索引一致性**：`index()` 先删后写（Qdrant + FTS5），`delete()`/`reindex()` 同步；失败恢复以 `catalog.status` 为唯一事实源。
4. **元文档排除**：`RAG_V2_HYBRID_PLAN.md` / `RAG_V2_HYBRID_EXECUTION_PLAN.md` / 本总结文件均加入入库 EXCLUDE，避免开发计划文档污染知识库与评测。

## 三、测试结果

### 3.1 单元与类型检查

| 检查 | 结果 |
|---|---|
| knowledge 相关测试（retriever/bm25/fusion/hybrid/app/config/service 等）| **46 passed** |
| mypy 全仓 | **Success: no issues found in 109 source files** |
| 全量 unit suite | 307 passed / 7 failed（失败均为既有 bash timeout + Compactor 事件循环问题，**与 RAG 改动无关**）|

### 3.2 Golden 受控对比（20 问，document 级，同索引同口径）

索引版本：V2 chunker + bge-small-zh-v1.5 + FTS5 trigram v1；Qdrant 与 FTS5 均 187 文档 / 2,989 chunks。Hybrid 候选倍率 2、RRF k=60。

**k=5：**

| 口径 | Retriever | Query Hit Rate | 严格 Recall | MRR | NDCG |
|---|---|---:|---:|---:|---:|
| single（主）| Dense | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| single（主）| **Hybrid** | **0.850 (17/20)** | **0.850** | **0.738** | **0.766** |
| multi（辅助）| Dense | 0.900 (18/20) | 0.792 | 0.812 | 0.757 |
| multi（辅助）| Hybrid | 0.900 (18/20) | 0.792 | 0.812 | 0.761 |

**k=8：**

| 口径 | Retriever | Query Hit Rate | 严格 Recall | MRR | NDCG |
|---|---|---:|---:|---:|---:|
| single（主）| Dense | 0.800 (16/20) | 0.800 | 0.713 | 0.735 |
| single（主）| **Hybrid** | **0.900 (18/20)** | **0.900** | **0.765** | **0.797** |
| multi（辅助）| Dense | 0.900 (18/20) | 0.792 | 0.812 | 0.757 |
| multi（辅助）| **Hybrid** | **0.950 (19/20)** | **0.842** | **0.823** | **0.772** |

### 3.3 逐问观察（Hybrid vs Dense）

| 表现 | 题目 | 详情 |
|---|---|---|
| 救回 | Q16 幻觉治理 | Dense 双口径 MISS → Hybrid k=5 rank4 / k=8 **rank1**（trigram 命中术语）|
| 前移 | Q3 RAG 核心思路 | single MISS → k=5 rank2 / k=8 rank3（multi 保持 rank1-2）|
| 前移 | Q14 / Q19 | rank 前移至 1 |
| 代价 | Q5 多头注意力 / Q11 MCP | Dense rank1 → Hybrid rank2（lexical 引入竞争文档）|
| 代价 | Q17 微调 vs RAG | k=5 漏召回（Dense rank4 → MISS），k=8 回到 rank8 |
| 未解决 | Q9 多 Agent 通信 | 仍 MISS：`chapter10.md` 单文档霸榜 top-8 占 5 席（chunk 级检索固有偏置）|
| 未解决 | Q18 框架对比 | **双口径均 MISS**：trigram 精确匹配也救不回，属术语/关系型查询盲区 |

## 四、结论：Hybrid 有效，但收益与代价并存

1. **主收益已验证**：k=8 single Hit/Recall 0.800 → **0.900**，MRR 0.713 → 0.765，NDCG 0.735 → 0.797；multi 严格 Recall 0.792 → **0.842**（结果多样性同步改善）；Q16（幻觉治理）由双口径 MISS 救回至 k=8 rank1——**精确关键词信号确实补上了纯向量的语义盲区**。
2. **代价明确**：Q5/Q11 从 rank1 退到 rank2，Q17 在 k=5 漏召回——trigram OR 引入的竞争文档会挤占部分原本排前的向量结果，这是融合的固有 trade-off，不是实现缺陷。
3. **未解决项指向下一阶段**：
   - **Q9 单文档霸榜**：chunk 级 top-k 中大部头文档（51 chunks）天然占位，RRF 只融合排名、不解决文档级多样性 → **Document Aggregation**；
   - **Q18 双口径盲区**：Dense 与 trigram 都召不回，"框架对比"类涉及专名+关系，可能需 **Query Rewrite**（术语规范化/查询改写）而非继续调检索器。

## 五、后续改进方向（按优先级）

| 优先级 | 改进项 | 目标问题 | 预期收益 |
|---|---|---|---|
| P1 | **Document Aggregation**：chunk 检索 → 文档级聚合/去重后再 Top-K（独立评估，不混入 Hybrid 实现）| Q9 霸榜、大部头淹没 | 缓解单文档多 chunk 占位，改善结果多样性 |
| P2 | **Query Rewrite**（查询改写/术语规范化）| Q18 及同类专名/关系型查询 | 提升 bge 与 trigram 对抽象查询的匹配度 |
| P3 | **Reranker**（cross-encoder 精排）| 位次靠后题目 | NDCG 进一步提升 |
| P3 | **更细 section 级评估** | 评估粒度 | 定位到具体 section 而非整篇 |

**V2.1 明确不做**（保持边界）：Qdrant Server、外部 Embedding API、独立 BM25 服务（Elasticsearch 等）、Subagent 自动 RAG、RPC/Knowledge Server、OCR/PDF/DOCX parser。Query Rewrite / Reranker 仅在 Document Aggregation 评测完成后按需推进。

## 附：复跑命令与文件清单

```powershell
# golden 受控对比（k=5 / k=8，dense / hybrid）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5 dense
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 5 hybrid
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8 dense
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\eval_rag.py 8 hybrid
# 全量双索引入库（content_hash 断点续传，Qdrant + FTS5 同步）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\index_knowledge.py
```

| 文件 | 用途 |
|---|---|
| `src/kama_claude/core/knowledge/retriever/dense.py` | DenseRetriever（包装 Qdrant 检索）|
| `src/kama_claude/core/knowledge/retriever/bm25.py` | BM25Retriever（lexical 检索）|
| `src/kama_claude/core/knowledge/retriever/lexical_store.py` | FTS5 trigram 索引与查询（含 scope 过滤）|
| `src/kama_claude/core/knowledge/retriever/fusion.py` | RRF 融合（独立可测）|
| `src/kama_claude/core/knowledge/retriever/hybrid.py` | HybridRetriever（双路并行 + 降级）|
| `src/kama_claude/core/knowledge/service.py` | 双索引生命周期同步 + Retriever 注入 |
| `src/kama_claude/core/config.py` / `app.py` | hybrid 配置 + FTS 初始化降级 |
| `knowledge_docs/eval_rag.py` | 双模式受控评测（dense/hybrid，双口径）|
| `knowledge_docs/golden_set.json` | 20 问 ground truth（含 6 题 related_docs）|
| `knowledge_docs/eval_baseline.md` | Dense/Hybrid 受控对比基线表 |
| `knowledge_docs/RAG_V2_HYBRID_PLAN.md` / `RAG_V2_HYBRID_EXECUTION_PLAN.md` | 方案设计 + 执行交接记录（均不入库）|

运行时数据：`~/.kama/knowledge/`（qdrant 2989 向量 + catalog.db 187 条档案 + lexical.db 2989 rows）
