# Document Aggregation 独立评估

> 评测日期：2026-10-07
>
> 目标：验证 RRF 后按 `document_id` 聚合能否缓解 Q9 的大文档 chunk 霸榜，并以同一 Golden Set 的整体 Recall/MRR/NDCG 判断后续方向。
>
> 这份结果先作为独立实验；`sum` 已实现为可选生产策略但默认关闭，不修改 `eval_rag.py` 的 dense/hybrid 基线，也不把文档去重本身表述为 Recall 提升。

## 实验条件

- 数据集：当前 `knowledge_docs` 187 篇文档、V2 索引 2,989 chunks、同一 `golden_set.json` 20 问。
- Retriever：Hybrid（Dense + BM25 trigram），`rrf_k=60`。
- 候选池：每路取 `2 * k`，与 Hybrid baseline 的 `candidate_multiplier=2` 相同；先保留完整 RRF 候选，再按 `chunk.document_id` 聚合，最后取 document top-k。
- 对照组：`chunk-top-k` 使用同一份完整 RRF 结果直接截取前 k 个 chunk，不做文档聚合。
- 聚合函数：`sum`（文档内 RRF 分数求和）、`max`（文档内最大 RRF 分数）、`first-seen`（文档首次出现在 RRF 序列中的 chunk 分数）。分数相同按 RRF 首次出现顺序稳定排序。
- 指标：single 为主口径，multi 为辅助口径；Recall 是相关文档覆盖率，MRR/NDCG 使用聚合后的 document rank。

复现命令（顺序执行，避免 Qdrant Local 并发锁）：

```powershell
uv run python knowledge_docs/eval_rag.py 5 dense
uv run python knowledge_docs/eval_rag.py 8 dense
uv run python knowledge_docs/eval_rag.py 5 hybrid
uv run python knowledge_docs/eval_rag.py 8 hybrid
uv run python knowledge_docs/eval_rag_aggregation.py 5 hybrid
uv run python knowledge_docs/eval_rag_aggregation.py 8 hybrid
```

脚本输出中的 `chunk-top-k` 是固定候选池/输出预算的控制组；它应与 `eval_rag.py` 的 Hybrid baseline 一致。

生产侧启用方式：

```toml
[knowledge]
hybrid_enabled = true
document_aggregation = "sum"
```

或设置环境变量 `KAMA_KNOWLEDGE_DOCUMENT_AGGREGATION=sum`。省略该配置、设置为空值、`none` 或 `null` 时保持原始 chunk top-k 行为。

## 整体结果

single 主指标：

| k | 方法 | Hit Rate | Recall | MRR | NDCG |
|---:|---|---:|---:|---:|---:|
| 5 | `chunk-top-k` 控制组 / Hybrid baseline | 0.850 | 0.850 | 0.738 | 0.766 |
| 5 | Aggregation / sum | 0.900 | 0.900 | 0.842 | 0.857 |
| 5 | Aggregation / max | 0.850 | 0.850 | 0.738 | 0.766 |
| 5 | Aggregation / first-seen | 0.850 | 0.850 | 0.738 | 0.766 |
| 8 | `chunk-top-k` 控制组 / Hybrid baseline | 0.900 | 0.900 | 0.765 | 0.797 |
| 8 | Aggregation / sum | 0.950 | 0.950 | 0.863 | 0.885 |
| 8 | Aggregation / max | 0.900 | 0.900 | 0.765 | 0.798 |
| 8 | Aggregation / first-seen | 0.900 | 0.900 | 0.765 | 0.798 |

multi 辅助指标：

| k | 方法 | Hit Rate | Recall | MRR | NDCG |
|---:|---|---:|---:|---:|---:|
| 5 | `chunk-top-k` 控制组 / Hybrid baseline | 0.900 | 0.792 | 0.812 | 0.761 |
| 5 | Aggregation / sum | 0.950 | 0.842 | 0.925 | 0.850 |
| 5 | Aggregation / max | 0.900 | 0.792 | 0.812 | 0.761 |
| 5 | Aggregation / first-seen | 0.900 | 0.792 | 0.812 | 0.761 |
| 8 | `chunk-top-k` 控制组 / Hybrid baseline | 0.950 | 0.842 | 0.823 | 0.772 |
| 8 | Aggregation / sum | 0.950 | 0.858 | 0.925 | 0.863 |
| 8 | Aggregation / max | 0.950 | 0.867 | 0.824 | 0.783 |
| 8 | Aggregation / first-seen | 0.950 | 0.867 | 0.824 | 0.783 |

## Q9 观察

Q9 的 Hybrid baseline 在 k=5/k=8 中分别被 `chapter10.md` 占 4/5、6/8 个 chunk 席位。聚合后每个 document 最多占一个席位，因此三种方法都把 `chapter10.md` 压到单席，文档频率偏置得到结构性缓解。

| k | 方法 | Q9 document top-k |
|---:|---|---|
| 5 | sum | `chapter10.md`, `reference-answers.md`, `chapter6.md`, `20260506bytedance.md`, `application_development.md` |
| 5 | max | `chapter10.md`, `20260506bytedance.md`, `application_development.md`, `chapter6.md`, `reference-answers.md` |
| 5 | first-seen | `chapter10.md`, `20260506bytedance.md`, `application_development.md`, `chapter6.md`, `reference-answers.md` |
| 8 | sum | `chapter10.md`, `multi_agent_communication_interview.md`, `chapter6.md`, `reference-answers.md`, `chapter2.md`, `20260506bytedance.md`, `application_development.md`, `multi_agent_context_governance.md` |
| 8 | max | `chapter10.md`, `20260506bytedance.md`, `application_development.md`, `chapter6.md`, `reference-answers.md`, `multi_agent_context_governance.md`, `chapter4.md`, `chapter2.md` |
| 8 | first-seen | `chapter10.md`, `20260506bytedance.md`, `application_development.md`, `chapter6.md`, `reference-answers.md`, `multi_agent_context_governance.md`, `chapter4.md`, `chapter2.md` |

`sum` 在 k=8 将 Q9 的 expected document 排到第 2；k=5 仍未进 top-5。这个现象说明 sum 值得单独考虑，但不能把“重复 chunk 被合并”直接等同于 Recall 提升：聚合实验保留了每路 `2*k` 候选并改变了最终截断位置，指标变化反映的是完整实验链路的结果。

## 决策

- Q9 霸榜缓解：**成立**。三种函数都将同一文档的多席占位变为单席。
- 整体指标：`sum` 在 k=5/k=8 的 single Recall、MRR、NDCG 都高于 Hybrid baseline，且 Q9 在 k=8 命中 expected document；`max` 与 `first-seen` 的 single 指标基本等于 baseline，但在 k=8 的 multi Recall 达到 `0.867`（baseline `0.842`），对应的 MRR/NDCG 仍明显低于 `sum`。
- 后续方向：**`sum` 已接入生产 Retriever 但默认关闭，保留为可控 rollout 选项；不启用 `max`/`first-seen`。** Query Rewrite/Reranker 继续后置，后续 rollout 仍应观察整体 Recall/MRR/NDCG。

固定候选池确认：脚本的 `chunk-top-k` 控制组与 `eval_rag.py` Hybrid baseline 使用同一 `2*k` 每路候选和同一 RRF 截断预算；只有控制组与 `sum` 之间的最终 document 截断规则不同。k=5/k=8 控制组均与 baseline 完全一致，因此 `sum` 的差异可以归因于文档级排序。
