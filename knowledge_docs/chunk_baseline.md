# Chunk 基线数据（V1 → V2 对比）

> 生成时间：V1 2026-10-06 / V2 2026-10-07
> 口径：`FileParser` + `BasicChunker(chunk_size=1600, overlap=200)`，范围 `knowledge_docs/` 全部 `.md`（187 篇，排除报告类 md）
> 明细：见同目录 `chunk_baseline.csv`（每篇的 chars / chunks / full_window_chunks / frontmatter）
> 用途：chunker V2 改良（跳 frontmatter / 滤空壳 / 段落边界切分）后的质量对比。

## 全局汇总（V1 vs V2）

| 指标 | V1 | V2 | 变化 |
|---|---|---|---|
| 文档数 | 187 | 187 | — |
| 总字符数 | 1,581,107 | 1,581,107 | — |
| 总 chunk 数 | 3,237 | 2,989 | **-248（-7.7%）** |
| 平均每篇 chunk | 17.3 | 16.0 | -1.3 |
| 空壳 chunk（<50 字符） | 109（3.4%） | **0** | 全部过滤 |
| 带 frontmatter 的文档 | 153（81.8%） | 153（已跳过） | 噪音 chunk 消失 |
| 满窗口 chunk（=1600） | 172（5.3%） | **1** | 段落边界切分生效 |
| chunk 长度均值 / 中位数 | 497 / 395 | 514 / 412 | 更均匀 |
| 最短 / 最长 chunk | 7 / 1600 | 51 / 1600 | 无 <50 空壳 |

## V1 → V2 三项改良

1. **跳过 frontmatter**：`_strip_frontmatter` 移除文档开头 YAML（--- 到 ---），元数据不再作为正文 chunk 参与检索；标题/描述等检索信号改由 Parser 提取、随 embedding 前缀注入（见下）。
2. **过滤空壳**：`chunk()` 跳过 `len(part.strip()) < 50` 的片段，109 个低质 chunk 清零，最短 chunk 从 7 → 51。
3. **段落边界切分**：`_split` 窗口末尾优先在 `\n\n`（其次 `\n`）断开，回退下限 50% 窗口；满窗口硬切块从 172 → 1，几乎消灭句子中间切断。

## 配套改动（V2 检索质量）

- **Parser 提取 frontmatter 元数据**：`FileParser` 解析 `title/description/keywords/tags`，`parsed.title` 优先用 frontmatter 中文标题。
- **标题/描述注入**：`KnowledgeService._embed_text` 将 `title + section + description + keywords + 正文` 拼为 embedding 输入，保留文档级检索信号。
- **content_hash 版本因子**：hash 拼入 `chunker.signature + embed 策略 + title`，任一索引输入变化自动触发全量重建（增量控制正确：仅受影响文档重建）。

## 复跑命令

```powershell
# 生成 CSV 明细 + 汇总（脚本已迁移到项目内）
& D:\tinyclaude\.venv\Scripts\python.exe D:\tinyclaude\knowledge_docs\chunk_baseline.py
```
