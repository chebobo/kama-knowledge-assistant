# Chunk 基线数据（V1 BasicChunker）

> 生成时间：2026-10-06
> 口径：`FileParser` + `BasicChunker(chunk_size=1600, overlap=200)`，范围 `knowledge_docs/` 全部 `.md`（187 篇）
> 明细：见同目录 `chunk_baseline.csv`（每篇的 chars / sections / chunks / over_limit_sections / empty_chunks / frontmatter）
> 用途：改良 chunker 后按同口径重跑，对比切分质量指标。

## 全局汇总

| 指标 | 值 |
|---|---|
| 文档数 | 187 |
| 总字符数 | 1,581,107 |
| 总 section 数 | 3,066 |
| 总 chunk 数 | 3,237 |
| 平均每篇 chunk | 17.3 |
| 超长 section 数（>1600，触发切分） | 126 |
| 空壳 chunk（<50 字符） | 109（占 3.4%） |
| 带 frontmatter 的文档 | 153（占 81.8%） |
| chunk 长度均值 / 中位数 | 497 / 395 |
| 最短 / 最长 chunk | 7 / 1600 |

## chunk 长度分布

| 区间 | 数量 | 占比 |
|---|---|---|
| 0-99 | 322 | 9.9% |
| 100-199 | 488 | 15.1% |
| 200-399 | 825 | 25.5% |
| 400-799 | 1051 | 32.5% |
| 800-1199 | 272 | 8.4% |
| 1200-1599 | 107 | 3.3% |
| =1600（满窗口） | 172 | 5.3% |

## V1 已量化的待改良点（V2 对比目标）

1. **frontmatter 噪音**：153/187 篇的 YAML frontmatter 会成为 `chunk[0]`（纯元数据），参与相似度检索。V2 预期：跳过 frontmatter → 至少减少 153 个噪音 chunk。
2. **空壳 chunk**：109 个 <50 字符的 chunk（多为"标题行 + 空行"），embedding 意义低。V2 预期：过滤 <50 字符的空壳 section → 减少约 109 个低质 chunk。
3. **半截句子**：固定 1600 字符窗口不认句子边界，`=1600` 满窗口 172 块的开头可能从句子中间切断。V2 预期：按句子/段落边界切分后，"满窗口"数量下降、检索语义完整性提升。
4. **平均 chunk 偏短（497）**：大部分 section 天然 <1600，说明"按标题切"主导了粒度；V2 若合并过短 section（如相邻 <200 合并），可降低 chunk 总数、提升每块信息密度。

## 复跑命令

```powershell
# 生成 CSV 明细 + 汇总（脚本：C:\Users\13683\AppData\Local\Temp\chunk_baseline.py）
& D:\tinyclaude\.venv\Scripts\python.exe C:\Users\13683\AppData\Local\Temp\chunk_baseline.py
```
