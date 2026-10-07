from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

KnowledgeScope = Literal["global", "project"]


@dataclass
class KnowledgeDocument:
    # 保存一个可索引知识文档的来源和索引状态
    id: str
    source: str
    title: str
    scope: KnowledgeScope = "global"
    project_id: str | None = None
    user_id: str | None = None
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    source_mtime: float | None = None
    indexed_at: datetime | None = None
    status: str = "pending"
    chunk_count: int = 0
    error_message: str | None = None


@dataclass
class KnowledgeChunk:
    # 保存一个可嵌入和检索的知识片段及其来源元数据
    id: str
    document_id: str
    content: str
    source: str
    title: str
    scope: KnowledgeScope = "global"
    project_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeSearchResult:
    # 表示一次检索命中的片段和相似度分数
    chunk: KnowledgeChunk
    score: float
    retriever: str | None = None
    rank: int | None = None


@dataclass(frozen=True)
class KnowledgeFilter:
    # 描述知识检索允许访问的用户和项目范围
    user_id: str | None = None
    project_id: str | None = None


@dataclass
class KnowledgeIndexResult:
    # 返回一次文档索引的摘要结果
    document_id: str
    chunk_count: int
    skipped: bool = False
