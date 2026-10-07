from __future__ import annotations

import hashlib
import re

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeDocument
from kama_claude.core.knowledge.parser.base import ParsedDocument

# 过滤低于该字符数的空壳 chunk（无检索价值）
MIN_CHUNK_CHARS = 50
# 切分时允许回退找边界的下限（至少保留一半窗口内容）
_BOUNDARY_LOW_RATIO = 0.5


class BasicChunker:
    # 初始化按字符长度和重叠长度切分文本的 chunker（V2：跳 frontmatter、滤空壳、段落边界感知）
    def __init__(self, chunk_size: int = 1600, overlap: int = 200) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller")
        self._chunk_size = chunk_size
        self._overlap = overlap

    # 切分策略签名：任何参数或逻辑变化都会让签名变化，从而触发重新索引
    @property
    def signature(self) -> str:
        return (
            f"BasicChunker:v2:{self._chunk_size}:{self._overlap}:"
            f"skip_frontmatter:min_chunk={MIN_CHUNK_CHARS}"
        )

    # 将解析文本切分成带来源和 section 元数据的知识片段
    def chunk(
        self,
        document: KnowledgeDocument,
        parsed: ParsedDocument,
    ) -> list[KnowledgeChunk]:
        sections = self._sections(parsed.content, parsed.mime_type)
        chunks: list[KnowledgeChunk] = []
        index = 0
        for section, content in sections:
            for part in self._split(content):
                if len(part.strip()) < MIN_CHUNK_CHARS:
                    continue
                chunk_id = self._chunk_id(document.id, index, part)
                chunks.append(
                    KnowledgeChunk(
                        id=chunk_id,
                        document_id=document.id,
                        content=part,
                        source=document.source,
                        title=document.title,
                        scope=document.scope,
                        project_id=document.project_id,
                        user_id=document.user_id,
                        metadata={"section": section, "chunk_index": index},
                    )
                )
                index += 1
        return chunks

    # 按 Markdown 标题保留 section 名称，其他文件使用文档标题；跳过 YAML frontmatter
    def _sections(self, text: str, mime_type: str) -> list[tuple[str, str]]:
        if mime_type != "text/markdown":
            return [("", text)]
        body = self._strip_frontmatter(text)
        sections: list[tuple[str, str]] = []
        current_title = ""
        current: list[str] = []
        for line in body.splitlines():
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if match:
                if current:
                    sections.append((current_title, "\n".join(current).strip()))
                current_title = match.group(1)
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append((current_title, "\n".join(current).strip()))
        return sections or [("", body)]

    # 移除文档开头的 YAML frontmatter（--- 到 --- 之间的元数据不进入正文）
    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        if not text.startswith("---"):
            return text
        lines = text.splitlines()
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1 :])
        return text

    # 按固定字符窗口切分，窗口结尾优先在段落边界（空行/换行）处断开
    def _split(self, text: str) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= self._chunk_size:
            return [normalized]
        parts: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self._chunk_size, len(normalized))
            if end < len(normalized):
                cut = self._find_boundary(normalized, start, end)
                if cut is not None:
                    end = cut
            parts.append(normalized[start:end])
            if end == len(normalized):
                break
            start = end - self._overlap
        return parts

    # 在窗口后半段找最近的段落边界：优先空行（\n\n），其次单换行（\n）
    @staticmethod
    def _find_boundary(text: str, start: int, end: int) -> int | None:
        window = text[start:end]
        low = max(int(len(window) * _BOUNDARY_LOW_RATIO), 1)
        idx = window.rfind("\n\n", low)
        if idx != -1:
            return start + idx + 2
        idx = window.rfind("\n", low)
        if idx != -1:
            return start + idx + 1
        return None

    # 根据文档 ID、序号和内容生成稳定 chunk ID
    @staticmethod
    def _chunk_id(document_id: str, index: int, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:{index}:{digest}"
