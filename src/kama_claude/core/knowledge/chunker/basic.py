from __future__ import annotations

import hashlib
import re

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeDocument
from kama_claude.core.knowledge.parser.base import ParsedDocument


class BasicChunker:
    # 初始化按字符长度和重叠长度切分文本的 chunker
    def __init__(self, chunk_size: int = 1600, overlap: int = 200) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller")
        self._chunk_size = chunk_size
        self._overlap = overlap

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
                if not part.strip():
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

    # 按 Markdown 标题保留 section 名称，其他文件使用文档标题
    def _sections(self, text: str, mime_type: str) -> list[tuple[str, str]]:
        if mime_type != "text/markdown":
            return [("", text)]
        sections: list[tuple[str, str]] = []
        current_title = ""
        current: list[str] = []
        for line in text.splitlines():
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
        return sections or [("", text)]

    # 按固定字符窗口切分并保留相邻窗口重叠内容
    def _split(self, text: str) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= self._chunk_size:
            return [normalized]
        parts: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self._chunk_size, len(normalized))
            parts.append(normalized[start:end])
            if end == len(normalized):
                break
            start = end - self._overlap
        return parts

    # 根据文档 ID、序号和内容生成稳定 chunk ID
    @staticmethod
    def _chunk_id(document_id: str, index: int, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:{index}:{digest}"
