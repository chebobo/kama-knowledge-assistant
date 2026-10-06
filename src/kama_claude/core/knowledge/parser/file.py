from __future__ import annotations

from pathlib import Path

from kama_claude.core.knowledge.parser.base import ParsedDocument


class FileParser:
    # 初始化支持的文本文件扩展名
    def __init__(self) -> None:
        self._markdown_suffixes = {".md", ".markdown"}
        self._code_suffixes = {
            ".c", ".cc", ".cpp", ".cxx", ".cs", ".go", ".java", ".js", ".jsx",
            ".php", ".py", ".rb", ".rs", ".sh", ".sql", ".ts", ".tsx", ".vue",
        }

    # 按扩展名读取 TXT、Markdown 或代码文件
    def parse(self, path: Path) -> ParsedDocument:
        resolved = path.expanduser().resolve()
        content = resolved.read_text(encoding="utf-8", errors="replace")
        suffix = resolved.suffix.lower()
        if suffix in self._markdown_suffixes:
            mime_type = "text/markdown"
        elif suffix in self._code_suffixes:
            mime_type = "text/x-code"
        else:
            mime_type = "text/plain"
        return ParsedDocument(
            content=content,
            title=resolved.stem,
            mime_type=mime_type,
            metadata={"suffix": suffix, "path": str(resolved)},
        )

