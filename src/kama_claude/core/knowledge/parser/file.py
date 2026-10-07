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

    # 按扩展名读取 TXT、Markdown 或代码文件；Markdown 提取 frontmatter 标题等元数据
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

        title = resolved.stem
        metadata: dict[str, str] = {"suffix": suffix, "path": str(resolved)}
        if mime_type == "text/markdown":
            front = self._parse_frontmatter(content)
            if front is not None:
                if front.get("title"):
                    title = front["title"]
                for key in ("description", "keywords", "tags"):
                    if key in front:
                        metadata[key] = front[key]
        return ParsedDocument(
            content=content,
            title=title,
            mime_type=mime_type,
            metadata=metadata,
        )

    # 提取 Markdown 开头的 YAML frontmatter 字段；没有返回 None
    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str] | None:
        if not text.startswith("---"):
            return None
        lines = text.splitlines()
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is None:
            return None
        front: dict[str, str] = {}
        for line in lines[1:end]:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            value = value.strip("\"'")
            if value.startswith("[") and value.endswith("]"):
                value = ", ".join(
                    v.strip().strip("\"'")
                    for v in value[1:-1].split(",")
                    if v.strip()
                )
            front[key] = value
        return front
