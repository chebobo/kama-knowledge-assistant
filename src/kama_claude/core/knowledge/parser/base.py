from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ParsedDocument:
    # 保存 parser 输出的纯文本和文档级元数据
    content: str
    title: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Parser(Protocol):
    # 将文件解析为可分段的结构化文本
    def parse(self, path: Path) -> ParsedDocument: ...

