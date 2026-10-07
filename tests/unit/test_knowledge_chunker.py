from __future__ import annotations

from kama_claude.core.knowledge.chunker.basic import BasicChunker, MIN_CHUNK_CHARS
from kama_claude.core.knowledge.models import KnowledgeDocument
from kama_claude.core.knowledge.parser.base import ParsedDocument


def _doc() -> KnowledgeDocument:
    return KnowledgeDocument(id="doc-1", source="x.md", title="T", scope="global", user_id="u")


def _parsed(content: str, mime_type: str = "text/markdown") -> ParsedDocument:
    return ParsedDocument(content=content, title="", mime_type=mime_type, metadata={})


# 功能：验证 frontmatter 被跳过，正文 section 才进入 chunks
# 设计：构造带 YAML frontmatter 的 Markdown，断言没有 chunk 以 --- 开头
def test_chunker_skips_frontmatter() -> None:
    text = (
        "---\n"
        "title: Agent 面试\n"
        "description: 深度解析\n"
        "---\n"
        "# 一、混合路由\n\n"
        "混合路由是工程最优解，用规则先筛简单请求，复杂请求再交给大模型处理。"
        "级联降级先让小模型尝试，置信度不足再升级到大模型。\n\n"
        "# 二、级联降级\n\n"
        "小模型先试，置信度不够再升级到大模型，代价是双倍延迟，需要控制好触发阈值。"
        "路由器和评分器共同决定请求走哪条路径。\n"
    )
    chunks = BasicChunker().chunk(_doc(), _parsed(text))
    assert chunks
    assert all(not c.content.lstrip().startswith("---") for c in chunks)
    assert any(c.metadata["section"] == "一、混合路由" for c in chunks)


# 功能：验证低于最小长度的空壳 chunk 被过滤
# 设计：短 section 内容不足 MIN_CHUNK_CHARS 时不应产出 chunk
def test_chunker_filters_short_chunks() -> None:
    text = "# 标题\n\n只有几个字。\n\n# 长章节\n\n" + ("有效正文内容。" * 30)
    chunks = BasicChunker().chunk(_doc(), _parsed(text))
    assert chunks
    assert all(len(c.content.strip()) >= MIN_CHUNK_CHARS for c in chunks)
    assert all(c.metadata["section"] != "标题" for c in chunks)


# 功能：验证长文本在段落边界附近切分，而不是硬切在字符中间
# 设计：构造三个超长段落，断言每个 chunk 的结尾落在段落结尾而非句子中间
def test_chunker_splits_at_paragraph_boundary() -> None:
    p1, p2, p3 = "一" * 1000, "二" * 1000, "三" * 1000
    text = p1 + "\n\n" + p2 + "\n\n" + p3
    chunks = BasicChunker().chunk(_doc(), _parsed(text))
    assert len(chunks) >= 2
    for c in chunks:
        stripped = c.content.rstrip()
        assert stripped.endswith(p1) or stripped.endswith(p2) or stripped.endswith(p3)


# 功能：验证非 Markdown 文件仍然按整篇切分且不过滤掉有意义的文本
# 设计：纯文本长文应产出 chunks，且 section 为空
def test_chunker_plain_text() -> None:
    text = ("普通文本内容。" * 300)  # 2100 字符，超过窗口触发切分
    chunks = BasicChunker().chunk(_doc(), _parsed(text, mime_type="text/plain"))
    assert len(chunks) >= 2
    assert all(c.metadata["section"] == "" for c in chunks)


# 功能：验证切分策略签名包含版本和参数，策略变化会改变签名
# 设计：比较不同参数的签名差异，并断言签名包含版本号
def test_chunker_signature_changes_with_strategy() -> None:
    base = BasicChunker()
    other = BasicChunker(chunk_size=1000, overlap=100)
    assert base.signature != other.signature
    assert "v2" in base.signature
    assert "skip_frontmatter" in base.signature
