from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path

from kama_claude.core.knowledge.models import KnowledgeChunk, KnowledgeFilter, KnowledgeSearchResult


class Fts5UnavailableError(RuntimeError):
    # 表示当前 SQLite 不支持项目要求的 FTS5 trigram tokenizer
    pass


class SqliteLexicalStore:
    # 初始化基于 SQLite FTS5 trigram 的 chunk 索引
    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._path = str(Path(self._path).expanduser())
        try:
            self._connection = sqlite3.connect(self._path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise Fts5UnavailableError(f"cannot open lexical index: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        try:
            self._initialize()
        except sqlite3.Error as exc:
            self._connection.close()
            raise Fts5UnavailableError(
                f"SQLite FTS5 trigram tokenizer is unavailable: {exc}"
            ) from exc

    # 返回 SQLite 版本并验证 FTS5 trigram 虚表可创建
    @staticmethod
    def probe_trigram_support() -> str:
        try:
            with sqlite3.connect(":memory:") as connection:
                connection.execute(
                    "CREATE VIRTUAL TABLE fts_probe USING fts5(content, tokenize=trigram)"
                )
        except sqlite3.Error as exc:
            raise Fts5UnavailableError(
                f"SQLite FTS5 trigram tokenizer is unavailable: {exc}"
            ) from exc
        return sqlite3.sqlite_version

    # 创建用于 BM25 排序和 scope 过滤的 FTS5 表
    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                content,
                section,
                title,
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                source UNINDEXED,
                scope UNINDEXED,
                project_id UNINDEXED,
                user_id UNINDEXED,
                metadata_json UNINDEXED,
                tokenize=trigram
            )
            """
        )
        self._connection.commit()

    # 幂等替换一批文档 chunk 的 lexical rows
    async def replace_document(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._replace_document_sync, document_id, chunks)

    # 删除指定文档旧 rows 并写入新 rows
    def _replace_document_sync(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM knowledge_chunks_fts WHERE document_id = ?", (document_id,)
            )
            self._connection.executemany(
                """
                INSERT INTO knowledge_chunks_fts (
                    content, section, title, chunk_id, document_id, source,
                    scope, project_id, user_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.content,
                        str(chunk.metadata.get("section", "")),
                        chunk.title,
                        chunk.id,
                        chunk.document_id,
                        chunk.source,
                        chunk.scope,
                        chunk.project_id,
                        chunk.user_id,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    )
                    for chunk in chunks
                ],
            )

    # 按 document_id 删除全部 lexical rows
    async def delete_document(self, document_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_document_sync, document_id)

    # 在事务中执行文档级 FTS 删除
    def _delete_document_sync(self, document_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM knowledge_chunks_fts WHERE document_id = ?", (document_id,)
            )

    # 按 trigram BM25 相关度和权限范围查询 chunks
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        match_query = self._make_match_query(query)
        if not match_query:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, match_query, top_k, filters)

    # 将查询词切成 trigram OR 表达式，允许文档命中关键子串而非整句
    @staticmethod
    def _make_match_query(query: str) -> str:
        terms = {
            token[index : index + 3]
            for token in re.findall(r"\w+", query)
            for index in range(len(token) - 2)
        }
        return " OR ".join(f'"{term}"' for term in sorted(terms))

    # 通过 SQL WHERE 在检索存储层限制 user、global 和当前 project
    def _search_sync(
        self,
        match_query: str,
        top_k: int,
        filters: KnowledgeFilter,
    ) -> list[KnowledgeSearchResult]:
        conditions = ["knowledge_chunks_fts MATCH ?"]
        parameters: list[object] = [match_query]
        if filters.user_id is not None:
            conditions.append("user_id = ?")
            parameters.append(filters.user_id)
        if filters.project_id is None:
            conditions.append("scope = 'global'")
        else:
            conditions.append("(scope = 'global' OR (scope = 'project' AND project_id = ?))")
            parameters.append(filters.project_id)
        parameters.append(top_k)
        sql = f"""
            SELECT content, section, title, chunk_id, document_id, source,
                   scope, project_id, user_id, metadata_json,
                   bm25(
                       knowledge_chunks_fts,
                       1.0, 1.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                   ) AS score
            FROM knowledge_chunks_fts
            WHERE {' AND '.join(conditions)}
            ORDER BY score ASC, chunk_id ASC
            LIMIT ?
        """
        rows = self._connection.execute(sql, parameters).fetchall()
        results: list[KnowledgeSearchResult] = []
        for row in rows:
            metadata = json.loads(str(row["metadata_json"]))
            if row["section"]:
                metadata["section"] = str(row["section"])
            results.append(
                KnowledgeSearchResult(
                    chunk=KnowledgeChunk(
                        id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=str(row["content"]),
                        source=str(row["source"]),
                        title=str(row["title"]),
                        scope=str(row["scope"]),  # type: ignore[arg-type]
                        project_id=(str(row["project_id"]) if row["project_id"] else None),
                        user_id=str(row["user_id"]) if row["user_id"] else None,
                        metadata=metadata,
                    ),
                    score=float(row["score"]),
                )
            )
        return results

    # 关闭 SQLite 连接并释放文件句柄
    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.close)
