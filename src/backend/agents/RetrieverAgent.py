"""Retriever Agent — Tool-Calling Architecture using llm.bind_tools() (LCEL).

Uses the raw LangChain Core tool-calling API (`llm.bind_tools()`) instead of
the high-level AgentExecutor so it works across ALL LangChain versions ≥0.1.

Two StructuredTools are defined:
  • vector_similarity_search  — semantic FAISS/Pinecone similarity search
  • sqlite_keyword_search      — fast lexical keyword search against the DB

The LLM is bound to both tools and runs a lightweight reasoning loop:
  1. LLM inspects the query and decides which tool(s) to call.
  2. Tool calls are executed locally and results fed back to the LLM.
  3. Collected chunks are deduplicated and reranked before returning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import tenacity
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from ..config import settings
from ..llm_factory import get_chat_model
from ..vector_store import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool input schemas  (Pydantic v2)
# ---------------------------------------------------------------------------

class VectorSearchInput(BaseModel):
    query: str = Field(description="Search query for semantic similarity retrieval.")
    k: int = Field(default=6, description="Max chunks to return.")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type.")


class SQLiteSearchInput(BaseModel):
    query: str = Field(description="Keyword(s) to search inside document content.")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type.")
    min_feedback_score: Optional[int] = Field(default=None, description="Minimum quality score (1-5).")
    limit: int = Field(default=6, description="Max chunks to return.")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RetrieverAgent(BaseAgent):
    """Tool-Calling Retriever using llm.bind_tools() — works on all LangChain versions."""

    def __init__(self):
        super().__init__(
            name="retriever",
            description="Tool-calling agent that retrieves relevant document chunks.",
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._chunk_buffer: List[Dict[str, Any]] = []

        # Embeddings + vector store
        self.embeddings = get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        if os.environ.get("PINECONE_API_KEY"):
            from langchain_pinecone import PineconeVectorStore
            self.vs = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=settings.pinecone_namespace or "default",
            )
        else:
            self.vs = get_vector_store()

        # Build tools
        self._tools = self._build_tools()
        self._tool_map: Dict[str, StructuredTool] = {t.name: t for t in self._tools}

        # Retrieval is deterministic by default to avoid spending an LLM call
        # just to decide which local search function to run.
        self.llm = None
        self._llm_with_tools = None
        self._tool_calling_supported = False

    # ------------------------------------------------------------------
    # Tool implementations (write results into self._chunk_buffer)
    # ------------------------------------------------------------------

    def _vector_search_impl(self, query: str, k: int = 6, doc_type: Optional[str] = None) -> str:
        """Semantic similarity search against the vector store."""
        try:
            vs_backend = self.vs.vs if hasattr(self.vs, "vs") else self.vs
            if vs_backend is None:
                return "Vector store unavailable."

            results: List[tuple] = []
            filter_dict = {"doc_type": doc_type} if doc_type else {}

            if hasattr(vs_backend, "similarity_search_with_score"):
                try:
                    results = vs_backend.similarity_search_with_score(query, k=k, filter=filter_dict)
                except Exception:
                    results = vs_backend.similarity_search_with_score(query, k=k)
            elif hasattr(vs_backend, "similarity_search"):
                docs = vs_backend.similarity_search(query, k=k)
                results = [(d, 0.5) for d in docs]

            for doc, score in results:
                self._chunk_buffer.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source": "vector",
                })

            return f"vector_similarity_search: found {len(results)} chunks."
        except Exception as exc:
            logger.warning("_vector_search_impl error: %s", exc)
            return f"vector_similarity_search failed: {exc}"

    def _sqlite_search_impl(
        self,
        query: str,
        doc_type: Optional[str] = None,
        min_feedback_score: Optional[int] = None,
        limit: int = 6,
    ) -> str:
        """Keyword search against the SQLite documents table."""
        try:
            from ..database import SessionLocal
            from ..models import Document

            db = SessionLocal()
            try:
                q = db.query(Document)
                if doc_type:
                    q = q.filter(Document.doc_type == doc_type)
                if min_feedback_score is not None:
                    q = q.filter(Document.feedback_score >= min_feedback_score)
                q = q.filter(Document.approved == True)  # noqa: E712
                docs = q.all()

                keywords = [w.lower() for w in query.split() if len(w) > 3]
                scored: List[tuple] = []
                for doc in docs:
                    content_lower = (doc.content or "").lower()
                    hits = sum(content_lower.count(kw) for kw in keywords)
                    if hits > 0:
                        scored.append((hits, doc))
                scored.sort(key=lambda x: x[0], reverse=True)

                found = 0
                for hits, doc in scored[:limit]:
                    text = doc.content or ""
                    for i in range(0, min(len(text), 2000), 400):
                        self._chunk_buffer.append({
                            "content": text[i: i + 400],
                            "metadata": {
                                "doc_type": doc.doc_type,
                                "title": doc.title,
                                "feedback_score": doc.feedback_score,
                                "document_id": str(doc.id),
                            },
                            "score": hits / max(len(keywords), 1),
                            "source": "sqlite",
                        })
                        found += 1

                return f"sqlite_keyword_search: found {found} chunks across {len(scored)} docs."
            finally:
                db.close()
        except Exception as exc:
            logger.warning("_sqlite_search_impl error: %s", exc)
            return f"sqlite_keyword_search failed: {exc}"

    # ------------------------------------------------------------------
    # Build StructuredTools
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._vector_search_impl,
                name="vector_similarity_search",
                description=(
                    "Semantic vector similarity search. Best for conceptual queries "
                    "like 'authentication requirements' or 'error handling policy'."
                ),
                args_schema=VectorSearchInput,
            ),
            StructuredTool.from_function(
                func=self._sqlite_search_impl,
                name="sqlite_keyword_search",
                description=(
                    "Keyword/lexical search across document text. Best for exact terms, "
                    "section names, or specific technology names. Supports quality filtering."
                ),
                args_schema=SQLiteSearchInput,
            ),
        ]

    # ------------------------------------------------------------------
    # LCEL Tool-Calling Loop  (uses llm.bind_tools, no AgentExecutor)
    # ------------------------------------------------------------------

    def _run_tool_loop(self, query: str) -> None:
        """Run a lightweight bind_tools loop synchronously.

        The LLM inspects the query, picks tools, we execute them,
        feed results back, and allow one more round of tool calls if needed.
        Max 3 iterations to avoid runaway loops.
        """
        system = (
            "You are a retrieval specialist. Your ONLY job is to call the provided tools "
            "to retrieve relevant document chunks for the user's query.\n"
            "Always call vector_similarity_search first. "
            "If the query contains very specific technical terms or exact names, "
            "also call sqlite_keyword_search.\n"
            "After calling tools, respond with a brief summary like 'Retrieved N chunks.'"
        )

        messages = [
            {"role": "system", "content": system},
            HumanMessage(content=query),
        ]

        for iteration in range(3):
            response: AIMessage = self._llm_with_tools.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                # LLM finished — no more tool calls
                break

            # Execute each tool call and feed results back
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc.get("id", tool_name)

                tool = self._tool_map.get(tool_name)
                if tool is None:
                    result_str = f"Unknown tool: {tool_name}"
                else:
                    try:
                        result_str = tool.invoke(tool_args)
                    except Exception as exc:
                        result_str = f"Tool error: {exc}"

                messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_id)
                )

            logger.debug("RetrieverAgent tool loop iteration %d complete.", iteration + 1)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        seen: set = set()
        unique = []
        for c in chunks:
            key = c.get("content", "")[:80]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _rerank(self, chunks: List[Dict], top_k: int) -> List[Dict]:
        def rank(c: Dict) -> float:
            relevance = float(c.get("score", 0.5))
            fb = int(c.get("metadata", {}).get("feedback_score", 3))
            return 0.7 * relevance + 0.3 * ((fb - 1) / 4.0)
        try:
            return sorted(chunks, key=rank, reverse=True)[:top_k]
        except Exception:
            return chunks[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=6),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    async def execute(
        self,
        query: str,
        top_k: int = None,
        doc_type: Optional[str] = None,
        min_feedback_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the tool-calling loop and return ranked document chunks."""

        if not query or not query.strip():
            return {"status": "error", "message": "Empty query.", "chunks": [], "total_results": 0}

        top_k = max(1, min(top_k or 3, 3))
        logger.info("RetrieverAgent.execute(): query='%s...', top_k=%d", query[:80], top_k)

        # Reset per-call buffer
        self._chunk_buffer = []

        try:
            logger.debug("Executing deterministic vector and SQLite search.")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self._vector_search_impl(query=query, k=top_k, doc_type=doc_type),
            )
            await loop.run_in_executor(
                self.executor,
                lambda: self._sqlite_search_impl(
                    query=query,
                    doc_type=doc_type,
                    min_feedback_score=min_feedback_score,
                    limit=top_k,
                ),
            )

            # Post-process
            chunks = self._deduplicate(self._chunk_buffer)
            if min_feedback_score is not None:
                chunks = [
                    c for c in chunks
                    if int(c.get("metadata", {}).get("feedback_score", 5)) >= min_feedback_score
                ]
            chunks = self._rerank(chunks, top_k)
            logger.info("RetrieverAgent: returning %d chunks.", len(chunks))

            return {
                "status": "success",
                "chunks": chunks,
                "total_results": len(chunks),
                "query": query,
                "filters": {"doc_type": doc_type, "min_feedback_score": min_feedback_score},
            }

        except Exception as exc:
            logger.error("RetrieverAgent.execute() failed: %s", exc)
            return {
                "status": "error",
                "message": str(exc),
                "chunks": [],
                "total_results": 0,
                "query": query,
            }

    async def retrieve_documents(self, query: str, doc_type: Optional[str] = None, min_score: Optional[int] = None) -> List[Dict]:
        result = await self.execute(query=query, doc_type=doc_type, min_feedback_score=min_score)
        return result.get("chunks", [])

    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        results = await self.retrieve_documents(query=query, doc_type=kwargs.get("doc_type"), min_score=kwargs.get("min_score"))
        return {"status": "success", "results": results, "query": query}
