"""DocAssist — MCP Server (Model Context Protocol).

This file exposes the document workspace's core capabilities as MCP tools,
allowing external AI clients (Claude Desktop, Cursor, Windsurf, etc.) to:

  • generate_document   — run the full agentic generation workflow
  • ask_documents       — query uploaded documents using RAG
  • list_document_types — discover what document types exist in the library
  • list_documents      — inspect saved document IDs and metadata
  • get_document_content — read a specific saved document by ID or title
  • search_documents    — inspect raw matching snippets before answering
  • import_pdf_from_url — ingest a lawful direct PDF download URL

Usage (stdio transport — works with any MCP client):
    python mcp_server.py

Usage (HTTP/SSE transport — for remote clients):
    python mcp_server.py --transport http --port 8765

Add to Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "docassist": {
          "command": "python",
          "args": ["/path/to/mcp_server.py"],
          "env": { "GROQ_API_KEY": "<your-key>" }
        }
      }
    }
"""

import asyncio
import sys
import os

# Ensure the project root is on the Python path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="DocAssist",
    instructions=(
        "This server exposes a GenAI-powered document generation and management workspace. "
        "Use generate_document to create new technical documents, "
        "ask_documents to query uploaded references, and "
        "list_document_types/list_documents to see what is available, "
        "search_documents to inspect raw evidence, and get_document_content "
        "to read a specific saved document when exact analysis is needed. "
        "Use import_pdf_from_url only for lawful direct PDF download links "
        "provided by the user."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: Generate a document
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_document(
    doc_type: str,
    topic: str,
    outline: str,
    style: str = "professional",
) -> str:
    """Generate a complete technical document using the agentic RAG workflow.

    Args:
        doc_type: The type of document to generate (e.g. 'User Guide', 'SRS',
                  'Report', 'Proposal', 'Policy', 'SOW').
        topic:    A short description of the document's subject or goal.
        outline:  Key sections, requirements, or bullet points the document
                  must cover.
        style:    Writing style — one of 'professional', 'formal', 'technical',
                  or 'concise'. Defaults to 'professional'.

    Returns:
        The generated document content in Markdown format.
    """
    from src.backend.workflow import workflow_manager
    from src.backend.database import SessionLocal, create_tables

    create_tables()
    db = SessionLocal()
    try:
        result = await workflow_manager.execute_generation_workflow(
            doc_type=doc_type,
            summary=topic,
            requirements=outline,
            style=style,
            db=db,
            max_iterations=2,
        )
        content = result.get("content", "")
        status = result.get("status", "unknown")
        if status == "failed" or not content.strip():
            error = result.get("error", "Unknown error during generation.")
            return f"[ERROR] Document generation failed: {error}"
        return content
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 2: Ask questions over uploaded documents (RAG)
# ---------------------------------------------------------------------------

@mcp.tool()
async def ask_documents(
    question: str,
    doc_type: str = "All",
    top_k: int = 3,
) -> str:
    """Ask a question over documents that have been uploaded to the workspace.

    Uses Retrieval-Augmented Generation (RAG) — fetches the most relevant
    chunks from the vector store and uses the LLM to synthesise an answer.

    Args:
        question: The question to answer (e.g. 'What are the authentication
                  requirements?').
        doc_type: Limit the search to a specific document type, or 'All' to
                  search across everything. Defaults to 'All'.
        top_k:    Number of document chunks to retrieve for context (1–3).
                  Defaults to 3.

    Returns:
        A cited answer in Markdown format, including source references.
    """
    from src.backend.rag_qa import answer_question
    from src.backend.database import SessionLocal, create_tables

    create_tables()
    db = SessionLocal()
    try:
        result = await answer_question(
            db,
            question,
            doc_type=doc_type if doc_type != "All" else None,
            top_k=max(1, min(top_k, 3)),
        )
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])

        if sources:
            def _source_line(source):
                updated_at = source.get("updated_at")
                updated_label = updated_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(updated_at, "strftime") else "unknown"
                match_label = "direct match" if source.get("direct_match", True) else "related source"
                retrieval_label = source.get("retrieval_source", "lexical")
                return (
                    f"- **{source['title']}** "
                    f"({source['doc_type']}, chunk {source['chunk_index']}, "
                    f"{match_label}, {retrieval_label}, updated {updated_label})"
                )

            source_lines = "\n".join(
                _source_line(s)
                for s in sources
            )
            return f"{answer}\n\n### Sources\n{source_lines}"
        return answer
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 3: List document types in the library
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_document_types() -> str:
    """List all document types currently available in the document library.

    Returns:
        A Markdown-formatted list of document types and how many documents
        of each type exist.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.models import Document
    from collections import Counter

    create_tables()
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        if not docs:
            return "No documents found in the library. Upload some documents first."

        counts = Counter(d.doc_type for d in docs)
        lines = [f"| Document Type | Count |", "|---|---|"]
        for dtype, count in sorted(counts.items()):
            lines.append(f"| {dtype} | {count} |")
        return "\n".join(lines)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 4: List saved documents in the library
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_documents(
    doc_type: str = "All",
    limit: int = 25,
) -> str:
    """List saved documents with IDs and metadata.

    Use this before get_document_content when the user asks about a specific
    document or when exact document-level analysis is needed.

    Args:
        doc_type: Optional document type filter, or 'All' for every type.
        limit: Maximum documents to return. Capped at 50.

    Returns:
        A Markdown table containing document IDs, titles, types, status, and size.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.models import Document

    create_tables()
    db = SessionLocal()
    try:
        query = db.query(Document)
        if doc_type and doc_type != "All":
            query = query.filter(Document.doc_type == doc_type)

        docs = (
            query
            .order_by(Document.updated_at.desc())
            .limit(max(1, min(limit, 50)))
            .all()
        )
        if not docs:
            return "No matching documents found in the library."

        lines = [
            "| ID | Title | Type | Status | Words | Updated |",
            "|---|---|---|---|---:|---|",
        ]
        for doc in docs:
            words = len((doc.content or "").split())
            updated = doc.updated_at.isoformat(sep=" ", timespec="seconds") if doc.updated_at else ""
            lines.append(
                f"| `{doc.id}` | {doc.title} | {doc.doc_type} | {doc.status} | {words} | {updated} |"
            )
        return "\n".join(lines)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 5: Read a saved document by ID or title search
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_document_content(
    document_id: str = "",
    title_query: str = "",
    max_chars: int = 20000,
) -> str:
    """Read the content of one saved document.

    This is read-only. Prefer document_id from list_documents for exact matches;
    title_query is a convenience fallback that selects the most recently updated
    matching title.

    Args:
        document_id: Exact document UUID to read.
        title_query: Case-insensitive title substring to search when ID is unknown.
        max_chars: Maximum content characters to return. Capped at 40000.

    Returns:
        Document metadata followed by Markdown content, possibly truncated.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.models import Document

    create_tables()
    db = SessionLocal()
    try:
        document = None
        if document_id.strip():
            document = db.query(Document).filter(Document.id == document_id.strip()).first()
        elif title_query.strip():
            needle = f"%{title_query.strip()}%"
            document = (
                db.query(Document)
                .filter(Document.title.ilike(needle))
                .order_by(Document.updated_at.desc())
                .first()
            )
        else:
            return "Provide either document_id or title_query."

        if not document:
            return "Document not found."

        cap = max(500, min(max_chars, 40000))
        content = document.content or ""
        clipped = content[:cap]
        truncated_note = ""
        if len(content) > cap:
            truncated_note = f"\n\n[TRUNCATED: showing {cap} of {len(content)} characters]"

        updated = document.updated_at.isoformat(sep=" ", timespec="seconds") if document.updated_at else ""
        header = (
            f"# {document.title}\n\n"
            f"- ID: `{document.id}`\n"
            f"- Type: {document.doc_type}\n"
            f"- Status: {document.status}\n"
            f"- Updated: {updated}\n"
            f"- Characters: {len(content)}\n\n"
            "---\n\n"
        )
        return f"{header}{clipped}{truncated_note}"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 6: Search raw document snippets
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_documents(
    query: str,
    doc_type: str = "All",
    limit: int = 5,
    snippet_chars: int = 900,
) -> str:
    """Search saved documents and return raw matching snippets.

    This is the best tool for evidence inspection before answering. It returns
    snippets, document IDs, chunk IDs, scores, and matched terms instead of an
    LLM-synthesized answer.

    Args:
        query: Search terms or question.
        doc_type: Optional document type filter, or 'All'.
        limit: Maximum snippets to return. Capped at 25.
        snippet_chars: Maximum characters per snippet. Capped at 3000.

    Returns:
        Markdown-formatted raw search results.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.document_maintenance import (
        format_search_results_markdown,
        search_documents as search_document_snippets,
    )

    create_tables()
    db = SessionLocal()
    try:
        results = search_document_snippets(
            db,
            query,
            doc_type=doc_type if doc_type != "All" else None,
            limit=limit,
            snippet_chars=snippet_chars,
        )
        return format_search_results_markdown(results)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 7: Import a direct PDF URL
# ---------------------------------------------------------------------------

@mcp.tool()
async def import_pdf_from_url(
    url: str,
    doc_type: str = "Reference",
    title: str = "",
    approved: bool = True,
    feedback_score: int = 3,
) -> str:
    """Download a lawful direct PDF URL, ingest it, and index it.

    This tool only supports direct PDF downloads. It does not scrape websites,
    bypass access controls, or determine whether the user has legal rights to
    store the file. Use it only when the user provides a PDF URL they are
    allowed to import.

    Args:
        url: Direct http/https PDF URL.
        doc_type: Library category for the imported document.
        title: Optional title override after import.
        approved: Whether to use this document for style/context learning.
        feedback_score: Quality score from 1 to 5.

    Returns:
        Markdown import summary with the new document ID and chunk count.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.url_importer import import_pdf_from_url as import_pdf

    create_tables()
    db = SessionLocal()
    try:
        result = await import_pdf(
            db,
            url,
            doc_type=doc_type,
            title=title,
            approved=approved,
            feedback_score=feedback_score,
        )
        if result.get("status") != "success":
            return f"[ERROR] {result.get('message', 'PDF import failed')}"

        metadata = result.get("metadata", {})
        return (
            "Imported PDF successfully.\n\n"
            f"- Document ID: `{result['document_id']}`\n"
            f"- Filename: {metadata.get('filename', 'unknown')}\n"
            f"- Type: {metadata.get('doc_type', doc_type)}\n"
            f"- Words: {metadata.get('word_count', 0)}\n"
            f"- Chunks: {result.get('chunk_count', 0)}\n"
            f"- Source URL: {result.get('source_url', url)}\n"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 8: Rebuild retrieval chunks for one document
# ---------------------------------------------------------------------------

@mcp.tool()
async def reindex_document(
    document_id: str,
) -> str:
    """Rebuild SQLite retrieval chunks for one document.

    This does not change document content. It is useful when a document appears
    in the Library but RAG/Claude cannot retrieve it.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.document_maintenance import reindex_document as rebuild_document_index

    create_tables()
    db = SessionLocal()
    try:
        result = rebuild_document_index(db, document_id, add_to_vector_store=False)
        if result.get("status") != "success":
            return f"[ERROR] {result.get('message', 'Reindex failed')}"
        warning = f"\n\nWarning: {result['vector_warning']}" if result.get("vector_warning") else ""
        return (
            f"Reindexed document `{result['document_id']}` with "
            f"{result['chunk_count']} SQLite chunks.{warning}"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 9: Rebuild retrieval chunks for every document
# ---------------------------------------------------------------------------

@mcp.tool()
async def reindex_all_documents() -> str:
    """Rebuild SQLite retrieval chunks for every saved document.

    This does not change document content. It can take longer on large
    libraries, but it repairs missing/stale chunks for RAG and Claude tools.
    """
    from src.backend.database import SessionLocal, create_tables
    from src.backend.document_maintenance import reindex_all_documents as rebuild_all_indexes

    create_tables()
    db = SessionLocal()
    try:
        result = rebuild_all_indexes(db, add_to_vector_store=False)
        return (
            f"Reindexed {result['document_count']} documents with "
            f"{result['total_chunks']} total SQLite chunks."
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DocAssist MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio for local clients like Claude Desktop)",
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (only used with --transport http)")
    args = parser.parse_args()

    if args.transport == "http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
