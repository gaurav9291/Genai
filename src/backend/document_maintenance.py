"""Document maintenance helpers: versions, reindexing, and lexical search."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from .config import settings
from .models import Document, DocumentChunk, DocumentVersion
from .rag_qa import _best_matching_window, _keywords, _lexical_score
from .vector_store import get_vector_store


def _nearest_heading(text: str) -> Optional[str]:
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#"):
            return cleaned.lstrip("#").strip()[:120]
        if re.match(r"^(\d+(\.\d+)*\.?|[A-Z][A-Za-z ]{2,}:?)\s+", cleaned):
            return cleaned[:120]
    return None


def create_document_version(
    db: Session,
    document: Document,
    *,
    change_note: str = "Manual snapshot",
) -> DocumentVersion:
    """Snapshot a document before a potentially mutating change."""
    version = DocumentVersion(
        document_id=document.id,
        title=document.title,
        content=document.content or "",
        status=document.status,
        feedback_score=document.feedback_score,
        change_note=change_note,
    )
    db.add(version)
    return version


def list_document_versions(db: Session, document_id: str, limit: int = 20) -> List[DocumentVersion]:
    return (
        db.query(DocumentVersion)
        .filter(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.created_at.desc())
        .limit(max(1, min(limit, 100)))
        .all()
    )


def restore_document_version(db: Session, version_id: str) -> Dict[str, Any]:
    version = db.query(DocumentVersion).filter(DocumentVersion.id == version_id).first()
    if not version:
        return {"status": "error", "message": "Version not found"}

    document = db.query(Document).filter(Document.id == version.document_id).first()
    if not document:
        return {"status": "error", "message": "Document not found"}

    create_document_version(db, document, change_note=f"Snapshot before restoring version {version.id}")
    document.title = version.title
    document.content = version.content
    document.status = version.status
    document.feedback_score = version.feedback_score
    db.commit()

    reindex_result = reindex_document(db, document.id, add_to_vector_store=False)
    return {
        "status": "success",
        "document_id": document.id,
        "restored_version_id": version.id,
        "reindex": reindex_result,
    }


def _split_content(content: str, document_id: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for index, chunk in enumerate(splitter.split_text(content or "")):
        text = chunk.strip()
        if not text:
            continue
        chunks.append(
            {
                "content": text,
                "metadata": {
                    "chunk_index": index,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "document_id": document_id,
                    "section_heading": _nearest_heading(text),
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "has_previous_overlap": index > 0 and settings.chunk_overlap > 0,
                    "split_strategy": "recursive_heading_paragraph_sentence_word",
                },
            }
        )
    return chunks


def reindex_document(
    db: Session,
    document_id: str,
    *,
    add_to_vector_store: bool = False,
) -> Dict[str, Any]:
    """Rebuild SQLite chunks for one document and optionally add fresh vector docs.

    Existing vector IDs are not persisted per chunk in the current schema, so old
    vector entries cannot be safely deleted. SQLite chunks are rebuilt exactly;
    vector additions are best-effort.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return {"status": "error", "message": "Document not found", "document_id": document_id}

    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(synchronize_session=False)

    chunks = _split_content(document.content or "", document.id)
    db_chunks = []
    lc_docs = []
    for index, chunk_data in enumerate(chunks):
        metadata = chunk_data["metadata"]
        metadata.update(
            {
                "approved": document.approved,
                "feedback_score": document.feedback_score,
                "doc_type": document.doc_type,
                "title": document.title,
            }
        )
        db_chunks.append(
            DocumentChunk(
                document_id=document.id,
                content=chunk_data["content"],
                chunk_index=index,
                chunk_metadata=metadata,
                embedding_model=settings.embedding_model,
            )
        )
        lc_docs.append(LCDocument(page_content=chunk_data["content"], metadata=metadata))

    if db_chunks:
        db.add_all(db_chunks)
    document.updated_at = datetime.utcnow()
    db.commit()

    vector_ids: List[str] = []
    vector_warning = None
    if add_to_vector_store and lc_docs:
        try:
            vector_store = get_vector_store()
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                vector_ids = asyncio.run(vector_store.async_add_documents(lc_docs))
            else:
                # Avoid blocking Streamlit/FastAPI event loops with nested runs.
                vector_warning = "Vector store update skipped inside active event loop; SQLite chunks were rebuilt."
        except Exception as exc:
            vector_warning = str(exc)

    return {
        "status": "success",
        "document_id": document.id,
        "chunk_count": len(db_chunks),
        "vector_ids": vector_ids,
        "vector_warning": vector_warning,
    }


def reindex_all_documents(db: Session, *, add_to_vector_store: bool = False) -> Dict[str, Any]:
    documents = db.query(Document).order_by(Document.updated_at.desc()).all()
    results = []
    for document in documents:
        results.append(reindex_document(db, document.id, add_to_vector_store=add_to_vector_store))

    return {
        "status": "success",
        "document_count": len(documents),
        "total_chunks": sum(item.get("chunk_count", 0) for item in results if item.get("status") == "success"),
        "results": results,
    }


def search_documents(
    db: Session,
    query: str,
    *,
    doc_type: Optional[str] = None,
    limit: int = 5,
    snippet_chars: int = 900,
) -> List[Dict[str, Any]]:
    """Return raw matching snippets for MCP/tool inspection."""
    terms = _keywords(query)
    if not terms:
        return []

    limit = max(1, min(limit, 25))
    snippet_chars = max(250, min(snippet_chars, 3000))
    results: List[Dict[str, Any]] = []
    seen = set()

    chunk_query = db.query(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
    if doc_type and doc_type != "All":
        chunk_query = chunk_query.filter(Document.doc_type == doc_type)

    for chunk, document in chunk_query.limit(2000).all():
        score = _lexical_score(terms, chunk.content or "")
        if score <= 0:
            continue
        key = (document.id, chunk.chunk_index, (chunk.content or "")[:60])
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "document_id": document.id,
                "title": document.title,
                "doc_type": document.doc_type,
                "chunk_index": chunk.chunk_index,
                "score": score,
                "matched_terms": sorted(terms & _keywords(chunk.content or "")),
                "snippet": (chunk.content or "")[:snippet_chars],
            }
        )

    document_query = db.query(Document)
    if doc_type and doc_type != "All":
        document_query = document_query.filter(Document.doc_type == doc_type)

    chunked_ids = {item["document_id"] for item in results}
    for document in document_query.limit(1000).all():
        if document.id in chunked_ids:
            continue
        content = document.content or ""
        score = _lexical_score(terms, content)
        if score <= 0:
            continue
        snippet = _best_matching_window(content, terms, window_chars=snippet_chars)
        results.append(
            {
                "document_id": document.id,
                "title": document.title,
                "doc_type": document.doc_type,
                "chunk_index": "full-document",
                "score": score,
                "matched_terms": sorted(terms & _keywords(content)),
                "snippet": snippet,
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def format_search_results_markdown(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No matching document snippets found."

    lines = []
    for index, result in enumerate(results, start=1):
        terms = ", ".join(result.get("matched_terms", [])) or "n/a"
        lines.extend(
            [
                f"### Result {index}: {result['title']}",
                "",
                f"- ID: `{result['document_id']}`",
                f"- Type: {result['doc_type']}",
                f"- Chunk: {result['chunk_index']}",
                f"- Score: {result['score']:.2f}",
                f"- Matched terms: {terms}",
                "",
                "```text",
                result["snippet"].strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip()
