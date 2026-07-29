"""Database-backed RAG Q&A for saved document chunks."""

import re
from datetime import datetime, timedelta
from math import sqrt
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import func
from sqlalchemy.orm import Session

from .guardrails import detect_prompt_injection
from .llm_factory import get_chat_model
from .models import Document, DocumentChunk

DEFAULT_TOP_K = 3
MAX_CONTEXT_CHARS_PER_SOURCE = 1600
FULL_DOCUMENT_WINDOW_CHARS = 1800
QA_CACHE_TTL = timedelta(minutes=10)
WEAK_RETRIEVAL_SCORE = 0.4
SEMANTIC_CANDIDATE_LIMIT = 80
SEMANTIC_WEIGHT = 0.45
EVIDENCE_CHUNKS_PER_DOCUMENT = 2
MAX_CHARS_PER_EVIDENCE_CHUNK = 650
_qa_cache: Dict[tuple, tuple[datetime, Dict]] = {}
_embedding_cache: Dict[str, List[float]] = {}

QUERY_SYNONYMS = {
    "love": {"affection", "relationship", "partner", "romance", "romantic", "heart"},
    "romance": {"love", "affection", "relationship", "partner", "heart"},
    "romantic": {"love", "affection", "relationship", "partner", "heart"},
    "relationship": {"love", "affection", "partner", "romance", "heart"},
    "story": {"narrative"},
}

QUESTION_FILLER_TERMS = {
    "about",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "does",
    "did",
    "was",
    "were",
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "their",
    "there",
    "story",
    "narrative",
    "document",
    "documents",
}

def _keywords(text: str) -> set[str]:
    terms = {word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text)}
    expanded = set(terms)
    for term in terms:
        expanded.update(QUERY_SYNONYMS.get(term, set()))
    return expanded


def _focus_terms(query_terms: set[str]) -> set[str]:
    """Return the meaningful part of a question after filler words."""
    return query_terms - QUESTION_FILLER_TERMS


def _lexical_score(query_terms: set[str], content: str) -> float:
    content_terms = _keywords(content)
    score_terms = _focus_terms(query_terms) or query_terms
    if not score_terms or not content_terms:
        return 0.0
    return len(score_terms & content_terms) / max(len(score_terms), 1)


def _direct_match(query_terms: set[str], content: str) -> bool:
    focus_terms = _focus_terms(query_terms)
    if not focus_terms:
        return True
    return bool(_keywords(content) & focus_terms)


def _content_terms(content: str) -> set[str]:
    return _keywords(content or "")


def _embedding_key(source_id: str, content: str) -> str:
    return f"{source_id}:{hash(content or '')}"


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _get_semantic_embeddings():
    try:
        from .vector_store import get_embeddings

        embeddings = get_embeddings()
        if getattr(embeddings, "is_fallback", False):
            return None
        return embeddings
    except Exception:
        return None


def _add_semantic_scores(query: str, candidates: List[Dict]) -> None:
    """Add optional embedding similarity scores to candidates in place.

    This is deliberately best-effort. Semantic retrieval should improve ranking
    when local embeddings are available, but it should never break SQLite RAG.
    """
    if not candidates:
        return

    embeddings = _get_semantic_embeddings()
    if embeddings is None:
        return

    try:
        query_embedding = embeddings.embed_query(query)
    except Exception:
        return

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.get("score", 0.0),
            item.get("updated_at") or datetime.min,
        ),
        reverse=True,
    )[:SEMANTIC_CANDIDATE_LIMIT]

    for candidate in ordered:
        content = candidate.get("content", "")
        key = _embedding_key(
            f"{candidate.get('document_id', '')}:{candidate.get('chunk_index', '')}",
            content,
        )
        try:
            if key not in _embedding_cache:
                _embedding_cache[key] = embeddings.embed_query(content)
            semantic_score = max(0.0, _cosine_similarity(query_embedding, _embedding_cache[key]))
        except Exception:
            continue
        candidate["semantic_score"] = semantic_score
        candidate["hybrid_score"] = (
            (1.0 - SEMANTIC_WEIGHT) * candidate.get("score", 0.0)
            + SEMANTIC_WEIGHT * semantic_score
        )


def _vector_store_candidates(
    db: Session,
    query: str,
    *,
    doc_type: str | None = None,
    limit: int = 12,
) -> List[Dict]:
    try:
        from .vector_store import get_vector_store

        vector_store = get_vector_store()
        if getattr(vector_store.embedding_model, "is_fallback", False):
            return []

        backend = vector_store.vs
        if backend is None or not hasattr(backend, "similarity_search_with_score"):
            return []

        filter_dict = {"doc_type": doc_type} if doc_type and doc_type != "All" else None
        try:
            if filter_dict:
                results = backend.similarity_search_with_score(query, k=limit, filter=filter_dict)
            else:
                results = backend.similarity_search_with_score(query, k=limit)
        except Exception:
            results = backend.similarity_search_with_score(query, k=limit)
    except Exception:
        return []

    candidates = []
    for lc_doc, raw_score in results:
        metadata = getattr(lc_doc, "metadata", {}) or {}
        document_id = metadata.get("document_id")
        document = db.query(Document).filter(Document.id == document_id).first() if document_id else None

        if doc_type and doc_type != "All":
            candidate_type = (document.doc_type if document else metadata.get("doc_type"))
            if candidate_type != doc_type:
                continue

        try:
            raw = float(raw_score)
        except Exception:
            raw = 0.0

        # FAISS returns lower distances; some vector stores return similarity.
        semantic_score = raw if 0.0 <= raw <= 1.0 else 1.0 / (1.0 + max(raw, 0.0))
        content = getattr(lc_doc, "page_content", "") or ""
        chunk_index = metadata.get("chunk_index", "vector")

        candidates.append(
            {
                "content": content,
                "score": 0.0,
                "semantic_score": semantic_score,
                "hybrid_score": semantic_score,
                "document_id": document.id if document else str(document_id or "vector"),
                "title": document.title if document else metadata.get("title", "Vector result"),
                "doc_type": document.doc_type if document else metadata.get("doc_type", "Unknown"),
                "chunk_index": chunk_index,
                "updated_at": document.updated_at if document else datetime.min,
                "created_at": document.created_at if document else datetime.min,
                "direct_match": _direct_match(_keywords(query), content),
                "retrieval_source": "vector",
            }
        )
    return candidates


def _candidate_key(candidate: Dict) -> tuple:
    return (
        candidate.get("document_id"),
        candidate.get("chunk_index"),
        (candidate.get("content") or "")[:100],
    )


def _dedupe_candidates(candidates: List[Dict]) -> List[Dict]:
    merged: Dict[tuple, Dict] = {}
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key not in merged:
            merged[key] = candidate
            continue

        existing = merged[key]
        existing["score"] = max(existing.get("score", 0.0), candidate.get("score", 0.0))
        existing["semantic_score"] = max(existing.get("semantic_score", 0.0), candidate.get("semantic_score", 0.0))
        existing["hybrid_score"] = max(
            existing.get("hybrid_score", existing.get("score", 0.0)),
            candidate.get("hybrid_score", candidate.get("score", 0.0)),
        )
        existing["direct_match"] = existing.get("direct_match", False) or candidate.get("direct_match", False)
        sources = {source for source in [existing.get("retrieval_source"), candidate.get("retrieval_source")] if source}
        if sources:
            existing["retrieval_source"] = "+".join(sorted(sources))

    return list(merged.values())


def _coverage_ratio(terms: set[str], content: str) -> float:
    if not terms:
        return 1.0
    return len(terms & _content_terms(content)) / len(terms)


def _combined_coverage(terms: set[str], sources: List[Dict]) -> float:
    if not terms:
        return 1.0
    combined_terms = set()
    for source in sources:
        combined_terms.update(_content_terms(source.get("content", "")))
    return len(terms & combined_terms) / len(terms)


def _second_pass_score(candidate: Dict, query_terms: set[str]) -> float:
    focus_terms = _focus_terms(query_terms)
    base_score = candidate.get("hybrid_score", candidate.get("score", 0.0))
    return (
        base_score
        + _coverage_ratio(focus_terms, candidate.get("content", "")) * 0.35
        + (0.05 if candidate.get("direct_match") else 0.0)
    )


def _needs_second_pass(ranked: List[Dict], query_terms: set[str]) -> bool:
    if not ranked:
        return False

    focus_terms = _focus_terms(query_terms)
    if not focus_terms:
        return False
    if len(focus_terms) > 4:
        return False

    best_score = max(source.get("hybrid_score", source.get("score", 0.0)) for source in ranked)
    if best_score < WEAK_RETRIEVAL_SCORE:
        return True

    return _combined_coverage(focus_terms, ranked) < 0.75


def _is_too_weak_for_broad_query(ranked: List[Dict], query_terms: set[str]) -> bool:
    focus_terms = _focus_terms(query_terms)
    if len(focus_terms) <= 4 or not ranked:
        return False

    best_score = max(source.get("hybrid_score", source.get("score", 0.0)) for source in ranked)
    return best_score < 0.35 and _combined_coverage(focus_terms, ranked) < 0.5


def _select_diverse_from_ordered(ordered: List[Dict], top_k: int) -> List[Dict]:
    selected: List[Dict] = []
    selected_documents = set()
    for item in ordered:
        if item["document_id"] in selected_documents:
            continue
        selected.append(item)
        selected_documents.add(item["document_id"])
        if len(selected) >= top_k:
            return selected

    for item in ordered:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= top_k:
            return selected

    return selected


def _rank_diverse_candidates(candidates: List[Dict], top_k: int) -> List[Dict]:
    """Prefer the best chunk from each document before repeating a document.

    This lets Ask Documents show different saved versions/sources separately
    instead of spending all context slots on near-duplicate chunks from one doc.
    """
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.get("direct_match", False),
            item.get("hybrid_score", item["score"]),
            item.get("updated_at") or datetime.min,
        ),
        reverse=True,
    )
    return _select_diverse_from_ordered(ordered, top_k)


def _rank_second_pass_candidates(candidates: List[Dict], query_terms: set[str], top_k: int) -> List[Dict]:
    reranked = []
    for candidate in candidates:
        updated = dict(candidate)
        updated["retrieval_pass"] = "second-pass"
        updated["rerank_score"] = _second_pass_score(updated, query_terms)
        reranked.append(updated)

    ordered = sorted(
        reranked,
        key=lambda item: (
            item.get("rerank_score", 0.0),
            item.get("updated_at") or datetime.min,
        ),
        reverse=True,
    )
    return _select_diverse_from_ordered(ordered, top_k)


def _candidate_rank_score(candidate: Dict) -> float:
    return candidate.get(
        "rerank_score",
        candidate.get("hybrid_score", candidate.get("score", 0.0)),
    )


def _candidate_sort_key(candidate: Dict) -> tuple:
    return (
        _candidate_rank_score(candidate),
        candidate.get("direct_match", False),
        candidate.get("semantic_score", 0.0),
        candidate.get("updated_at") or datetime.min,
    )


def _chunk_distance_score(candidate: Dict, selected: List[Dict], max_chunk_index: int) -> float:
    chunk_index = candidate.get("chunk_index")
    if not isinstance(chunk_index, int) or not selected or max_chunk_index <= 0:
        return 0.0

    selected_indices = [
        item.get("chunk_index")
        for item in selected
        if isinstance(item.get("chunk_index"), int)
    ]
    if not selected_indices:
        return 0.0

    distance = min(abs(chunk_index - selected_index) for selected_index in selected_indices)
    return min(1.0, distance / max_chunk_index)


def _select_document_evidence(ranked_items: List[Dict]) -> List[Dict]:
    if len(ranked_items) <= EVIDENCE_CHUNKS_PER_DOCUMENT:
        return ranked_items

    selected = [ranked_items[0]]
    numeric_indices = [
        item.get("chunk_index")
        for item in ranked_items
        if isinstance(item.get("chunk_index"), int)
    ]
    max_chunk_index = max(numeric_indices) if numeric_indices else 0

    while len(selected) < EVIDENCE_CHUNKS_PER_DOCUMENT:
        remaining = [item for item in ranked_items if item not in selected]
        if not remaining:
            break
        next_item = max(
            remaining,
            key=lambda item: (
                _candidate_rank_score(item) + 0.08 * _chunk_distance_score(item, selected, max_chunk_index),
                _candidate_rank_score(item),
            ),
        )
        selected.append(next_item)

    return selected


def _format_evidence_item(item: Dict) -> str:
    content = item.get("content", "")
    return f"[Evidence chunk {item.get('chunk_index')}]\n{content[:MAX_CHARS_PER_EVIDENCE_CHUNK]}"


def _group_candidates_by_document(candidates: List[Dict], query_terms: set[str], top_k: int) -> List[Dict]:
    """Return one evidence bundle per document/version.

    Ask Documents answers are easier to trust when each source maps to one saved
    document/version. This prevents multiple chunks from one version from
    crowding out a different version.
    """
    grouped: Dict[str, List[Dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["document_id"], []).append(candidate)

    sources = []
    focus_terms = _focus_terms(query_terms)
    for document_id, items in grouped.items():
        ranked_items = sorted(items, key=_candidate_sort_key, reverse=True)
        evidence_items = _select_document_evidence(ranked_items)
        primary = evidence_items[0]

        combined_content = "\n\n".join(_format_evidence_item(item) for item in evidence_items if item.get("content"))
        evidence_indices = [
            str(item.get("chunk_index"))
            for item in evidence_items
            if item.get("chunk_index") is not None
        ]
        retrieval_sources = sorted(
            {
                item.get("retrieval_source")
                for item in evidence_items
                if item.get("retrieval_source")
            }
        )

        source = dict(primary)
        source["content"] = combined_content or primary.get("content", "")
        source["chunk_index"] = ", ".join(evidence_indices) if evidence_indices else primary.get("chunk_index")
        source["retrieval_source"] = "+".join(retrieval_sources) if retrieval_sources else primary.get("retrieval_source", "lexical")
        source["direct_match"] = any(item.get("direct_match", False) for item in evidence_items)
        source["score"] = max(item.get("score", 0.0) for item in evidence_items)
        source["semantic_score"] = max(item.get("semantic_score", 0.0) for item in evidence_items)
        source["hybrid_score"] = max(item.get("hybrid_score", item.get("score", 0.0)) for item in evidence_items)
        source["rerank_score"] = max(_candidate_rank_score(item) for item in evidence_items)
        source["document_coverage"] = _coverage_ratio(focus_terms, source["content"])
        source["evidence_chunk_count"] = len(evidence_items)
        sources.append(source)

    ordered_sources = sorted(
        sources,
        key=lambda item: (
            item.get("document_coverage", 0.0),
            _candidate_rank_score(item),
            item.get("updated_at") or datetime.min,
        ),
        reverse=True,
    )
    return ordered_sources[:top_k]


def _best_matching_window(content: str, query_terms: set[str], window_chars: int = FULL_DOCUMENT_WINDOW_CHARS) -> str:
    """Return a compact full-document excerpt around the first strong term hit."""
    if len(content) <= window_chars:
        return content

    lowered = content.lower()
    positions = [
        lowered.find(term)
        for term in query_terms
        if len(term) > 3 and lowered.find(term) >= 0
    ]
    if not positions:
        return content[:window_chars]

    center = min(positions)
    start = max(0, center - window_chars // 4)
    end = min(len(content), start + window_chars)
    start = max(0, end - window_chars)
    return content[start:end]


def _document_signature(db: Session) -> tuple:
    count, latest = db.query(func.count(Document.id), func.max(Document.updated_at)).one()
    return int(count or 0), latest.isoformat() if latest else ""


def _is_inventory_question(question: str) -> bool:
    terms = _keywords(question)
    return bool(
        {"inventory", "count", "counts", "types", "categories", "documents"} & terms
        and {"document", "documents", "type", "types", "category", "categories", "inventory"} & terms
    )


def _answer_inventory_question(db: Session) -> Dict:
    rows = (
        db.query(Document.doc_type, func.count(Document.id))
        .group_by(Document.doc_type)
        .order_by(Document.doc_type)
        .all()
    )
    if not rows:
        answer = "No documents found in the library."
    else:
        lines = ["| Document Type | Count |", "|---|---:|"]
        lines.extend(f"| {doc_type} | {count} |" for doc_type, count in rows)
        answer = "\n".join(lines)

    return {
        "status": "success",
        "answer": answer,
        "sources": [],
        "reasons": [],
    }


def retrieve_db_context(
    db: Session,
    query: str,
    *,
    doc_type: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:
    """Retrieve relevant SQLite context using chunks plus full-document fallback."""
    query_terms = _keywords(query)
    if not query_terms:
        return []

    candidates = []

    chunk_query = db.query(DocumentChunk, Document).join(
        Document,
        DocumentChunk.document_id == Document.id,
    )
    if doc_type and doc_type != "All":
        chunk_query = chunk_query.filter(Document.doc_type == doc_type)

    chunked_document_ids = set()
    for chunk, document in chunk_query.limit(1000).all():
        chunked_document_ids.add(document.id)
        content = chunk.content or ""
        score = _lexical_score(query_terms, content)
        if score <= 0:
            continue
        candidates.append(
            {
                "content": content,
                "score": score,
                "document_id": document.id,
                "title": document.title,
                "doc_type": document.doc_type,
                "chunk_index": chunk.chunk_index,
                "updated_at": document.updated_at,
                "created_at": document.created_at,
                "direct_match": _direct_match(query_terms, content),
                "retrieval_source": "lexical",
            }
        )

    document_query = db.query(Document)
    if doc_type and doc_type != "All":
        document_query = document_query.filter(Document.doc_type == doc_type)

    for document in document_query.limit(1000).all():
        # Some Streamlit save paths write the document row without rebuilding
        # DocumentChunk rows. Search those full documents so RAG does not miss
        # visible Library content.
        if document.id in chunked_document_ids:
            continue

        content = document.content or ""
        score = _lexical_score(query_terms, content)
        if score <= 0:
            continue

        candidates.append(
            {
                "content": _best_matching_window(content, query_terms),
                "score": score,
                "document_id": document.id,
                "title": document.title,
                "doc_type": document.doc_type,
                "chunk_index": "full-document",
                "updated_at": document.updated_at,
                "created_at": document.created_at,
                "direct_match": _direct_match(query_terms, content),
                "retrieval_source": "full-document",
            }
        )

    candidates.extend(_vector_store_candidates(db, query, doc_type=doc_type, limit=max(12, top_k * 4)))
    candidates = _dedupe_candidates(candidates)
    _add_semantic_scores(query, candidates)

    preliminary = _rank_diverse_candidates(candidates, top_k)
    if _needs_second_pass(preliminary, query_terms):
        candidates = [
            dict(candidate, retrieval_pass="second-pass", rerank_score=_second_pass_score(candidate, query_terms))
            for candidate in candidates
        ]

    ranked = _group_candidates_by_document(candidates, query_terms, top_k)
    if _is_too_weak_for_broad_query(ranked, query_terms):
        return []

    # If a matching chunk sits beside the actual answer, include nearby chunks
    # from the same document so split boundaries do not hide important context.
    for item in ranked:
        chunk_index = item.get("chunk_index")
        if not isinstance(chunk_index, int):
            continue
        neighbors = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == item["document_id"],
                DocumentChunk.chunk_index.in_([chunk_index - 1, chunk_index, chunk_index + 1]),
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        combined = "\n\n".join(chunk.content or "" for chunk in neighbors if chunk.content)
        if combined:
            item["content"] = combined

    return ranked


def _source_timestamp(source: Dict, key: str) -> str:
    value = source.get(key)
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else "unknown"


def _format_context_source(idx: int, source: Dict) -> str:
    match_label = "direct topic match" if source.get("direct_match", True) else "related source; topic may be absent"
    retrieval_source = source.get("retrieval_source", "lexical")
    semantic_score = source.get("semantic_score")
    semantic_label = f"\nSemantic score: {semantic_score:.3f}" if isinstance(semantic_score, (int, float)) else ""
    evidence_count = source.get("evidence_chunk_count")
    evidence_label = f"\nEvidence chunks in this source: {evidence_count}" if evidence_count else ""
    return (
        f"Source {idx}: {source['title']} "
        f"({source['doc_type']}, chunk {source['chunk_index']})\n"
        f"Document ID: {source['document_id']}\n"
        f"Created: {_source_timestamp(source, 'created_at')}\n"
        f"Updated: {_source_timestamp(source, 'updated_at')}\n"
        f"Retrieval: {retrieval_source}\n"
        f"Match type: {match_label}\n"
        f"{semantic_label}\n"
        f"{evidence_label}\n"
        f"{source['content'][:MAX_CONTEXT_CHARS_PER_SOURCE]}"
    )


async def answer_question(
    db: Session,
    question: str,
    *,
    doc_type: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict:
    guardrail = detect_prompt_injection(question)
    if not guardrail.allowed:
        return {
            "status": "blocked",
            "answer": "Question blocked because it looks like a prompt-injection attempt.",
            "sources": [],
            "reasons": guardrail.reasons,
        }

    if _is_inventory_question(question):
        return _answer_inventory_question(db)

    top_k = max(1, min(top_k, DEFAULT_TOP_K))
    cache_key = (question.strip().lower(), doc_type or "All", top_k, _document_signature(db))
    cached = _qa_cache.get(cache_key)
    if cached and datetime.utcnow() - cached[0] < QA_CACHE_TTL:
        return cached[1]

    sources = retrieve_db_context(db, question, doc_type=doc_type, top_k=top_k)
    if not sources:
        return {
            "status": "success",
            "answer": "I could not find relevant saved document context for that question.",
            "sources": [],
            "reasons": [],
        }

    context = "\n\n".join(_format_context_source(idx, source) for idx, source in enumerate(sources, start=1))

    llm = get_chat_model(temperature=0.1, max_tokens=800, mode="qa")
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You answer questions about saved project documents. "
                    "Use only the provided context. If the context is insufficient, say so. "
                    "Do not invent dates, ages, relationships, or chronology. "
                    "Do not combine timeline or age facts from separate sources unless one source "
                    "explicitly supports that exact relationship. If source details conflict or the "
                    "timeline is unclear, say that the context is inconsistent or insufficient. "
                    "When multiple sources or versions answer differently, give separate short "
                    "answers per source/version instead of forcing one merged answer. "
                    "Each source may contain multiple evidence chunks from the same saved "
                    "document/version; judge that source using all of its evidence chunks together. "
                    "If a source is only related and does not state the requested detail, say that "
                    "the detail is not stated in that source. "
                    "Cite sources as [Source 1], [Source 2], etc."
                )
            ),
            HumanMessage(
                content=f"Question:\n{question}\n\nRetrieved context:\n{context}"
            ),
        ]
    )

    answer = response.content if hasattr(response, "content") else str(response)
    result = {
        "status": "success",
        "answer": answer,
        "sources": sources,
        "reasons": [],
    }
    _qa_cache[cache_key] = (datetime.utcnow(), result)
    return result
