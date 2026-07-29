import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.backend.database import SessionLocal, check_database_health, create_tables
from src.backend.models import Document
from src.backend.quality_report import generate_quality_report, format_quality_report_markdown


st.set_page_config(
    page_title="DocAssist",
    layout="wide",
)


def run_async(coro):
    return asyncio.run(coro)


@st.cache_resource
def initialize_database():
    create_tables()
    return check_database_health()


def get_documents():
    db = SessionLocal()
    try:
        return db.query(Document).order_by(Document.updated_at.desc()).all()
    finally:
        db.close()


def get_document(document_id):
    db = SessionLocal()
    try:
        return db.query(Document).filter(Document.id == document_id).first()
    finally:
        db.close()


def save_document(document_id, title, content, status, feedback_score):
    db = SessionLocal()
    try:
        from src.backend.document_maintenance import create_document_version, reindex_document

        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False

        changed = (
            document.title != title
            or (document.content or "") != (content or "")
            or document.status != status
            or int(document.feedback_score or 3) != int(feedback_score)
        )
        if changed:
            create_document_version(db, document, change_note="Snapshot before Streamlit save")

        document.title = title
        document.content = content
        document.status = status
        document.feedback_score = max(1, min(5, int(feedback_score)))
        document.updated_at = datetime.utcnow()
        db.commit()
        reindex_document(db, document_id, add_to_vector_store=False)
        return True
    finally:
        db.close()


def delete_document(document_id):
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            db.delete(document)
            db.commit()
    finally:
        db.close()


def create_generated_document(doc_type, summary, content, feedback_score):
    db = SessionLocal()
    try:
        from src.backend.document_maintenance import reindex_document

        document = Document(
            title=f"{doc_type}: {summary}",
            filename=f"{doc_type.lower()}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.md",
            doc_type=doc_type,
            content=content,
            status="final",
            approved=True,
            feedback_score=max(1, min(5, int(feedback_score))),
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        reindex_document(db, document.id, add_to_vector_store=False)
        return document.id
    finally:
        db.close()


def get_document_versions(document_id, limit=10):
    db = SessionLocal()
    try:
        from src.backend.document_maintenance import list_document_versions

        return list_document_versions(db, document_id, limit=limit)
    finally:
        db.close()


def restore_version(version_id):
    db = SessionLocal()
    try:
        from src.backend.document_maintenance import restore_document_version

        return restore_document_version(db, version_id)
    finally:
        db.close()


def document_options(documents):
    return {
        f"{doc.title} ({doc.doc_type}) - {doc.updated_at.strftime('%Y-%m-%d %H:%M')}": doc.id
        for doc in documents
    }


def render_upload():
    st.header("Upload Documents")
    st.caption(
        "Upload existing documents (reports, manuals, specifications, guidelines) for retrieval "
        "and style learning. You can upload a local file or import a direct PDF link."
    )

    col_x, col_y = st.columns([1, 1])
    with col_x:
        doc_type = st.text_input("Document category/type", value="Reference", help="e.g., Guide, Policy, SRS")
    with col_y:
        feedback_score = st.slider("Quality score", min_value=1, max_value=5, value=3)
        
    approved = st.checkbox("Use this document for style/context learning", value=True)

    upload_tab, url_tab = st.tabs(["Local file", "Direct PDF link"])

    with upload_tab:
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt", "md"])

        if st.button("Process Document", type="primary", disabled=uploaded_file is None):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            db = SessionLocal()
            try:
                with st.spinner("Parsing, chunking, and indexing the document..."):
                    from src.backend.agents.DocumentIngestionAgent import DocumentIngestionAgent

                    agent = DocumentIngestionAgent()
                    result = run_async(
                        agent.execute(
                            db=db,
                            filename=uploaded_file.name,
                            file_path=temp_path,
                            doc_type=doc_type.strip() if doc_type.strip() else "Reference",
                            approved=approved,
                            feedback_score=feedback_score,
                        )
                    )
                if result.get("status") == "success":
                    st.success("Document processed successfully.")
                    st.code(result.get("document_id", ""), language="text")
                    st.session_state["selected_document_id"] = result.get("document_id")
                else:
                    st.error(result.get("message", "Document processing failed."))
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
            finally:
                db.close()
                Path(temp_path).unlink(missing_ok=True)

    with url_tab:
        pdf_url = st.text_input(
            "Direct PDF URL",
            placeholder="https://example.edu/reports/reference.pdf",
        )
        title_override = st.text_input(
            "Optional title override",
            placeholder="Leave blank to use the PDF filename",
        )

        if st.button("Import PDF Link", type="primary", disabled=not pdf_url.strip()):
            db = SessionLocal()
            try:
                with st.spinner("Downloading, parsing, chunking, and indexing the PDF..."):
                    from src.backend.url_importer import import_pdf_from_url

                    result = run_async(
                        import_pdf_from_url(
                            db,
                            pdf_url,
                            doc_type=doc_type.strip() if doc_type.strip() else "Reference",
                            title=title_override.strip(),
                            approved=approved,
                            feedback_score=feedback_score,
                        )
                    )
                if result.get("status") == "success":
                    st.success("PDF imported successfully.")
                    st.code(result.get("document_id", ""), language="text")
                    st.session_state["selected_document_id"] = result.get("document_id")
                    metadata = result.get("metadata", {})
                    st.caption(
                        f"Imported {metadata.get('word_count', 0)} words into "
                        f"{result.get('chunk_count', 0)} chunks."
                    )
                else:
                    st.error(result.get("message", "PDF import failed."))
            except Exception as exc:
                st.error(f"PDF import failed: {exc}")
            finally:
                db.close()


def render_generate():
    st.header("Generate Document")

    with st.form("generate_form"):
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            col_preset, col_custom = st.columns([1.2, 1.2])
            with col_preset:
                doc_type_preset = st.selectbox(
                    "Document type preset",
                    ["Report", "Proposal", "User Guide", "Technical Spec", "SRS", "SOW", "Business Plan", "Story", "Creative Brief"],
                )
            with col_custom:
                doc_type_custom = st.text_input("Or specify custom type", placeholder="e.g. Policy, Manual")
                doc_type = doc_type_custom.strip() if doc_type_custom.strip() else doc_type_preset
        with col_b:
            style = st.selectbox("Style", ["professional", "formal", "technical", "concise"])

        summary = st.text_input("Document Topic / Goal")
        requirements = st.text_area("Key Outline / Sections / Requirements", height=220)
        feedback_score = st.slider("Initial quality score", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Generate Document", type="primary")

    if submitted:
        from src.backend.guardrails import check_inputs

        if not summary.strip() and not requirements.strip():
            st.warning("Add a topic or key points before generating.")
            return
        guardrail = check_inputs([summary, requirements])
        if not guardrail.allowed:
            st.error("Blocked: the input looks like a prompt-injection attempt.")
            return

        db = SessionLocal()
        try:
            with st.spinner("Running the agent workflow..."):
                from src.backend.workflow import workflow_manager

                result = run_async(
                    workflow_manager.execute_generation_workflow(
                        doc_type=doc_type,
                        summary=summary,
                        requirements=requirements,
                        style=style,
                        db=db,
                        max_iterations=3,
                    )
                )

            if result.get("status") == "failed":
                st.error(result.get("error", "Generation failed."))
                return

            content = result.get("content", "")
            document_id = result.get("document_id")
            title = f"{doc_type}: {summary or 'Untitled'}"
            if not document_id or not save_document(document_id, title, content, "final", feedback_score):
                document_id = create_generated_document(
                    doc_type,
                    summary or "Untitled",
                    content,
                    feedback_score,
                )
            st.success("Document generated and saved.")
            st.session_state["selected_document_id"] = document_id
            with st.expander("Document Quality Report", expanded=True):
                report = generate_quality_report(content, doc_type)
                st.markdown(format_quality_report_markdown(report))
            st.markdown(content)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
        finally:
            db.close()


def render_review_edit():
    st.header("Review / Edit")
    documents = get_documents()
    if not documents:
        st.info("No documents yet. Generate or upload one first.")
        return

    options = document_options(documents)
    current_id = st.session_state.get("selected_document_id")
    labels = list(options.keys())
    index = 0
    if current_id in options.values():
        index = list(options.values()).index(current_id)

    selected_label = st.selectbox("Document", labels, index=index)
    document_id = options[selected_label]
    document = get_document(document_id)
    if not document:
        st.warning("Document not found.")
        return

    title = st.text_input("Title", value=document.title)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        status_values = ["draft", "review", "final"]
        current_status = document.status if document.status in status_values else "draft"
        status = st.selectbox(
            "Status",
            status_values,
            index=status_values.index(current_status),
        )
    with col_b:
        feedback_score = st.slider("Quality score", min_value=1, max_value=5, value=int(document.feedback_score or 3))

    content = st.text_area("Markdown content", value=document.content or "", height=520)
    feedback = st.text_area(
        "Optional review instructions",
        placeholder="Example: tighten the non-functional requirements and add acceptance criteria.",
    )

    with st.expander("Document Quality Report"):
        report = generate_quality_report(content, document.doc_type)
        st.markdown(format_quality_report_markdown(report))

    with st.expander("Version History"):
        versions = get_document_versions(document_id, limit=10)
        if not versions:
            st.caption("No previous versions yet. Versions are created before saves, AI reviews, and restores.")
        for version in versions:
            label = f"{version.created_at.strftime('%Y-%m-%d %H:%M:%S')} - {version.change_note or 'Snapshot'}"
            with st.container():
                st.markdown(f"**{label}**")
                st.caption(f"Title: {version.title} | Status: {version.status} | Score: {version.feedback_score}")
                col_prev, col_restore = st.columns([3, 1])
                with col_prev:
                    st.code((version.content or "")[:600], language="markdown")
                with col_restore:
                    if st.button("Restore", key=f"restore_review_{version.id}"):
                        result = restore_version(version.id)
                        if result.get("status") == "success":
                            st.success("Version restored and reindexed.")
                            st.rerun()
                        else:
                            st.error(result.get("message", "Restore failed."))

    col_save, col_review = st.columns([1, 1])
    with col_save:
        if st.button("Save Changes", type="primary"):
            if save_document(document_id, title, content, status, feedback_score):
                st.success("Document saved.")
            else:
                st.error("Unable to save document.")

    with col_review:
        if st.button("AI Review / Improve"):
            from src.backend.guardrails import check_inputs

            guardrail = check_inputs([feedback])
            if not guardrail.allowed:
                st.error("Blocked: the review instructions look like a prompt-injection attempt.")
                return

            try:
                with st.spinner("Reviewing document..."):
                    from src.backend.agents.ReviewEditingAgent import ReviewEditingAgent

                    agent = ReviewEditingAgent()
                    result = run_async(
                        agent.execute(
                            content=content,
                            doc_type=document.doc_type,
                            style_profile=document.style_metadata or {},
                            feedback=[feedback] if feedback.strip() else [],
                            review_type="both" if feedback.strip() else "formatting",
                            feedback_score=feedback_score,
                        )
                    )

                improved = result.get("improved_content", content)
                if result.get("status") != "success":
                    st.error(result.get("message", "Review failed."))
                    return
                if feedback.strip() and improved.strip() == content.strip():
                    st.warning("Review completed, but no changes were returned. Try a more specific instruction or a shorter selected document.")
                    return
                save_document(document_id, title, improved, status, feedback_score)
                st.success("Review applied and saved.")
                st.markdown(improved)
            except Exception as exc:
                st.error(f"Review failed: {exc}")


def render_chat():
    st.header("Ask Documents")
    st.caption("Ask questions over saved document chunks. Answers cite retrieved sources.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    doc_types = ["All"] + sorted({doc.doc_type for doc in get_documents()})
    col_a, col_b = st.columns([1, 1])
    with col_a:
        doc_type = st.selectbox("Scope", doc_types)
    with col_b:
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=3, value=3)

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Inline form for chat input to keep it visible inside the main workspace container
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Question",
                placeholder="Ask about requirements, risks, scope, or document details...",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if not (submitted and question):
        return

    st.session_state["chat_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    db = SessionLocal()
    try:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and answering..."):
                from src.backend.rag_qa import answer_question

                result = run_async(
                    answer_question(
                        db,
                        question,
                        doc_type=doc_type,
                        top_k=top_k,
                    )
                )

            answer = result.get("answer", "")
            st.markdown(answer)
            sources = result.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for idx, source in enumerate(sources, start=1):
                        updated_at = source.get("updated_at")
                        updated_label = updated_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(updated_at, "strftime") else "unknown"
                        match_label = "direct match" if source.get("direct_match", True) else "related source"
                        retrieval_label = source.get("retrieval_source", "lexical")
                        st.markdown(
                            f"**Source {idx}:** {source['title']} "
                            f"({source['doc_type']}, chunk {source['chunk_index']})"
                        )
                        st.caption(
                            f"Updated: {updated_label} | Match: {match_label} | "
                            f"Retrieval: {retrieval_label} | Document ID: {source.get('document_id', 'unknown')}"
                        )
                        st.caption(source["content"][:700])

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": answer}
        )
    finally:
        db.close()


def render_library():
    st.header("Document Library")
    documents = get_documents()
    if not documents:
        st.info("No documents have been saved yet.")
        return

    options = document_options(documents)
    selected_label = st.selectbox("Document", list(options.keys()), key="library_document")
    document = get_document(options[selected_label])
    if not document:
        st.warning("Document not found.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Type", document.doc_type)
    col_b.metric("Status", document.status)
    col_c.metric("Words", len((document.content or "").split()))
    col_d.metric("Score", document.feedback_score)

    st.subheader(document.title)
    with st.expander("Document Quality Report", expanded=False):
        report = generate_quality_report(document.content or "", document.doc_type)
        st.markdown(format_quality_report_markdown(report))

    with st.expander("Version History", expanded=False):
        versions = get_document_versions(document.id, limit=10)
        if not versions:
            st.caption("No previous versions yet.")
        for version in versions:
            st.markdown(f"**{version.created_at.strftime('%Y-%m-%d %H:%M:%S')}** - {version.change_note or 'Snapshot'}")
            st.caption(f"Status: {version.status} | Score: {version.feedback_score}")
            st.code((version.content or "")[:500], language="markdown")
            if st.button("Restore this version", key=f"restore_library_{version.id}"):
                result = restore_version(version.id)
                if result.get("status") == "success":
                    st.success("Version restored and reindexed.")
                    st.rerun()
                else:
                    st.error(result.get("message", "Restore failed."))

    st.markdown(document.content or "")

    file_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in document.title)[:80] or "document"
    st.download_button(
        "Download Markdown",
        data=document.content or "",
        file_name=f"{file_stem}.md",
        mime="text/markdown",
    )

    if st.button("Delete Document", type="secondary"):
        delete_document(document.id)
        st.success("Document deleted.")
        st.rerun()


def render_maintenance():
    st.header("Maintenance")
    st.caption("Repair retrieval chunks and inspect indexing health without changing document content.")

    documents = get_documents()
    if not documents:
        st.info("No documents have been saved yet.")
        return

    from src.backend.models import DocumentChunk
    from src.backend.document_maintenance import reindex_all_documents, reindex_document

    db = SessionLocal()
    try:
        total_chunks = db.query(DocumentChunk).count()
        unchunked = []
        for document in documents:
            chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).count()
            if chunk_count == 0:
                unchunked.append(document)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Documents", len(documents))
        col_b.metric("Chunks", total_chunks)
        col_c.metric("Unchunked Docs", len(unchunked))

        if unchunked:
            with st.expander("Unchunked documents", expanded=True):
                for document in unchunked:
                    st.markdown(f"- `{document.id}` - {document.title}")

        options = document_options(documents)
        selected_label = st.selectbox("Document to reindex", list(options.keys()))
        selected_id = options[selected_label]

        col_one, col_all = st.columns(2)
        with col_one:
            if st.button("Reindex Selected Document", type="primary"):
                result = reindex_document(db, selected_id, add_to_vector_store=False)
                if result.get("status") == "success":
                    st.success(f"Reindexed {result['chunk_count']} chunks.")
                    if result.get("vector_warning"):
                        st.warning(result["vector_warning"])
                    st.rerun()
                else:
                    st.error(result.get("message", "Reindex failed."))
        with col_all:
            if st.button("Reindex All Documents"):
                result = reindex_all_documents(db, add_to_vector_store=False)
                st.success(f"Reindexed {result['document_count']} documents with {result['total_chunks']} chunks.")
                st.rerun()
    finally:
        db.close()


def main():
    db_ok = initialize_database()

    st.title("DocAssist")
    st.caption("GenAI-powered document generation and management.")
    if not db_ok:
        st.warning("Database health check failed. The app may need migrations or DEV_MODE=true for a fresh local reset.")

    page = st.sidebar.radio(
        "Workspace",
        ["Generate", "Upload", "Review / Edit", "Ask Documents", "Library", "Maintenance"],
    )

    if page == "Generate":
        render_generate()
    elif page == "Upload":
        render_upload()
    elif page == "Review / Edit":
        render_review_edit()
    elif page == "Ask Documents":
        render_chat()
    elif page == "Library":
        render_library()
    else:
        render_maintenance()


if __name__ == "__main__":
    main()
