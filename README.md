# DocAssist

DocAssist is a GenAI-powered document generation and management platform for creating, reviewing, organizing, and querying SRS, SOW, proposal, technical, and business documents with an agentic RAG backend.

The app lets you upload historical documents, generate new documents from project requirements, review/edit the generated Markdown, and download saved documents.

## What It Does

- Uploads PDF, DOCX, TXT, and Markdown documents, or imports direct PDF links.
- Extracts text and style signals from uploaded documents.
- Chunks and indexes uploaded content for retrieval.
- Generates new documents using a multi-agent workflow.
- Reviews and improves documents with AI-assisted editing.
- Answers questions over saved document chunks with a RAG chat page.
- Generates deterministic document quality reports with completeness score, missing sections, vague terms, and improvement suggestions.
- Keeps document version snapshots before saves/reviews/restores, with restore support in the UI.
- Rebuilds SQLite retrieval chunks through the Maintenance page or MCP reindex tools.
- Lets Claude inspect raw matching snippets through the `search_documents` MCP tool.
- Lets Claude import lawful direct PDF download links through the `import_pdf_from_url` MCP tool.
- Supports local Ollama, Groq, Gemini, Cerebras, NVIDIA NIM, or mock model fallback through one shared model factory.
- Blocks common prompt-injection style instructions in generation, review, and chat inputs.
- Stores documents, chunks, versions, workflow history, and feedback in SQLite.
- Provides a Streamlit UI for generation, upload, editing, library browsing, and Markdown export.

## Backend Agents

- `DocumentIngestionAgent`: parses, chunks, stores, and indexes uploaded documents.
- `StyleProfileBuilderAgent`: learns tone and structure from approved documents.
- `RetrieverAgent`: retrieves relevant historical context using Pinecone when configured, with FAISS fallback.
- `DocGenerationAgent`: generates draft documents using Groq when configured, with a mock fallback.
- `ReviewEditingAgent`: improves formatting, style, and feedback-driven edits.
- `AgenticRAGWorkflow`: orchestrates the generation flow with LangGraph.
- `document_maintenance.py`: shared helpers for version snapshots, restore, reindexing, and raw lexical search.
- `quality_report.py`: deterministic document quality scoring and Markdown report formatting.

## Implemented GenAI Concepts

- LangChain-compatible LLM abstraction through `src/backend/llm_factory.py`.
- Local Ollama support, including `llama3.1:8b`.
- Groq, Gemini, Cerebras, and NVIDIA NIM API support for online models.
- RAG document Q&A in the Streamlit `Ask Documents` workspace.
- SQLite-backed chunk retrieval for chat, so saved context survives app restarts.
- Full-document fallback for saved documents that do not yet have chunks.
- Source-aware Ask Documents answers that can cite multiple documents/versions
  separately when saved sources disagree.
- MCP tools for Claude: generation, RAG Q&A, document listing, exact document reading, raw snippet search, direct PDF URL import, and safe reindexing.
- Version history and restore flow for safer AI-assisted editing.
- Prompt-injection detection with lightweight regex guardrails.
- Streamlit session-based chat history.

## Tech Stack

- Streamlit
- FastAPI backend modules
- SQLAlchemy with SQLite
- LangChain and LangGraph
- Groq, Gemini, Cerebras, and NVIDIA NIM LLM integration
- Hugging Face sentence-transformer embeddings
- Pinecone or FAISS vector storage

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root, `src/`, or `src/backend/`.

```env
OLLAMA_BASE_URL=http://localhost:11434

GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
CEREBRAS_API_KEY=your_cerebras_key
NVIDIA_API_KEY=your_nvidia_key
# Optional for self-hosted NVIDIA NIM endpoints:
# NVIDIA_NIM_BASE_URL=http://localhost:8000/v1

PINECONE_API_KEY=your_pinecone_key
DATABASE_URL=sqlite:///./agentic_rag.db
```

Choose the active LLM in `src/backend/llm_factory.py` by uncommenting exactly one line:

```python
# ACTIVE_MODEL_CONFIG = use_ollama("llama3.1:8b")
ACTIVE_MODEL_CONFIG = use_groq("openai/gpt-oss-120b")
# ACTIVE_MODEL_CONFIG = use_groq("openai/gpt-oss-20b")
# ACTIVE_MODEL_CONFIG = use_gemini("gemini-3.5-flash")
# ACTIVE_MODEL_CONFIG = use_cerebras("gpt-oss-120b")
# ACTIVE_MODEL_CONFIG = use_nvidia_nim("nemotron-3-super-120b-a12b")
# ACTIVE_MODEL_CONFIG = use_groq("llama-3.3-70b-versatile")
# ACTIVE_MODEL_CONFIG = use_mock()
```

Provider API keys stay in `.env`; the selected model is the string passed to the active helper such as `use_groq(...)`, `use_cerebras(...)`, or `use_nvidia_nim(...)`. Without `PINECONE_API_KEY`, vector retrieval falls back to local FAISS. The chat page also retrieves from SQLite document chunks, so it can answer from saved documents even after a restart. Ask Documents diversifies retrieved sources, includes neighboring chunks, and instructs the model to answer separately per source/version when facts conflict instead of merging them silently.

## Document Type Behavior

On the Generate page, `Report`, `Proposal`, `User Guide`, `Technical Spec`, `SRS`, `SOW`, `Business Plan`, `Story`, and `Creative Brief` are backed by preset guidance in `src/backend/document_presets.py`. These are not separate generators, but each preset provides stronger structure, tone guidance, output rules, and a quality checklist.

The selected preset affects:

- the generation prompt: the LLM receives preset-specific sections, tone, rules, and checklist
- retrieval filtering: reference chunks are searched using the same document type
- style learning: approved documents of the same type are preferred for style profile building
- saved metadata: generated documents are stored with that type
- compliance checks: required sections come from the same preset definitions
- quality reports: missing-section checks use the same preset definitions

Current preset section examples:

```text
SRS -> Introduction, Overall Description, Functional Requirements, Non-Functional Requirements, External Interfaces, Acceptance Criteria, Assumptions, Constraints
SOW -> Scope, Objectives, Deliverables, Timeline, Roles and Responsibilities, Acceptance Criteria, Assumptions, Out of Scope
Technical Spec -> Overview, Goals, Architecture, Data Model, API / Interfaces, Implementation Plan, Security, Testing, Risks
Business Plan -> Executive Summary, Market Analysis, Customer Segments, Product / Service, Business Model, Go-To-Market Strategy, Operations, Financial Plan, Risks
Story -> Title, Premise, Characters, Setting, Story, Ending
Creative Brief -> Project Overview, Objective, Audience, Key Message, Tone and Style, Deliverables, Channels, Success Metrics
Fallback -> Introduction, Content, Conclusion
```

Aliases such as `Technical` -> `Technical Spec` and `Business` -> `Business Plan` are normalized so older labels do not fall back to generic checks.

For your local Ollama setup, make sure Ollama is running:

```bash
ollama serve
ollama pull llama3.1:8b
```

## Run

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Project Structure

```text
streamlit_app.py          Streamlit user interface
mcp_server.py             Claude/MCP tool server
requirements.txt          Python dependencies
src/backend/              Backend agents, database, workflow, and API modules
src/backend/agents/       Agent implementations
src/backend/rag_qa.py     SQLite-backed document Q&A
src/backend/url_importer.py          Direct PDF URL import helper
src/backend/document_maintenance.py  Versioning, reindexing, raw search
src/backend/quality_report.py        Document quality report helpers
```

## Notes

The old React/Vite frontend has been removed. The Python backend remains because the Streamlit app reuses the existing agent, database, workflow, direct PDF import, and document-processing code directly.

After changing `mcp_server.py`, restart Claude Desktop or kill the old MCP process so Claude reloads the latest tools:

```bash
ps aux | grep '[m]cp_server.py'
pkill -f mcp_server.py
```
