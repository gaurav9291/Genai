"""Review Editing Agent - Uses Pydantic structured output (with_structured_output) for robust
extraction of review results from the LLM. Falls back to plain-text when the
LLM does not support schema enforcement."""

from typing import Any, Dict, List, Optional
import re
import uuid
import asyncio
import tenacity
import logging

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.prompts import PromptTemplate

from .base_agent import BaseAgent
from .DocumentIngestionAgent import DocumentIngestionAgent
from ..config import settings
from ..guardrails import check_inputs
from ..llm_factory import get_chat_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema enforced via .with_structured_output()
# ---------------------------------------------------------------------------

class ReviewedDocument(BaseModel):
    """Structured output schema for the Review/Editing LLM call."""

    improved_content: str = Field(
        description="The fully revised document content in proper Markdown format."
    )
    changes_summary: List[str] = Field(
        default_factory=list,
        description="A concise bullet-point list of every change that was made."
    )
    quality_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Estimated quality score of the improved document, from 1 (poor) to 5 (excellent)."
    )
    approved: bool = Field(
        default=False,
        description="True if the improved document meets professional publication standards."
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReviewEditingAgent(BaseAgent):
    """Review Editing Agent that uses .with_structured_output() to guarantee
    typed, validated results from the LLM instead of ad hoc string parsing."""

    def __init__(self):
        super().__init__(
            name="review_editing",
            description="Enhances formatting and style; returns a validated ReviewedDocument."
        )
        raw_llm = get_chat_model(
            temperature=0.1,
            max_tokens=max(settings.max_tokens, 4096),
            mode="review",
        )

        # Bind schema — ChatGroq, ChatOllama, and MockLLM all support this API
        try:
            self.structured_llm = raw_llm.with_structured_output(ReviewedDocument)
            self._use_structured = True
            logger.info("ReviewEditingAgent: using structured output (Pydantic schema).")
        except (AttributeError, NotImplementedError):
            self.structured_llm = None
            self._use_structured = False
            logger.warning("ReviewEditingAgent: LLM does not support structured output, falling back to text mode.")

        # Fallback plain-text LLM (used when structured output is not available)
        self.raw_llm = raw_llm

        import difflib
        self.difflib = difflib

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_structured_prompt(self, content: str, style_text: str, feedback_text: str) -> str:
        base = (
            "You are an expert technical editor. Review and improve the document below. "
            "When additional feedback is provided, you must apply it in the revised document.\n\n"
            f"STYLE REQUIREMENTS:\n{style_text}\n\n"
            f"DOCUMENT CONTENT:\n{content}\n\n"
        )
        if feedback_text:
            base += f"ADDITIONAL FEEDBACK TO ADDRESS:\n{feedback_text}\n\n"
        base += (
            "Return a JSON object matching this schema exactly:\n"
            "{\n"
            '  "improved_content": "<full revised markdown>",\n'
            '  "changes_summary": ["change 1", "change 2"],\n'
            '  "quality_score": <int 1-5>,\n'
            '  "approved": <true|false>\n'
            "}\n"
            "Return ONLY the JSON — no extra text."
        )
        return base

    def _format_style_profile(self, style_profile: Dict[str, Any]) -> str:
        if not style_profile:
            return "Professional technical writing style with clear structure."
        parts = []
        if "tone_analysis" in style_profile:
            tone = style_profile["tone_analysis"]
            dominant = max(tone, key=tone.get) if tone else "professional"
            parts.append(f"Primary tone: {dominant}")
        if "heading_patterns" in style_profile:
            headings = style_profile["heading_patterns"]
            preferred = max(headings, key=headings.get) if headings else "hash_headers"
            parts.append(f"Heading style: {preferred}")
        return "; ".join(parts) if parts else "Professional technical writing style."

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _post_process(self, content: str) -> str:
        lines, out = content.split("\n"), []
        blank = 0
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if out and out[-1].strip():
                    out.append("")
                out.append(line)
                if i < len(lines) - 1 and lines[i + 1].strip():
                    out.append("")
            else:
                if not line.strip():
                    blank += 1
                    if blank <= 2:
                        out.append(line)
                else:
                    blank = 0
                    out.append(line)
        result = "\n".join(out)
        result = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", result)
        return result.strip()

    def _diff_summary(self, original: str, improved: str) -> Dict[str, Any]:
        orig_lines = original.split("\n")
        new_lines = improved.split("\n")
        diff = list(self.difflib.unified_diff(orig_lines, new_lines, lineterm="", n=2))
        if len(diff) > 2:
            diff = diff[3:]
        removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        return {"removed": removed, "added": added, "unified_diff": diff[:100]}

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    async def execute(
        self,
        content: str,
        doc_type: str = "document",
        style_profile: Optional[Dict[str, Any]] = None,
        feedback: Optional[List[str]] = None,
        review_type: str = "formatting",
        approved: bool = False,
        feedback_score: int = 3,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Run the review pipeline and return a validated result dict."""
        logger.info(
            "ReviewEditingAgent.execute(): type=%s, doc_type=%s, feedback_items=%d, content_length=%d",
            review_type,
            doc_type,
            len(feedback or []),
            len(content),
        )

        # Guardrail check
        guardrail = check_inputs(feedback or [])
        if not guardrail.allowed:
            return {
                "status": "error",
                "message": "Feedback blocked: looks like a prompt-injection attempt.",
                "improved_content": content,
                "changes_made": [],
                "original_word_count": len(content.split()),
                "final_word_count": len(content.split()),
            }

        style_text = self._format_style_profile(style_profile or {})
        feedback_text = (
            "\n".join(f"- {item}" for item in feedback) if feedback else ""
        )

        try:
            final_content = None
            changes_made = []
            quality_score = feedback_score

            if self._use_structured and self.structured_llm is not None:
                # ── Structured path ──────────────────────────────────────────
                logger.debug("Executing review with Pydantic structured output.")
                prompt = self._build_structured_prompt(content, style_text, feedback_text)
                try:
                    loop = asyncio.get_running_loop()
                    result: ReviewedDocument = await loop.run_in_executor(
                        None, lambda: self.structured_llm.invoke(prompt)
                    )
                    if result is None or not getattr(result, "improved_content", None):
                        raise ValueError("Structured review returned no improved content")

                    final_content = self._post_process(result.improved_content)
                    changes_made = result.changes_summary or ["Applied formatting improvements"]
                    quality_score = result.quality_score or feedback_score
                    logger.debug("Structured review complete. Score: %s, approved: %s", quality_score, result.approved)
                except Exception as structured_exc:
                    logger.warning("Structured review failed, falling back to text mode: %s", structured_exc)

            if final_content is None:
                # ── Plain-text fallback path ──────────────────────────────────
                logger.debug("Executing review with plain-text fallback chain.")
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import PromptTemplate

                tmpl = PromptTemplate.from_template(
                    "You are an expert editor. Review and improve this document.\n\n"
                    "Style: {style_profile}\n\nContent:\n{content}\n\n"
                    "{feedback_section}"
                    "Apply every feedback item. Return ONLY the complete improved Markdown content."
                )
                chain = tmpl | self.raw_llm | StrOutputParser()
                feedback_section = (
                    f"Feedback to address:\n{feedback_text}\n\n" if feedback_text else ""
                )
                improved = await chain.ainvoke({
                    "content": content,
                    "style_profile": style_text,
                    "feedback_section": feedback_section,
                })
                final_content = self._post_process(improved)
                changes_made = ["Applied formatting improvements (text mode)"]
                quality_score = feedback_score

        except Exception as exc:
            logger.warning("ReviewEditingAgent LLM call failed: %s", exc)
            return {
                "status": "error",
                "message": f"Review LLM call failed: {exc}",
                "document_id": None,
                "improved_content": content,
                "changes_made": [],
                "diff_details": {"removed": [], "added": [], "unified_diff": []},
                "quality_score": feedback_score,
                "original_word_count": len(content.split()),
                "final_word_count": len(content.split()),
            }

        # Optionally ingest the reviewed document back into the vector store
        document_id = None
        if approved and db_session:
            try:
                ingestion_agent = DocumentIngestionAgent()
                ingest_result = await ingestion_agent.execute(
                    db=db_session,
                    filename=f"reviewed_{doc_type}_{uuid.uuid4().hex[:8]}.md",
                    content=final_content,
                    doc_type=doc_type,
                    approved=approved,
                    feedback_score=quality_score,
                )
                document_id = ingest_result.get("document_id")
            except Exception as exc:
                logger.warning("Post-review ingestion failed: %s", exc)

        diff_details = self._diff_summary(content, final_content)

        return {
            "status": "success",
            "document_id": document_id,
            "improved_content": final_content,
            "changes_made": changes_made,
            "diff_details": diff_details,
            "quality_score": quality_score,
            "original_word_count": len(content.split()),
            "final_word_count": len(final_content.split()),
        }
