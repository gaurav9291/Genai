"""Document Generation Agent - Production-ready implementation with LangGraph and RLHF integration"""

from typing import Dict, Any, List, Optional, TypedDict
from sqlalchemy.orm import Session
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import re
import uuid
import asyncio
import tenacity
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .RetrieverAgent import RetrieverAgent
from .StyleProfileBuilderAgent import StyleProfileBuilderAgent
from .ReviewEditingAgent import ReviewEditingAgent
from .DocumentIngestionAgent import DocumentIngestionAgent
from ..config import settings
from ..document_presets import format_preset_for_prompt
from ..guardrails import check_inputs
from ..llm_factory import get_chat_model

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State for the Langgraph workflow with proper typing"""
    doc_type: str
    summary: str
    requirements: str
    context: List[Dict[str, Any]]
    style_profile: Dict[str, Any]
    approved: bool
    feedback_score: int
    db_session: Optional[Session]
    generated_content: str
    final_content: str

class DocGenerationAgent(BaseAgent):
    """Production-ready Document Generation Agent with LangGraph workflows"""
    
    def __init__(self):
        super().__init__(name="doc_generation", description="Generates draft SRS documents based on user input and prior examples")
        self.llm = get_chat_model(
            temperature=settings.temperature,
            max_tokens=max(settings.max_tokens, 4096),
            mode="document",
        )
        
        # Create prompt templates for different document types
        self.srs_template = PromptTemplate.from_template(
            "You are an expert technical writer and document generator. "
            "Generate a comprehensive and detailed {doc_type} document based on the following information:\n\n"
            "Document Subject/Goal: {summary}\n"
            "Key Points/Outline: {requirements}\n"
            "Style Profile: {style_profile}\n"
            "Reference Context: {context}\n\n"
            "Preset Guidance:\n{preset_guidance}\n\n"
            "Follow the preset guidance closely unless the user's outline explicitly asks for a different structure. "
            "Ensure the document follows professional standards with proper sections, formatting, and technical accuracy. "
            "Avoid generic filler; make every section specific to the subject/goal and key points. "
            "Use markdown formatting with appropriate headers, lists, tables, and code blocks where necessary."
        )
        
        self.generation_chain = self.srs_template | self.llm | StrOutputParser()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for document generation"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("build_style_profile", self._build_style_profile)
        workflow.add_node("generate_draft", self._generate_draft)
        
        # Set up edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "build_style_profile")
        workflow.add_edge("build_style_profile", "generate_draft")
        workflow.add_edge("generate_draft", END)
        
        return workflow.compile()
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context data for prompt"""
        if not context:
            return "No relevant examples found."
        
        formatted_chunks = []
        for i, chunk in enumerate(context[:3]):  # Limit to top 3 examples
            content = chunk.get('content', '').strip()
            if content:
                formatted_chunks.append(f"Example {i+1}:\n{content[:500]}...")
        
        return "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant examples found."
    
    def _format_style_profile(self, style_profile: Dict[str, Any]) -> str:
        """Format style profile for prompt"""
        if not style_profile:
            return "Writing Style: Professional\nStructure: Standard\nFormatting: Markdown"
        
        style_text = f"Writing Style: {style_profile.get('tone', 'professional')}\n"
        style_text += f"Structure: {style_profile.get('structure', 'standard')}\n"
        style_text += f"Formatting: {style_profile.get('formatting', 'markdown')}"
        return style_text
    
    async def _retrieve_context(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve relevant context from vector store"""
        logger.info("Starting context retrieval")
        if state.get("context"):
            logger.info("Context already provided in state, skipping retrieval")
            return {"context": state["context"]}
            
        try:
            retriever = RetrieverAgent()
            query = f"{state.get('summary', '')} {state.get('requirements', '')}".strip()
            
            if not query:
                logger.warning("No query available for context retrieval")
                return {"context": []}
            
            result = await retriever.execute(
                query=query,
                doc_type=state.get("doc_type", "SRS"),
                min_feedback_score=3,
                top_k=3
            )
            
            context = result.get("chunks", []) if result.get("status") == "success" else []
            logger.info(f"Retrieved {len(context)} context chunks")
            
            state.update({"context": context})
            return {"context": context}
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            state.update({"context": []})
            return {"context": []}
    
    async def _build_style_profile(self, state: GraphState) -> Dict[str, Any]:
        """Build style profile from existing documents"""
        logger.info("Building style profile")
        if state.get("style_profile"):
            logger.info("Style profile already provided in state, skipping building")
            return {"style_profile": state["style_profile"]}
            
        try:
            style_builder = StyleProfileBuilderAgent()
            db_session = state.get("db_session")
            
            if db_session:
                result = await style_builder.execute(
                    db=db_session,
                    doc_types=[state.get("doc_type", "SRS")],
                    min_feedback_score=3
                )
                profile = result.get("profile_data", {}) if result.get("status") == "success" else {}
            else:
                profile = {}
            
            logger.info(f"Built style profile: {profile}")
            state.update({"style_profile": profile})
            return {"style_profile": profile}
            
        except Exception as e:
            logger.warning(f"Style profile building failed: {e}")
            state.update({"style_profile": {}})
            return {"style_profile": {}}
    
    async def _generate_draft(self, state: GraphState) -> Dict[str, Any]:
        """Generate initial document draft"""
        logger.info("Starting document generation")
        
        # Validate inputs
        summary = state.get("summary", "").strip()
        requirements = state.get("requirements", "").strip()
        
        if not summary and not requirements:
            error_msg = "No input provided for document generation"
            logger.error(error_msg)
            state.update({"generated_content": f"# Error\n\n{error_msg}"})
            return {"generated_content": f"# Error\n\n{error_msg}"}

        guardrail = check_inputs([summary, requirements])
        if not guardrail.allowed:
            error_msg = "Input blocked because it looks like a prompt-injection attempt"
            logger.warning("%s: %s", error_msg, guardrail.reasons)
            state.update({"generated_content": f"# Blocked\n\n{error_msg}."})
            return {"generated_content": f"# Blocked\n\n{error_msg}."}
        
        try:
            # Format inputs for prompt
            context_text = self._format_context(state.get("context", []))
            style_text = self._format_style_profile(state.get("style_profile", {}))
            
            prompt_data = {
                "doc_type": state.get("doc_type", "document"),
                "summary": summary or "Not provided",
                "requirements": requirements or "Not specified",
                "style_profile": style_text,
                "context": context_text,
                "preset_guidance": format_preset_for_prompt(state.get("doc_type", "document")),
            }
            
            logger.info(f"Generating document with prompt data keys: {list(prompt_data.keys())}")
            formatted_prompt = self.srs_template.format(**prompt_data)
            
            # Generate document using proper async/sync handling
            if hasattr(self.llm, 'ainvoke'):
                # For async LLMs
                generated_content = await self.llm.ainvoke(formatted_prompt)
                # Handle different response types
                if hasattr(generated_content, 'content'):
                    generated_content = generated_content.content
                elif isinstance(generated_content, dict):
                    generated_content = generated_content.get('content', str(generated_content))
            else:
                # For sync LLMs and MockLLM
                generated_content = self.generation_chain.invoke(prompt_data)
            
            # Ensure we have string content
            if not isinstance(generated_content, str):
                generated_content = str(generated_content)
            
            if not generated_content.strip():
                generated_content = self._create_fallback_document(summary, requirements, state.get("doc_type", "SRS"))
            
            # Check for incompleteness and continue generation if needed
            is_incomplete = len(generated_content) < 1500 and not any(term in generated_content.lower() for term in ['conclusion', 'summary', 'final', 'end of document'])
            if is_incomplete:
                logger.warning("Generation appears short and incomplete; continuing...")
                continuation_prompt = formatted_prompt + f"\n\nContinue from where you left off to complete the full {state.get('doc_type', 'document')} details."
                if hasattr(self.llm, 'ainvoke'):
                    continued_content = await self.llm.ainvoke(continuation_prompt)
                else:
                    continued_content = self.llm.invoke(continuation_prompt)
                if hasattr(continued_content, 'content'):
                    continued_content = continued_content.content
                elif isinstance(continued_content, dict):
                    continued_content = continued_content.get('content', str(continued_content))
                generated_content += "\n\n" + str(continued_content)
            
            logger.info(f"Generated document length: {len(generated_content)} characters")
            state.update({"generated_content": generated_content, "final_content": generated_content})
            return {"generated_content": generated_content, "final_content": generated_content}
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            fallback_content = self._create_fallback_document(summary, requirements, state.get("doc_type", "SRS"))
            state.update({"generated_content": fallback_content, "final_content": fallback_content})
            return {"generated_content": fallback_content, "final_content": fallback_content}
            
    async def stream_generate_draft(self, state: GraphState):
        """Generate initial document draft with streaming output"""
        logger.info("Starting streaming document generation")
        
        # Validate inputs
        summary = state.get("summary", "").strip()
        requirements = state.get("requirements", "").strip()
        
        if not summary and not requirements:
            error_msg = "No input provided for document generation"
            logger.error(error_msg)
            yield f"# Error\n\n{error_msg}"
            return

        guardrail = check_inputs([summary, requirements])
        if not guardrail.allowed:
            yield "# Blocked\n\nInput blocked because it looks like a prompt-injection attempt."
            return
        
        try:
            # Format inputs for prompt
            context_text = self._format_context(state.get("context", []))
            style_text = self._format_style_profile(state.get("style_profile", {}))
            
            prompt_data = {
                "doc_type": state.get("doc_type", "document"),
                "summary": summary or "Not provided",
                "requirements": requirements or "Not specified",
                "style_profile": style_text,
                "context": context_text,
                "preset_guidance": format_preset_for_prompt(state.get("doc_type", "document")),
            }
            
            logger.info(f"Streaming document with prompt data keys: {list(prompt_data.keys())}")
            formatted_prompt = self.srs_template.format(**prompt_data)
            
            # Use streaming capabilities if available
            generated_content = ""
            if hasattr(self.llm, 'astream') and callable(getattr(self.llm, 'astream')):
                # For LLMs that support streaming
                async for chunk in self.llm.astream(formatted_prompt):
                    content_piece = ""
                    if hasattr(chunk, 'content'):
                        content_piece = chunk.content
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        content_piece = chunk['content']
                    elif isinstance(chunk, str):
                        content_piece = chunk
                    else:
                        content_piece = str(chunk)
                    
                    # Add to the complete content
                    generated_content += content_piece
                    # Yield each chunk for streaming
                    yield content_piece
            else:
                # Fallback for LLMs without streaming - simulate streaming with chunks
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(formatted_prompt)
                    if hasattr(response, 'content'):
                        generated_content = response.content
                    elif isinstance(response, dict):
                        generated_content = response.get('content', str(response))
                    else:
                        generated_content = str(response)
                else:
                    generated_content = self.generation_chain.invoke(prompt_data)
                
                # Simulate streaming by yielding chunks
                chunk_size = 50  # Increased chunk size for better readability
                for i in range(0, len(generated_content), chunk_size):
                    yield generated_content[i:i+chunk_size]
                    await asyncio.sleep(0.02)  # Faster streaming for better UX
            
            # Store the complete content in the state
            state.update({"generated_content": generated_content})
            
        except Exception as e:
            logger.error(f"Streaming document generation failed: {e}")
            fallback_content = self._create_fallback_document(summary, requirements, state.get("doc_type", "SRS"))
            yield fallback_content
            state.update({"generated_content": fallback_content})
    
    def _create_fallback_document(self, summary: str, requirements: str, doc_type: str) -> str:
        """Create a basic fallback document when generation fails"""
        doc_title = f"{doc_type} Document"
        if summary:
            doc_title += f": {summary}"
        
        content = f"# {doc_title}\n\n"
        content += "## Overview\n"
        if summary:
            content += f"{summary}\n\n"
        else:
            content += "This document outlines the requirements and specifications.\n\n"
        
        content += "## Requirements\n"
        if requirements:
            # Split requirements into list items if not already formatted
            req_lines = requirements.split('\n')
            for line in req_lines:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('*'):
                    content += f"- {line}\n"
                elif line:
                    content += f"{line}\n"
        else:
            content += "- Core functionality requirements to be defined\n"
            content += "- Performance requirements to be specified\n"
            content += "- Security requirements to be outlined\n"
        
        content += "\n## Conclusion\n"
        content += "This document serves as the foundation for the project requirements and will be updated as needed.\n"
        
        return content
    
    async def _review_document(self, state: GraphState) -> Dict[str, Any]:
        """Review and refine the generated document"""
        logger.info("Starting document review")
        
        generated_content = state.get("generated_content", "")
        
        if not generated_content or generated_content.strip() == "":
            logger.warning("No content available for review, skipping review step")
            state.update({"final_content": "# Error\n\nNo content was generated"})
            return {"final_content": "# Error\n\nNo content was generated"}
        
        try:
            reviewer = ReviewEditingAgent()
            result = await reviewer.execute(
                content=generated_content,
                doc_type=state.get("doc_type", "SRS"),
                style_profile=state.get("style_profile", {}),
                review_type="formatting"
            )
            
            if result.get("status") == "success":
                final_content = result.get("improved_content", generated_content)
            else:
                logger.warning(f"Review failed: {result.get('message', 'Unknown error')}")
                final_content = generated_content
            
            logger.info(f"Review completed, final content length: {len(final_content)}")
            state.update({"final_content": final_content})
            return {"final_content": final_content}
            
        except Exception as e:
            logger.warning(f"Document review failed: {e}")
            # Return original content if review fails
            state.update({"final_content": generated_content})
            return {"final_content": generated_content}
    
    async def _direct_generation(self, summary: str, requirements: str, doc_type: str) -> Dict[str, Any]:
        """Direct document generation without workflow (fallback method)"""
        logger.info("Attempting direct document generation")
        
        try:
            prompt_data = {
                "doc_type": doc_type,
                "summary": summary or "Not provided",
                "requirements": requirements or "Not specified", 
                "style_profile": "Writing Style: Professional\nStructure: Standard\nFormatting: Markdown",
                "context": "No relevant examples found.",
                "preset_guidance": format_preset_for_prompt(doc_type),
            }
            
            if hasattr(self.llm, 'ainvoke'):
                formatted_prompt = self.srs_template.format(**prompt_data)
                generated_content = await self.llm.ainvoke(formatted_prompt)
                if hasattr(generated_content, 'content'):
                    generated_content = generated_content.content
            else:
                generated_content = self.generation_chain.invoke(prompt_data)
            
            if not isinstance(generated_content, str):
                generated_content = str(generated_content)
            
            if not generated_content.strip():
                generated_content = self._create_fallback_document(summary, requirements, doc_type)
            
            return {
                "status": "success",
                "generated_content": generated_content,
                "word_count": len(generated_content.split()),
                "document_id": None
            }
            
        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            fallback_content = self._create_fallback_document(summary, requirements, doc_type)
            return {
                "status": "success",
                "generated_content": fallback_content,
                "word_count": len(fallback_content.split()),
                "document_id": None
            }
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    async def execute(
        self, 
        doc_type: str, 
        summary: str, 
        requirements: str, 
        context: Optional[List[Dict[str, Any]]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
        approved: bool = False,
        feedback_score: int = 3,
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Execute the document generation workflow"""
        
        # Validate inputs
        if not summary and not requirements:
            return {
                "status": "error",
                "message": "Either summary or requirements must be provided",
                "document_id": None,
                "generated_content": "",
                "word_count": 0
            }

        guardrail = check_inputs([summary, requirements])
        if not guardrail.allowed:
            return {
                "status": "error",
                "message": "Input blocked because it looks like a prompt-injection attempt",
                "document_id": None,
                "generated_content": "",
                "word_count": 0,
                "guardrail_reasons": guardrail.reasons,
            }
        
        try:
            # Prepare initial state
            initial_state = GraphState({
                "doc_type": doc_type,
                "summary": summary.strip() if summary else "",
                "requirements": requirements.strip() if requirements else "",
                "context": context or [],
                "style_profile": style_profile or {},
                "approved": approved,
                "feedback_score": feedback_score,
                "db_session": db_session
            })
            
            logger.info(f"Executing DocGenerationAgent workflow for {doc_type}")
            
            # Execute workflow with timeout
            try:
                result = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state),
                    timeout=300  # 5 minute timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Workflow timeout, attempting direct generation")
                return await self._direct_generation(summary, requirements, doc_type)
            
            # Validate result
            if not result or not isinstance(result, dict):
                logger.warning("Workflow returned invalid result, attempting direct generation")
                return await self._direct_generation(summary, requirements, doc_type)
            
            final_content = result.get("final_content", "").strip()
            
            # If no content was generated, try direct generation
            if not final_content:
                logger.warning("No content from workflow, attempting direct generation")
                return await self._direct_generation(summary, requirements, doc_type)
            
            # Store in database if approved
            document_id = None
            if approved and db_session:
                try:
                    ingestion_agent = DocumentIngestionAgent()
                    ingest_result = await ingestion_agent.execute(
                        db=db_session,
                        filename=f"{doc_type}_{uuid.uuid4().hex[:8]}.md",
                        content=final_content,
                        doc_type=doc_type,
                        approved=approved,
                        feedback_score=feedback_score
                    )
                    if ingest_result.get("status") == "success":
                        document_id = ingest_result.get("document_id")
                except Exception as e:
                    logger.warning(f"Document storage failed: {e}")
            
            return {
                "status": "success",
                "document_id": document_id,
                "generated_content": final_content,
                "word_count": len(final_content.split()),
                "execution_time": 0  # Could add timing if needed
            }
            
        except Exception as e:
            logger.error(f"DocGenerationAgent execution failed: {e}")
            # Try direct generation as last resort
            try:
                return await self._direct_generation(summary, requirements, doc_type)
            except Exception as direct_error:
                logger.error(f"Direct generation also failed: {direct_error}")
                return {
                    "status": "error",
                    "message": f"All generation methods failed: {str(e)}",
                    "document_id": None,
                    "generated_content": self._create_fallback_document(summary, requirements, doc_type),
                    "word_count": 0
                }
