"""LangGraph workflow orchestration for DocAssist."""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from sqlalchemy.orm import Session
import asyncio
import logging
from datetime import datetime
import uuid

from .agents.DocumentIngestionAgent import DocumentIngestionAgent
from .agents.StyleProfileBuilderAgent import StyleProfileBuilderAgent
from .agents.RetrieverAgent import RetrieverAgent
from .agents.DocGenerationAgent import DocGenerationAgent
from .agents.ReviewEditingAgent import ReviewEditingAgent
from .database import get_db_sync
from .document_presets import get_required_sections
from .models import WorkflowExecution, AgentExecution

import json

logger = logging.getLogger(__name__)

def safe_serialize_for_db(data):
    """Safely serialize data for database storage as JSON."""
    try:
        return json.dumps(data, default=str)
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        return "{}"

class GraphState(TypedDict):
    """State for the Langgraph workflow with proper typing"""
    # Input parameters
    doc_type: str
    summary: str
    requirements: str
    style: str
    
    # Workflow state
    style_profile: Optional[Dict[str, Any]]
    retrieved_context: Optional[List[Dict[str, Any]]]
    draft_document: Optional[str]
    reviewed_document: Optional[str]
    final_document: Optional[str]
    
    # Metadata
    document_id: Optional[str]
    workflow_id: Optional[str]
    workflow_status: str
    error_message: Optional[str]
    
    # Quality and compliance
    compliance_check: Optional[Dict[str, Any]]
    quality_score: Optional[float]
    
    # Feedback and iteration
    feedback: Optional[List[str]]
    iteration_count: int
    max_iterations: int
    
    # Agent execution tracking
    agent_executions: List[Dict[str, Any]]
    
    # Messages for debugging
    messages: Annotated[List[str], add_messages]

class AgenticRAGWorkflow:
    """Orchestrates multi-agent workflow using Langgraph StateGraph"""
    
    def __init__(self):
        self.agents = {
            "ingestion": DocumentIngestionAgent(),
            "style_builder": StyleProfileBuilderAgent(),
            "retriever": RetrieverAgent(),
            "generator": DocGenerationAgent(),
            "reviewer": ReviewEditingAgent()
        }
        
        self.workflow = self._build_workflow()
        self.logger = logging.getLogger("workflow")
    
    def _build_workflow(self) -> StateGraph:
        """Build the Langgraph workflow"""
        
        # Create the graph with proper state
        workflow = StateGraph(GraphState)
        
        # Add nodes for each workflow step
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("build_style_profile", self._build_style_profile_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_document", self._generate_document_node)
        workflow.add_node("compliance_check", self._compliance_check_node)
        workflow.add_node("review_document", self._review_document_node)
        workflow.add_node("finalize_document", self._finalize_document_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.add_edge(START, "initialize")
        
        # Add sequential edges
        workflow.add_edge("initialize", "build_style_profile")
        workflow.add_edge("build_style_profile", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_document")
        workflow.add_edge("generate_document", "compliance_check")
        
        # Add conditional edges for decision points
        workflow.add_conditional_edges(
            "compliance_check",
            self._should_review,
            {
                "review": "review_document",
                "finalize": "finalize_document",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "review_document",
            self._should_regenerate,
            {
                "regenerate": "generate_document",
                "finalize": "finalize_document",
                "error": "handle_error"
            }
        )
        
        # End nodes
        workflow.add_edge("finalize_document", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def execute_generation_workflow(
        self,
        doc_type: str,
        summary: str,
        requirements: str,
        style: str = "professional",
        db: Session = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Execute the complete document generation workflow"""
        workflow_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting workflow {workflow_id} for {doc_type}: {summary}")
            
            # Create workflow execution record
            if db:
                workflow_execution = WorkflowExecution(
                    id=workflow_id,
                    workflow_type="document_generation",
                    current_state="initializing",
                    status="running"
                )
                db.add(workflow_execution)
                db.commit()
            
            # Initialize state
            initial_state = GraphState(
                doc_type=doc_type,
                summary=summary,
                requirements=requirements,
                style=style,
                style_profile=None,
                retrieved_context=None,
                draft_document=None,
                reviewed_document=None,
                final_document=None,
                document_id=None,
                workflow_id=workflow_id,
                workflow_status="initializing",
                error_message=None,
                compliance_check=None,
                quality_score=0.8,
                feedback=[],
                iteration_count=0,
                max_iterations=max_iterations,
                agent_executions=[],
                messages=[]
            )
            
            # Debug state before execution
            self.logger.debug(f"Initial state: {initial_state}")
            
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Debug final state
            self.logger.debug(f"Final state: {result}")
            
            # Update workflow execution record
            if db:
                workflow_execution.status = result.get("workflow_status", "completed")
                workflow_execution.current_state = result.get("workflow_status", "completed")
                workflow_execution.result_data = safe_serialize_for_db(result)
                workflow_execution.completed_at = datetime.utcnow()
                db.commit()
            
            # Prepare response
            response = {
                "workflow_id": workflow_id,
                "document_id": result.get("document_id"),
                "content": result.get("final_document", ""),
                "status": result.get("workflow_status", "completed"),
                "messages": result.get("messages", []),
                "quality_score": result.get("quality_score", 0.7),
                "agent_executions": result.get("agent_executions", [])
            }
            
            if result.get("error_message"):
                response["error"] = result["error_message"]
            
            self.logger.info(f"Workflow {workflow_id} completed with status: {response['status']}")
            return response
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            if db:
                workflow_execution.status = "failed"
                workflow_execution.current_state = "failed"
                workflow_execution.error_message = str(e)
                workflow_execution.completed_at = datetime.utcnow()
                db.commit()
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "messages": [f"Workflow failed: {str(e)}"]
            }
    
    async def _initialize_node(self, state: GraphState) -> GraphState:
        """Initialize the workflow state"""
        try:
            self.logger.info("Initializing workflow")
            
            state["workflow_status"] = "initializing"
            state["iteration_count"] = 0
            state["agent_executions"] = []
            state["messages"] = [f"Workflow initialized for {state['doc_type']}"]
            
            # Record agent execution
            execution = {
                "agent": "initializer",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"doc_type": state["doc_type"]}
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after initialization: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error initializing workflow: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _build_style_profile_node(self, state: GraphState) -> GraphState:
        """Build style profile for document generation"""
        try:
            self.logger.info("Building style profile")
            
            agent = self.agents["style_builder"]
            db = get_db_sync()
            result = await agent.execute(db=db, doc_types=[state["doc_type"]], min_feedback_score=3)
            self.logger.info(f"Style profile result: {result}")
            
            if result is None:
                self.logger.error("StyleProfileBuilderAgent returned None")
                state["error_message"] = "StyleProfileBuilderAgent returned None"
                state["workflow_status"] = "error"
                return state
                
            state["style_profile"] = result.get("profile_data", {})
            state["workflow_status"] = "style_profile_built"
            state["messages"].append("Style profile built successfully")
            
            # Record agent execution
            execution = {
                "agent": "style_builder",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "profile_id": result.get("profile_id"),
                    "document_count": result.get("profile_data", {}).get("document_count", 0)
                }
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after style profile: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error building style profile: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _retrieve_context_node(self, state: GraphState) -> GraphState:
        """Retrieve context for document generation"""
        try:
            self.logger.info("Retrieving context")
            
            agent = self.agents["retriever"]
            result = await agent.execute(
                query=f"{state['summary']} {state['requirements']}",
                doc_type=state["doc_type"],
                min_feedback_score=3,
                top_k=3
            )
            self.logger.info(f"Retrieval result: {result}")
            
            if result is None:
                self.logger.error("RetrieverAgent returned None")
                state["error_message"] = "RetrieverAgent returned None"
                state["workflow_status"] = "error"
                return state
                
            state["retrieved_context"] = result.get("chunks", [])
            state["workflow_status"] = "context_retrieved"
            state["messages"].append(f"Retrieved {len(result.get('chunks', []))} context chunks")
            
            # Record agent execution
            execution = {
                "agent": "retriever",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "chunk_count": len(result.get("chunks", [])),
                    "query": f"{state['summary']} {state['requirements']}"[:100]
                }
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after retrieval: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _generate_document_node(self, state: GraphState) -> GraphState:
        """Generate document draft"""
        try:
            self.logger.info("Generating document")
            
            agent = self.agents["generator"]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            result = await agent.execute(
                doc_type=state["doc_type"],
                summary=state["summary"],
                requirements=state["requirements"],
                context=state["retrieved_context"],
                style_profile=state["style_profile"],
                approved=False,
                feedback_score=3,
                db_session=None
            )
            self.logger.info(f"Generation result: {result}")
            
            if result is None:
                self.logger.error("DocGenerationAgent returned None")
                state["error_message"] = "DocGenerationAgent returned None"
                state["workflow_status"] = "error"
                return state
                
            state["draft_document"] = result.get("generated_content", "")
            state["workflow_status"] = "document_generated"
            state["messages"].append(f"Document generated (iteration {state['iteration_count']})")
            
            # Record agent execution
            execution = {
                "agent": "generator",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "word_count": len(state["draft_document"].split()),
                    "iteration": state["iteration_count"]
                }
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after generation: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error generating document: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _compliance_check_node(self, state: GraphState) -> GraphState:
        """Check document compliance using preset-required sections."""
        try:
            self.logger.info("Checking compliance")
            
            required_sections = self._get_required_sections(state["doc_type"])
            content = state.get("draft_document", "")
            compliance = {
                "compliant": True,
                "missing_sections": [],
                "issues": []
            }
            
            # Basic section presence check
            for section in required_sections:
                if section.lower() not in content.lower():
                    compliance["missing_sections"].append(section)
                    compliance["compliant"] = False
            
            # Stricter check for completeness (e.g., minimum word count)
            if len(content.split()) < 500:
                compliance["compliant"] = False
                compliance["issues"].append("Document too short - likely incomplete or truncated")
            
            state["compliance_check"] = compliance
            state["workflow_status"] = "compliance_checked"
            state["messages"].append(f"Compliance check: {'Compliant' if compliance['compliant'] else 'Issues found'}")
            
            # Record agent execution
            execution = {
                "agent": "compliance_checker",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"compliant": compliance["compliant"], "missing_sections": compliance["missing_sections"]}
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after compliance check: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error checking compliance: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _review_document_node(self, state: GraphState) -> GraphState:
        """Review and improve document"""
        try:
            self.logger.info("Reviewing document")
            
            agent = self.agents["reviewer"]
            result = await agent.execute(
                content=state["draft_document"],
                doc_type=state["doc_type"],
                style_profile=state["style_profile"],
                feedback=state["feedback"],
                review_type="both",
                approved=False,
                feedback_score=3,
                db_session=None
            )
            self.logger.info(f"Review result: {result}")
            
            if result is None:
                self.logger.error("ReviewEditingAgent returned None")
                state["error_message"] = "ReviewEditingAgent returned None"
                state["workflow_status"] = "error"
                return state
                
            state["reviewed_document"] = result.get("improved_content", "")
            state["workflow_status"] = "document_reviewed"
            
            # Update quality score based on improvements
            improvements = result.get("changes_made", [])
            current_quality = state.get("quality_score", 0.7)
            state["quality_score"] = min(1.0, current_quality + len(improvements) * 0.05)
            
            state["messages"].append(f"Document reviewed with {len(improvements)} improvements")
            
            # Record agent execution
            execution = {
                "agent": "reviewer",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "improvements_made": len(improvements),
                    "quality_score": state["quality_score"]
                }
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after review: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error reviewing document: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _finalize_document_node(self, state: GraphState) -> GraphState:
        """Finalize document and store in database"""
        try:
            self.logger.info("Finalizing document")
            
            # Use reviewed document if available, otherwise draft
            final_content = (
                state.get("reviewed_document") or 
                state.get("draft_document") or 
                ""
            )
            
            state["final_document"] = final_content
            state["workflow_status"] = "completed"
            
            # Generate document ID if not exists
            if not state.get("document_id"):
                state["document_id"] = str(uuid.uuid4())
            
            # NEW: Ensure feedback_score is an integer
            raw_feedback_score = state.get("quality_score", 0.7) * 5
            feedback_score = max(1, min(5, int(raw_feedback_score)))  # Convert to int
            self.logger.debug(f"Calculated feedback_score: {feedback_score} (from raw {raw_feedback_score})")
            
            # Store document in database and vector store
            agent = self.agents["ingestion"]
            db = get_db_sync()
            result = await agent.execute(
                db=db,
                filename=f"final_{state['doc_type']}_{state['document_id']}.md",
                content=final_content,
                doc_type=state["doc_type"],
                approved=True,
                feedback_score=feedback_score
            )
            
            word_count = len(final_content.split())
            state["messages"].append(f"Document finalized with {word_count} words and stored with ID {result['document_id']}")
            
            # Record agent execution
            execution = {
                "agent": "finalizer",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "final_word_count": word_count,
                    "total_iterations": state.get("iteration_count", 0),
                    "document_id": result["document_id"],
                    "feedback_score": feedback_score
                }
            }
            state["agent_executions"].append(execution)
            
            # Debug state
            self.logger.debug(f"State after finalization: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error finalizing document: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    async def _handle_error_node(self, state: GraphState) -> GraphState:
        """Handle workflow errors"""
        try:
            error_msg = state.get("error_message", "Unknown error")
            self.logger.error(f"Workflow error: {error_msg}")
            
            state["workflow_status"] = "failed"
            state["messages"].append(f"Workflow failed: {error_msg}")
            
            # Persist error to database
            db = get_db_sync()
            execution = AgentExecution(
                workflow_id=state["workflow_id"],
                agent_name="error_handler",
                status="failed",
                error_message=error_msg,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            db.add(execution)
            db.commit()
            
            # Debug state
            self.logger.debug(f"State after error handling: {state}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            state["workflow_status"] = "failed"
            state["messages"].append(f"Error handler failed: {str(e)}")
            return state
    
    def _should_review(self, state: GraphState) -> str:
        """Determine if document should be reviewed"""
        if state.get("error_message"):
            return "error"

        compliance = state.get("compliance_check", {})
        if not compliance.get("compliant", True):
            return "review"
        quality_score = state.get("quality_score", 0.0)
        if quality_score < 0.8:
            return "review"
        return "finalize"
    
    def _should_regenerate(self, state: GraphState) -> str:
        """Determine if document should be regenerated"""
        if state.get("error_message"):
            return "error"
        
        # Check iteration limit
        if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
            return "finalize"
        
        # Check quality score after review
        quality_score = state.get("quality_score", 0.0)
        if quality_score < 0.7:
            return "regenerate"
        
        return "finalize"
    
    def _get_required_sections(self, doc_type: str) -> List[str]:
        """Get required sections for document type"""
        return get_required_sections(doc_type)

# Global workflow instance
workflow_manager = AgenticRAGWorkflow()
