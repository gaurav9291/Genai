"""Database models for DocAssist."""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    title = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)  # SRS, SOW, Proposal, etc.
    content = Column(Text, nullable=False)
    file_path = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    status = Column(String, default="draft")  # draft, review, final
    approved = Column(Boolean, default=False)  # Whether the document is approved for learning
    feedback_score = Column(Integer, default=3)  # RLHF feedback score (1-5 scale)
    
    # Metadata
    style_metadata = Column(JSON, nullable=True)
    generation_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    generation_history = relationship("GenerationHistory", back_populates="document")
    feedback = relationship("UserFeedback", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Metadata
    section_type = Column(String, nullable=True)  # introduction, requirements, etc.
    chunk_metadata = Column(JSON, nullable=True)
    
    # Vector embedding (stored as JSON for FAISS compatibility)
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class StyleProfile(Base):
    __tablename__ = "style_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, default="default")
    profile_data = Column(JSON, nullable=False)
    doc_types = Column(JSON, nullable=True)  # List of doc types this profile applies to
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)

    # Snapshot fields
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String, nullable=True)
    feedback_score = Column(Integer, nullable=True)
    change_note = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="versions")

class GenerationHistory(Base):
    __tablename__ = "generation_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    
    # Generation parameters
    generation_params = Column(JSON, nullable=False)
    workflow_state = Column(JSON, nullable=True)
    
    # Results
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Metrics
    execution_time = Column(Float, nullable=True)  # seconds
    tokens_used = Column(Integer, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="generation_history")

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_type = Column(String, nullable=False)  # generation, review, etc.
    
    # State tracking
    current_state = Column(String, nullable=False)
    state_data = Column(JSON, nullable=True)
    
    # Status
    status = Column(String, default="running")  # running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Results
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class AgentExecution(Base):
    __tablename__ = "agent_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("workflow_executions.id"), nullable=True)
    
    # Agent info
    agent_name = Column(String, nullable=False)
    agent_version = Column(String, nullable=True)
    
    # Execution details
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    
    # Status
    status = Column(String, default="running")  # running, completed, failed
    execution_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    
    # Feedback details
    feedback_type = Column(String, nullable=False)  # quality, structure, content, etc.
    feedback_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)  # 1-5 scale
    
    # Context
    section = Column(String, nullable=True)
    context_data = Column(JSON, nullable=True)
    
    # Processing
    processed = Column(Boolean, default=False)
    applied = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="feedback")
