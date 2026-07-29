"""Style Profile Builder Agent - Production-ready implementation for learning document styles"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from collections import defaultdict
import re
import tenacity
import logging
from datetime import datetime

from .base_agent import BaseAgent
from ..models import Document, StyleProfile
from ..config import settings

logger = logging.getLogger(__name__)

class StyleProfileBuilderAgent(BaseAgent):
    """Production-ready Style Profile Builder Agent that learns tone, structure, and points from documents"""
    
    def __init__(self):
        super().__init__(name="style_profile_builder", description="Learns tone, structure, and points from previously uploaded SRS docs")
        self._cache = {}  # In-memory cache for style profiles
    
    def _analyze_tone(self, content: str) -> Dict[str, float]:
        """Analyze the tone of a document"""
        if not content:
            return {"professional": 0.5, "technical": 0.5, "formal": 0.5}
        
        content_lower = content.lower()
        total_words = len(content.split())
        
        if total_words == 0:
            return {"professional": 0.5, "technical": 0.5, "formal": 0.5}
        
        # Define keywords for different tones
        professional_keywords = [
            "requirements", "specifications", "implementation", "deliverables", 
            "stakeholders", "objectives", "methodology", "framework"
        ]
        
        technical_keywords = [
            "system", "architecture", "database", "api", "interface", 
            "algorithm", "protocol", "configuration", "deployment"
        ]
        
        formal_keywords = [
            "shall", "must", "should", "will", "hereby", "therefore", 
            "furthermore", "consequently", "accordingly"
        ]
        
        # Count occurrences
        professional_count = sum(content_lower.count(word) for word in professional_keywords)
        technical_count = sum(content_lower.count(word) for word in technical_keywords)
        formal_count = sum(content_lower.count(word) for word in formal_keywords)
        
        # Normalize scores
        return {
            "professional": min(professional_count / total_words * 1000, 1.0),
            "technical": min(technical_count / total_words * 1000, 1.0),
            "formal": min(formal_count / total_words * 1000, 1.0)
        }
    
    def _extract_terminology(self, content: str) -> Dict[str, int]:
        """Extract common terminology from a document"""
        if not content:
            return {}
        
        # Find words with 4 or more characters
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Define relevant technical terms
        relevant_terms = [
            "requirements", "specifications", "implementation", "system", 
            "architecture", "design", "development", "testing", "deployment", 
            "database", "interface", "api", "security", "performance", 
            "functionality", "feature", "module", "component", "service"
        ]
        
        # Count occurrences of relevant terms
        term_counts = {
            term: words.count(term) 
            for term in relevant_terms 
            if words.count(term) > 0
        }
        
        # Return top 15 terms
        return dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of a document"""
        if not content:
            return {"heading_patterns": {}, "section_types": []}
        
        # Find all headers
        headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        
        # Analyze heading patterns
        heading_patterns = defaultdict(int)
        section_types = []
        
        for level, text in headers:
            # Count heading levels
            heading_patterns[f"level_{len(level)}"] += 1
            
            # Identify section types
            text_lower = text.lower()
            if any(word in text_lower for word in ['introduction', 'overview']):
                section_types.append("introduction")
            elif any(word in text_lower for word in ['requirement', 'specification']):
                section_types.append("requirements")
            elif any(word in text_lower for word in ['architecture', 'design']):
                section_types.append("architecture")
            elif any(word in text_lower for word in ['implementation', 'development']):
                section_types.append("implementation")
            elif any(word in text_lower for word in ['testing', 'validation']):
                section_types.append("testing")
            elif any(word in text_lower for word in ['conclusion', 'summary']):
                section_types.append("conclusion")
        
        return {
            "heading_patterns": dict(heading_patterns),
            "section_types": list(set(section_types))
        }
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Get a default style profile"""
        return {
            "tone": "professional",
            "structure": "standard",
            "formatting": "markdown",
            "heading_style": "atx",
            "list_style": "bulleted",
            "document_count": 0,
            "is_default": True,
            "created_at": datetime.utcnow().isoformat()
        }
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    async def execute(
        self,
        db: Session,
        doc_types: List[str] = None,
        min_feedback_score: Optional[int] = 3
    ) -> Dict[str, Any]:
        """Build a style profile from documents"""
        logger.info(f"StyleProfileBuilderAgent executing with doc_types: {doc_types}, min_feedback_score: {min_feedback_score}")
        try:
            # Create cache key
            cache_key = (tuple(sorted(doc_types or [])), min_feedback_score)
            
            # Check cache first
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                # Check if cache is still valid (less than 5 minutes old)
                if datetime.utcnow().timestamp() - cached_result.get("timestamp", 0) < 300:
                    return cached_result["data"]
            
            # Query documents
            query = db.query(Document)
            
            # Filter by document types if specified
            if doc_types:
                query = query.filter(Document.doc_type.in_(doc_types))
            
            # Filter by minimum feedback score
            if min_feedback_score is not None:
                query = query.filter(Document.feedback_score >= min_feedback_score)
            
            # Filter only approved documents
            query = query.filter(Document.approved == True)
            
            documents = query.all()
            
            # Return default profile if no documents found
            if not documents:
                default_profile = self._get_default_profile()
                result = {
                    "status": "success",
                    "profile_data": default_profile,
                    "document_count": 0
                }
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.utcnow().timestamp()
                }
                
                return result
            
            # Analyze documents
            tone_analysis = defaultdict(float)
            terminology = defaultdict(int)
            structure_analysis = defaultdict(lambda: defaultdict(int))
            total_documents = len(documents)
            
            # Weight documents by feedback score
            total_weight = 0
            for doc in documents:
                weight = doc.feedback_score / 5.0  # Normalize to 0-1
                total_weight += weight
                
                # Analyze tone
                doc_tone = self._analyze_tone(doc.content)
                for tone, score in doc_tone.items():
                    tone_analysis[tone] += score * weight
                
                # Extract terminology
                doc_terms = self._extract_terminology(doc.content)
                for term, count in doc_terms.items():
                    terminology[term] += count * weight
                
                # Analyze structure
                doc_structure = self._analyze_structure(doc.content)
                for level, count in doc_structure.get("heading_patterns", {}).items():
                    structure_analysis["heading_patterns"][level] += count * weight
            
            # Normalize tone analysis
            if total_weight > 0:
                for tone in tone_analysis:
                    tone_analysis[tone] /= total_weight
            
            # Get dominant tone
            dominant_tone = max(tone_analysis, key=tone_analysis.get) if tone_analysis else "professional"
            
            # Get common terminology
            common_terms = dict(sorted(terminology.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Determine structure patterns
            heading_patterns = dict(structure_analysis["heading_patterns"])
            dominant_heading = max(heading_patterns, key=heading_patterns.get) if heading_patterns else "level_1"
            
            # Create style profile
            style_profile = {
                "tone": dominant_tone,
                "tone_analysis": dict(tone_analysis),
                "terminology": common_terms,
                "structure": "standard",
                "heading_style": "atx" if "level_1" in dominant_heading else "setext",
                "list_style": "bulleted",
                "formatting": "markdown",
                "document_count": total_documents,
                "is_default": False,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Save to database
            profile_record = StyleProfile(
                name=f"{'_'.join(doc_types or ['all'])}_profile",
                profile_data=style_profile,
                doc_types=doc_types
            )
            db.add(profile_record)
            db.commit()
            
            result = {
                "status": "success",
                "profile_id": profile_record.id,
                "profile_data": style_profile,
                "document_count": total_documents
            }
            
            # Cache the result
            self._cache[cache_key] = {
                "data": result,
                "timestamp": datetime.utcnow().timestamp()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"StyleProfileBuilderAgent execution failed: {e}")
            db.rollback()
            
            # Return default profile on error
            default_profile = self._get_default_profile()
            return {
                "status": "error",
                "message": str(e),
                "profile_data": default_profile,
                "document_count": 0
            }