"""Document generation preset definitions.

These presets keep generation guidance and compliance checks aligned without
turning the workflow into many separate hardcoded generators.
"""

from __future__ import annotations

from typing import Dict, List


DEFAULT_PRESET = {
    "label": "Document",
    "tone": "clear, professional, and useful",
    "sections": ["Introduction", "Content", "Conclusion"],
    "rules": [
        "Use descriptive Markdown headings.",
        "Prefer concrete details over generic filler.",
        "Include assumptions, risks, and next steps when useful.",
    ],
    "checklist": [
        "The document has a clear purpose.",
        "Sections are complete enough to be actionable.",
        "The ending summarizes the important points.",
    ],
}


DOCUMENT_PRESETS: Dict[str, Dict[str, List[str] | str]] = {
    "Report": {
        "label": "Report",
        "tone": "analytical, evidence-oriented, and concise",
        "sections": ["Executive Summary", "Background", "Findings", "Analysis", "Recommendations", "Conclusion"],
        "rules": [
            "Start with a brief executive summary.",
            "Separate observations from recommendations.",
            "Use tables for comparisons, metrics, or status summaries when helpful.",
        ],
        "checklist": [
            "Findings are clearly separated from opinions.",
            "Recommendations are specific and practical.",
            "The conclusion connects back to the goal.",
        ],
    },
    "Proposal": {
        "label": "Proposal",
        "tone": "persuasive, credible, and business-focused",
        "sections": ["Overview", "Problem Statement", "Proposed Solution", "Approach", "Timeline", "Budget", "Benefits", "Risks"],
        "rules": [
            "Make the value proposition explicit.",
            "Include scope, assumptions, timeline, and budget placeholders if exact values are not provided.",
            "Use persuasive but honest language.",
        ],
        "checklist": [
            "The proposal explains why the solution is needed.",
            "The approach and deliverables are concrete.",
            "Costs, timeline, risks, and benefits are visible.",
        ],
    },
    "User Guide": {
        "label": "User Guide",
        "tone": "friendly, instructional, and task-oriented",
        "sections": ["Overview", "Getting Started", "Prerequisites", "Core Workflows", "Step-by-Step Instructions", "Troubleshooting", "FAQ"],
        "rules": [
            "Write for an end user, not an internal engineer.",
            "Use numbered steps for procedures.",
            "Include expected results after important steps.",
        ],
        "checklist": [
            "A new user can understand what to do first.",
            "Common workflows have step-by-step instructions.",
            "Troubleshooting covers likely mistakes.",
        ],
    },
    "Technical Spec": {
        "label": "Technical Specification",
        "tone": "precise, implementation-oriented, and unambiguous",
        "sections": ["Overview", "Goals", "Architecture", "Data Model", "API / Interfaces", "Implementation Plan", "Security", "Testing", "Risks"],
        "rules": [
            "Be specific about components, data flow, interfaces, and constraints.",
            "Include tables for APIs, fields, endpoints, or module responsibilities when useful.",
            "Separate design decisions from open questions.",
        ],
        "checklist": [
            "Architecture and component responsibilities are clear.",
            "Interfaces or APIs are described concretely.",
            "Testing, security, and risks are addressed.",
        ],
    },
    "SRS": {
        "label": "Software Requirements Specification",
        "tone": "formal, structured, and requirement-focused",
        "sections": ["Introduction", "Overall Description", "Functional Requirements", "Non-Functional Requirements", "External Interfaces", "Acceptance Criteria", "Assumptions", "Constraints"],
        "rules": [
            "Write requirements as testable statements using shall/should where appropriate.",
            "Separate functional and non-functional requirements.",
            "Include acceptance criteria and constraints.",
        ],
        "checklist": [
            "Functional requirements are numbered or clearly separated.",
            "Non-functional requirements include performance, security, reliability, or usability when relevant.",
            "Acceptance criteria are testable.",
        ],
    },
    "SOW": {
        "label": "Statement of Work",
        "tone": "contract-ready, clear, and scope-controlled",
        "sections": ["Scope", "Objectives", "Deliverables", "Timeline", "Roles and Responsibilities", "Acceptance Criteria", "Assumptions", "Out of Scope"],
        "rules": [
            "Make deliverables and boundaries explicit.",
            "Include acceptance criteria for major deliverables.",
            "State assumptions and out-of-scope items to reduce ambiguity.",
        ],
        "checklist": [
            "Scope and out-of-scope items are visible.",
            "Deliverables are measurable.",
            "Timeline and responsibilities are clear.",
        ],
    },
    "Business Plan": {
        "label": "Business Plan",
        "tone": "strategic, investor-aware, and practical",
        "sections": ["Executive Summary", "Market Analysis", "Customer Segments", "Product / Service", "Business Model", "Go-To-Market Strategy", "Operations", "Financial Plan", "Risks"],
        "rules": [
            "Connect strategy to market, customers, operations, and finances.",
            "Use realistic placeholders when exact figures are not provided.",
            "Include risks and mitigation ideas.",
        ],
        "checklist": [
            "Market and customer segments are described.",
            "Revenue model and operations are clear.",
            "Financial assumptions and risks are visible.",
        ],
    },
    "Story": {
        "label": "Story",
        "tone": "imaginative, vivid, and emotionally engaging",
        "sections": ["Title", "Premise", "Characters", "Setting", "Story", "Ending"],
        "rules": [
            "Prioritize scene, character motivation, and sensory detail.",
            "Give the story a beginning, middle, and satisfying ending.",
            "Avoid sounding like a business document unless the user asks for that style.",
        ],
        "checklist": [
            "Characters have clear motivations.",
            "The plot has conflict and resolution.",
            "The ending feels intentional.",
        ],
    },
    "Creative Brief": {
        "label": "Creative Brief",
        "tone": "clear, imaginative, and campaign-ready",
        "sections": ["Project Overview", "Objective", "Audience", "Key Message", "Tone and Style", "Deliverables", "Channels", "Success Metrics"],
        "rules": [
            "Make the audience and creative direction explicit.",
            "Include deliverables and success metrics.",
            "Keep the brief inspiring but operational.",
        ],
        "checklist": [
            "Audience and objective are clear.",
            "Key message and tone are distinct.",
            "Deliverables and metrics are actionable.",
        ],
    },
}


ALIASES = {
    "Technical": "Technical Spec",
    "Business": "Business Plan",
    "Story Generator": "Story",
}


def normalize_doc_type(doc_type: str) -> str:
    clean = (doc_type or "").strip()
    return ALIASES.get(clean, clean)


def get_document_preset(doc_type: str) -> Dict[str, List[str] | str]:
    normalized = normalize_doc_type(doc_type)
    return DOCUMENT_PRESETS.get(normalized, DEFAULT_PRESET)


def get_required_sections(doc_type: str) -> List[str]:
    preset = get_document_preset(doc_type)
    return list(preset.get("sections", DEFAULT_PRESET["sections"]))


def format_preset_for_prompt(doc_type: str) -> str:
    preset = get_document_preset(doc_type)
    sections = "\n".join(f"- {section}" for section in preset.get("sections", []))
    rules = "\n".join(f"- {rule}" for rule in preset.get("rules", []))
    checklist = "\n".join(f"- {item}" for item in preset.get("checklist", []))
    return (
        f"Preset Label: {preset.get('label', doc_type)}\n"
        f"Recommended Tone: {preset.get('tone', DEFAULT_PRESET['tone'])}\n\n"
        f"Required / Recommended Sections:\n{sections}\n\n"
        f"Output Rules:\n{rules}\n\n"
        f"Quality Checklist:\n{checklist}"
    )
