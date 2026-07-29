"""Deterministic document quality reporting helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .document_presets import get_required_sections

VAGUE_TERMS = [
    "etc",
    "as needed",
    "user-friendly",
    "robust",
    "fast",
    "secure",
    "scalable",
    "easy",
    "appropriate",
    "various",
]


def _headings(content: str) -> List[str]:
    return [match.group(2).strip() for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)]


def _requirement_lines(content: str) -> List[str]:
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if re.search(r"\b(FR|NFR|REQ|AC)-?\d+\b", stripped, re.IGNORECASE):
            lines.append(stripped)
        elif re.match(r"^[-*]\s+(the system|system|user|admin|customer)\b", stripped, re.IGNORECASE):
            lines.append(stripped)
    return lines


def _count_section_terms(content: str, terms: List[str]) -> int:
    lowered = content.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def generate_quality_report(content: str, doc_type: str = "document") -> Dict[str, Any]:
    """Return a lightweight quality report without mutating document data."""
    content = content or ""
    doc_type = doc_type or "document"
    words = re.findall(r"\b\w+\b", content)
    headings = _headings(content)
    requirement_lines = _requirement_lines(content)

    required = get_required_sections(doc_type)
    missing_sections = [section for section in required if section.lower() not in content.lower()]

    vague_hits = []
    lowered = content.lower()
    for term in VAGUE_TERMS:
        count = len(re.findall(rf"\b{re.escape(term)}\b", lowered))
        if count:
            vague_hits.append({"term": term, "count": count})

    duplicate_headings = sorted({heading for heading in headings if headings.count(heading) > 1})
    has_tables = "|" in content and "---" in content
    has_acceptance = "acceptance" in lowered or "criteria" in lowered
    has_security = "security" in lowered or "authentication" in lowered or "authorization" in lowered
    has_nfr = bool(re.search(r"\b(non[- ]functional|performance|availability|reliability|scalability)\b", lowered))

    issues = []
    suggestions = []

    if len(words) < 500:
        issues.append("Document is short and may not contain enough implementation detail.")
        suggestions.append("Expand each major section with assumptions, constraints, and acceptance criteria.")
    if len(headings) < 4:
        issues.append("Document has few Markdown headings.")
        suggestions.append("Break the document into clearer sections and subsections.")
    if missing_sections:
        issues.append(f"Missing expected sections: {', '.join(missing_sections)}.")
        suggestions.append("Add the missing sections or explicitly explain why they are not applicable.")
    if len(requirement_lines) < 5 and doc_type.upper() in {"SRS", "SOW", "TECHNICAL SPEC"}:
        issues.append("Few explicit requirement-style statements were detected.")
        suggestions.append("Add numbered requirements such as FR-001, NFR-001, and acceptance criteria.")
    if not has_acceptance and doc_type.upper() in {"SRS", "SOW", "TECHNICAL SPEC"}:
        issues.append("Acceptance criteria are missing or weak.")
        suggestions.append("Add measurable acceptance criteria for important requirements.")
    if not has_security and doc_type.upper() in {"SRS", "TECHNICAL SPEC"}:
        issues.append("Security considerations are not clearly covered.")
        suggestions.append("Add authentication, authorization, data protection, and audit requirements.")
    if not has_nfr and doc_type.upper() in {"SRS", "TECHNICAL SPEC"}:
        issues.append("Non-functional requirements are not clearly covered.")
        suggestions.append("Add performance, reliability, scalability, usability, and availability targets.")
    if duplicate_headings:
        issues.append(f"Duplicate headings detected: {', '.join(duplicate_headings[:5])}.")
        suggestions.append("Merge duplicate sections or rename them to clarify their scope.")
    if vague_hits:
        suggestions.append("Replace vague wording with measurable, testable statements.")
    if not has_tables and doc_type.upper() in {"SRS", "SOW", "TECHNICAL SPEC"}:
        suggestions.append("Consider adding summary tables for requirements, risks, or traceability.")

    completeness = 100
    completeness -= min(35, len(missing_sections) * 8)
    completeness -= 15 if len(words) < 500 else 0
    completeness -= 10 if len(headings) < 4 else 0
    completeness -= 10 if len(requirement_lines) < 5 and doc_type.upper() in {"SRS", "SOW", "TECHNICAL SPEC"} else 0
    completeness -= min(10, len(vague_hits) * 2)
    completeness = max(0, min(100, completeness))

    return {
        "score": completeness,
        "word_count": len(words),
        "heading_count": len(headings),
        "requirement_count": len(requirement_lines),
        "expected_sections": required,
        "missing_sections": missing_sections,
        "duplicate_headings": duplicate_headings,
        "vague_terms": vague_hits,
        "has_tables": has_tables,
        "issues": issues,
        "suggestions": suggestions,
    }


def format_quality_report_markdown(report: Dict[str, Any]) -> str:
    """Format a quality report dict as Markdown."""
    lines = [
        f"### Document Quality Report",
        "",
        f"**Score:** {report.get('score', 0)}/100",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Words | {report.get('word_count', 0)} |",
        f"| Headings | {report.get('heading_count', 0)} |",
        f"| Requirement-like lines | {report.get('requirement_count', 0)} |",
        f"| Tables detected | {'Yes' if report.get('has_tables') else 'No'} |",
    ]

    if report.get("missing_sections"):
        lines.extend(["", "**Missing sections**"])
        lines.extend(f"- {section}" for section in report["missing_sections"])

    if report.get("issues"):
        lines.extend(["", "**Issues**"])
        lines.extend(f"- {issue}" for issue in report["issues"])

    if report.get("suggestions"):
        lines.extend(["", "**Suggestions**"])
        lines.extend(f"- {suggestion}" for suggestion in report["suggestions"])

    if report.get("vague_terms"):
        lines.extend(["", "**Vague terms detected**", "", "| Term | Count |", "|---|---:|"])
        lines.extend(f"| {item['term']} | {item['count']} |" for item in report["vague_terms"])

    return "\n".join(lines)
