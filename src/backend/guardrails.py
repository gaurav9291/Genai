"""Lightweight AI safety checks for user-controlled prompts."""

import re
from dataclasses import dataclass
from typing import Iterable, List


INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|above|prior) (instructions|prompts|rules)\b",
    r"\bdisregard (all )?(previous|above|prior) (instructions|prompts|rules)\b",
    r"\breveal (the )?(system|developer) (prompt|message|instructions)\b",
    r"\bprint (the )?(system|developer) (prompt|message|instructions)\b",
    r"\byou are now\b",
    r"\bact as\b.*\bwithout restrictions\b",
    r"\bdo not follow\b.*\b(system|developer|instructions)\b",
    r"\bexfiltrate\b",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: List[str]


def detect_prompt_injection(text: str) -> GuardrailResult:
    """Detect common prompt-injection instructions with conservative regex rules."""
    if not text:
        return GuardrailResult(allowed=True, reasons=[])

    reasons = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            reasons.append(pattern)

    return GuardrailResult(allowed=not reasons, reasons=reasons)


def check_inputs(values: Iterable[str]) -> GuardrailResult:
    """Check multiple input fields and merge prompt-injection findings."""
    reasons = []
    for value in values:
        result = detect_prompt_injection(value or "")
        reasons.extend(result.reasons)

    return GuardrailResult(allowed=not reasons, reasons=reasons)
