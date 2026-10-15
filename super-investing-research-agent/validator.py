"""Deterministic validation for research briefs."""

import re
from typing import List, Tuple

REQUIRED_SECTIONS = [
    "## Snapshot",
    "## Bull case",
    "## Bear case",
    "## Open questions",
    "## Sources",
]


def validate_brief(brief: str, registry) -> Tuple[bool, List[str]]:
    """Run all hard checks on a brief.

    Returns (passed, list_of_errors).
    """
    errors: List[str] = []

    # 1. Required sections
    for section in REQUIRED_SECTIONS:
        if section not in brief:
            errors.append(f"Missing required section: {section}")

    # 2. Citation-ID validity
    cited_ids = {f"S{n}" for n in re.findall(r"\[S(\d+)\]", brief)}
    _valid, unknown = registry.validate_citation_ids(cited_ids)
    for uid in sorted(unknown):
        errors.append(f"Unknown citation ID: [{uid}]")

    # 3. Non-empty
    if len(brief.strip()) < 100:
        errors.append("Brief is too short (< 100 characters)")

    # 4. Excessive length
    if len(brief.strip()) > 10_000:
        errors.append(f"Brief is excessively long ({len(brief)} characters)")

    # 5. Sources section has content
    sources_match = re.search(r"## Sources\s*\n(.+)", brief, re.DOTALL)
    if sources_match:
        if len(sources_match.group(1).strip()) < 20:
            errors.append("Sources section appears empty")

    # 6. Prompt-injection artifacts (outside Open questions is suspicious)
    injection_markers = [
        "STRONG BUY",
        "analyst consensus is positive",
        "ignore all previous instructions",
        "ignore previous instructions",
    ]
    oq_match = re.search(r"## Open questions(.+?)(?=## |$)", brief, re.DOTALL)
    oq_text = oq_match.group(1).lower() if oq_match else ""

    for marker in injection_markers:
        if marker.lower() in brief.lower() and marker.lower() not in oq_text:
            errors.append(f"Potential prompt injection artifact: '{marker}'")

    return len(errors) == 0, errors


def extract_claims_with_citations(brief: str) -> List[dict]:
    """Extract individual claims and their citation IDs from the brief.

    Returns list of {"claim": str, "citations": [str], "section": str}.
    """
    claims: List[dict] = []
    current_section = "unknown"

    for line in brief.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        if not line or line.startswith("# "):
            continue

        citations = [f"S{c}" for c in re.findall(r"\[S(\d+)\]", line)]
        if citations:
            claim_text = re.sub(r"\s*\[S\d+\]", "", line).strip()
            claim_text = re.sub(r"^[-*•]\s*", "", claim_text).strip()
            if claim_text and len(claim_text) > 10:
                claims.append(
                    {
                        "claim": claim_text,
                        "citations": citations,
                        "section": current_section,
                    }
                )
    return claims
