"""Evidence ledger, normalization, and deterministic conflict detection."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SupportStrength(Enum):
    DIRECT = "direct"
    INFERRED = "inferred"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"


@dataclass
class EvidenceClaim:
    """A single extracted claim backed by a source."""
    claim: str
    source_id: str
    source_type: str
    published: str
    support: SupportStrength
    raw_value: Optional[str] = None       # original numeric string
    normalized_value: Optional[str] = None
    entity: Optional[str] = None          # company or entity name
    metric: Optional[str] = None          # revenue, margin, etc.
    period: Optional[str] = None          # Q1 FY27, Jun 2026, etc.


@dataclass
class CandidateConflict:
    """Two claims that may conflict — flagged for semantic audit."""
    claim_a: EvidenceClaim
    claim_b: EvidenceClaim
    conflict_type: str   # numeric_mismatch | date_mismatch | entity_ambiguity
    description: str


@dataclass
class EvidenceLedger:
    """Accumulates evidence claims and tracks conflicts and gaps."""
    claims: List[EvidenceClaim] = field(default_factory=list)
    conflicts: List[CandidateConflict] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)

    def add_claim(self, claim: EvidenceClaim) -> None:
        self.claims.append(claim)

    def add_gap(self, description: str) -> None:
        if description not in self.gaps:
            self.gaps.append(description)

    def to_prompt_context(self) -> str:
        """Format the entire ledger for inclusion in an LLM prompt."""
        parts: List[str] = []

        # ── claims ──
        parts.append("=== EVIDENCE CLAIMS ===")
        for i, c in enumerate(self.claims, 1):
            parts.append(
                f"{i}. [{c.source_id}] ({c.support.value}) {c.claim}"
                + (f"  value={c.raw_value}" if c.raw_value else "")
                + (f"  entity={c.entity}" if c.entity else "")
                + (f"  metric={c.metric}" if c.metric else "")
                + (f"  period={c.period}" if c.period else "")
            )

        # ── conflicts ──
        if self.conflicts:
            parts.append("\n=== CANDIDATE CONFLICTS ===")
            for j, cf in enumerate(self.conflicts, 1):
                parts.append(
                    f"{j}. [{cf.conflict_type}] {cf.description}\n"
                    f"   A: [{cf.claim_a.source_id}] {cf.claim_a.claim}\n"
                    f"   B: [{cf.claim_b.source_id}] {cf.claim_b.claim}"
                )

        # ── gaps ──
        if self.gaps:
            parts.append("\n=== EVIDENCE GAPS ===")
            for g in self.gaps:
                parts.append(f"- {g}")

        return "\n".join(parts)


# ======================================================================
# Normalization helpers
# ======================================================================

_CURRENCY_RE = re.compile(
    r"(?:Rs\.?|INR)\s*", re.IGNORECASE
)


def normalize_currency(text: str) -> str:
    """Replace Rs / Rs. / INR with ₹.  Leave crore/lakh as-is."""
    return _CURRENCY_RE.sub("₹", text)


def normalize_percentage(text: str) -> str:
    """Normalise 'bps' → 'basis points', keep % as-is."""
    text = re.sub(r"\bbps\b", "basis points", text, flags=re.IGNORECASE)
    return text


def normalize_claim(claim: EvidenceClaim) -> EvidenceClaim:
    """Apply all normalization to a single claim in place and return it."""
    claim.claim = normalize_currency(normalize_percentage(claim.claim))
    if claim.raw_value:
        claim.normalized_value = normalize_currency(claim.raw_value)
    return claim


# ======================================================================
# Deterministic conflict detection
# ======================================================================

def _key(claim: EvidenceClaim) -> Optional[Tuple[str, str, str]]:
    """Grouping key: (entity, metric, period). None if any part is missing."""
    if claim.entity and claim.metric and claim.period:
        return (
            claim.entity.strip().lower(),
            claim.metric.strip().lower(),
            claim.period.strip().lower(),
        )
    return None


def detect_numeric_conflicts(
    claims: List[EvidenceClaim],
) -> List[CandidateConflict]:
    """Flag claims about the same entity+metric+period with different values.

    This is *deterministic* code — it produces candidates, not verdicts.
    The semantic auditor (an LLM call) decides whether each candidate is
    a genuine conflict, a different-period issue, or an identity mismatch.
    """
    from collections import defaultdict

    groups: Dict[Tuple, List[EvidenceClaim]] = defaultdict(list)
    for c in claims:
        k = _key(c)
        if k is not None and c.raw_value:
            groups[k].append(c)

    conflicts: List[CandidateConflict] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        # Compare every pair
        seen_pairs: set = set()
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pair_key = (a.source_id, b.source_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if a.normalized_value != b.normalized_value:
                    entity, metric, period = key
                    conflicts.append(
                        CandidateConflict(
                            claim_a=a,
                            claim_b=b,
                            conflict_type="numeric_mismatch",
                            description=(
                                f"{metric} for {entity} in {period}: "
                                f"{a.raw_value} [{a.source_id}] vs "
                                f"{b.raw_value} [{b.source_id}]"
                            ),
                        )
                    )
    return conflicts
