"""Configuration for the research agent."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Runtime configuration with budget limits."""

    # Research agent tool-call budget
    max_research_tool_calls: int = 20
    # Maximum correction / revision attempts for draft validation
    max_correction_attempts: int = 3
    # Maximum evidence gaps to surface before stopping research
    max_evidence_gaps: int = 10
    # LLM model name
    model: str = "gemini-2.5-flash"
    # Source mode: "pack_only" or "live"  (live not implemented for fictional tickers)
    source_mode: str = "pack_only"
    # Temperature for generation calls
    generation_temperature: float = 0.2
    # Temperature for evaluation / judge calls
    evaluation_temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            source_mode=os.getenv("SOURCE_MODE", "pack_only"),
            max_research_tool_calls=int(os.getenv("MAX_RESEARCH_CALLS", "20")),
            max_correction_attempts=int(os.getenv("MAX_CORRECTIONS", "3")),
        )