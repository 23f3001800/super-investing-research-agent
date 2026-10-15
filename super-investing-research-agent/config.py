"""Configuration for the research agent."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentConfig:
    """Runtime configuration with budget limits and model fallback options."""

    # Research agent tool-call budget
    max_research_tool_calls: int = 15
    # Maximum correction / revision attempts for draft validation
    max_correction_attempts: int = 3
    # Maximum evidence gaps to surface before stopping research
    max_evidence_gaps: int = 10
    # Primary LLM model name (defaults to verified available flash-lite model)
    model: str = "gemini-3.5-flash-lite"
    # Fallback model chain in case of temporary 503 capacity issues or 404
    fallback_models: List[str] = field(
        default_factory=lambda: ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    )
    # Source mode: "pack_only" or "live"  (live not implemented for fictional tickers)
    source_mode: str = "pack_only"
    # Temperature for generation calls
    generation_temperature: float = 0.2
    # Temperature for evaluation / judge calls
    evaluation_temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "AgentConfig":
        env_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return cls(
            model=env_model,
            source_mode=os.getenv("SOURCE_MODE", "pack_only"),
            max_research_tool_calls=int(os.getenv("MAX_RESEARCH_CALLS", "15")),
            max_correction_attempts=int(os.getenv("MAX_CORRECTIONS", "3")),
        )