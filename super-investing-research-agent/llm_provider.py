"""LLM provider: wraps Google Gemini API with fallback, usage tracking, and logging."""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from logger import agent_logger


@dataclass
class LLMUsage:
    """Tracks token usage and latency across calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_seconds: float = 0.0

    def record(
        self,
        prompt_tok: int = 0,
        completion_tok: int = 0,
        latency: float = 0.0,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_latency_seconds += latency

    def summary(self) -> str:
        return (
            f"calls={self.calls}, "
            f"prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"latency={self.total_latency_seconds:.2f}s"
        )


class LLMProvider:
    """Wraps Google Gemini API for structured LLM calls with resilient fallback."""

    def __init__(
        self,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.fallback_models = fallback_models or [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
        ]
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.usage = LLMUsage()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Make a single LLM call and return the text response.

        Attempts the configured primary model, and falls back to alternate
        models if temporary capacity (503) or availability (404) errors occur.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY not set in environment or .env")

        candidate_models = [self.model]
        for m in self.fallback_models:
            if m not in candidate_models:
                candidate_models.append(m)

        last_error = None
        for model_name in candidate_models:
            start_t = time.time()
            try:
                agent_logger.debug(
                    f"Calling model '{model_name}' (temp={temperature}, prompt_len={len(user_prompt)})"
                )
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=system_prompt + "\n\n" + user_prompt,
                    config={"temperature": temperature},
                )
                elapsed = time.time() - start_t

                # Extract token usage if returned by API
                prompt_tok = 0
                compl_tok = 0
                um = getattr(response, "usage_metadata", None)
                if um:
                    prompt_tok = getattr(um, "prompt_token_count", 0) or 0
                    compl_tok = getattr(um, "candidates_token_count", 0) or 0

                self.usage.record(prompt_tok, compl_tok, elapsed)
                agent_logger.info(
                    f"LLM call [{model_name}] succeeded in {elapsed:.2f}s "
                    f"({prompt_tok}+{compl_tok} tokens)"
                )
                return response.text

            except Exception as e:
                elapsed = time.time() - start_t
                last_error = e
                agent_logger.warning(
                    f"LLM call [{model_name}] failed after {elapsed:.2f}s: {e}. "
                    f"Trying fallback..."
                )
                time.sleep(0.5)

        # All model candidates exhausted
        agent_logger.error(f"All LLM candidate models failed. Last error: {last_error}")
        raise RuntimeError(f"All model candidates failed: {last_error}")