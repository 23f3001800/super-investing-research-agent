"""LLM provider: wraps Google Gemini API with usage tracking."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMUsage:
    """Tracks token usage across calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, prompt_tok: int = 0, completion_tok: int = 0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok

    def summary(self) -> str:
        return (
            f"LLM calls: {self.calls}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"completion tokens: {self.completion_tokens}"
        )


class LLMProvider:
    """Wraps Google Gemini API for structured LLM calls."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

        Raises RuntimeError when the API key is absent.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY not set")

        response = self.client.models.generate_content(
            model=self.model,
            contents=system_prompt + "\n\n" + user_prompt,
            config={"temperature": temperature},
        )

        # Track usage when the metadata is available
        um = getattr(response, "usage_metadata", None)
        if um:
            self.usage.record(
                prompt_tok=getattr(um, "prompt_token_count", 0) or 0,
                completion_tok=getattr(um, "candidates_token_count", 0) or 0,
            )
        else:
            self.usage.record()

        return response.text
    