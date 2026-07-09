"""One LLM client that works for local Ollama and any OpenAI-compatible endpoint.

Ollama exposes an OpenAI-compatible API at /v1, and so do Groq, vLLM, and others.
So a single `openai` client covers both the free/local dev path and the hosted
demo path — you switch with env vars, not code. See docs/DESIGN.md.
"""
from __future__ import annotations

from src.common.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key
            )
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self._get().chat.completions.create(
            model=self.settings.llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
