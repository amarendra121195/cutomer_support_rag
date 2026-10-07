"""LLM client: every call to the chat model goes through here."""
import logging
from dataclasses import dataclass

import openai

from src.utils.helpers import openai_client, openai_errors

logger = logging.getLogger(__name__)

# USD per million tokens (input, output), from https://developers.openai.com/api/docs/pricing
# (checked 2026-10-07). Used to report what an evaluation run cost; update when prices change.
PRICES_PER_MTOK = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.3-codex": (1.75, 14.00),
}


@dataclass
class LLMResponse:
    text: str
    model: str                  # the model that produced the response, as reported by the API
    status: str                 # "ok", "truncated" (ran out of output tokens) or "refused"
    input_tokens: int
    output_tokens: int          # includes the model's reasoning tokens

    @property
    def cost_usd(self) -> float | None:
        """Cost of this call, or None when the model's price is not in the table."""
        if self.model not in PRICES_PER_MTOK:
            return None
        input_price, output_price = PRICES_PER_MTOK[self.model]
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000


class LLMClient:
    """Sends one system + user prompt to an OpenAI model and returns the text of the reply."""

    def __init__(self, model: str, max_output_tokens: int = 16000, effort: str = "medium", client: openai.OpenAI | None = None):
        self.client = client or openai_client()
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.effort = effort

    def generate(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        """Run one request. With `json_schema`, the reply text is JSON matching that schema."""
        request = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "reasoning": {"effort": self.effort},
            "max_output_tokens": self.max_output_tokens,
        }
        if json_schema:
            request["text"] = {
                "format": {"type": "json_schema", "name": "structured_output", "schema": json_schema, "strict": True}
            }
        with openai_errors(self.model):
            response = self.client.responses.create(**request)

        status, text = self._read(response)
        if status != "ok":
            logger.warning("%s returned a %s response", response.model, status)
        return LLMResponse(
            text=text,
            model=response.model,
            status=status,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

    @staticmethod
    def _read(response) -> tuple[str, str]:
        """Classify a response and extract its text."""
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        return "refused", part.refusal
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            # The other reason is "content_filter": the output was blocked.
            return ("truncated" if reason == "max_output_tokens" else "refused"), response.output_text
        return "ok", response.output_text
