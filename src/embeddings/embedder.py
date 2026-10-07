"""Embeddings: turn text into vectors with OpenAI's embedding API."""
import logging

import numpy as np
import openai

from src.utils.helpers import batched, openai_client, openai_errors

logger = logging.getLogger(__name__)

# USD per million tokens, from https://developers.openai.com/api/docs/pricing (checked 2026-10-07).
PRICES_PER_MTOK = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


class Embedder:
    """Wraps the OpenAI embeddings endpoint.

    The API returns vectors normalised to length 1, so their dot product is the cosine similarity.
    """

    def __init__(self, model: str, batch_size: int = 256, dimensions: int | None = None, client: openai.OpenAI | None = None):
        # Embedding requests normally return in a second or two, so a short timeout notices a
        # dropped connection quickly and lets the SDK retry.
        self.client = client or openai_client(timeout=30.0)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.tokens_used = 0            # running total of billed tokens

    @property
    def cost_usd(self) -> float | None:
        """What the embedding calls made so far cost, or None if the model's price is not in the table."""
        if self.model not in PRICES_PER_MTOK:
            return None
        return self.tokens_used * PRICES_PER_MTOK[self.model] / 1_000_000

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed texts, `batch_size` per API request. Texts must not be empty."""
        options = {"dimensions": self.dimensions} if self.dimensions else {}
        vectors = []
        for batch in batched(texts, self.batch_size):
            with openai_errors(self.model):
                response = self.client.embeddings.create(model=self.model, input=batch, **options)
            self.tokens_used += response.usage.total_tokens
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query."""
        return self.embed_documents([text])[0]
