"""Shared helpers: config loading, path resolution, logging setup and the OpenAI client."""
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import openai
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class APIError(RuntimeError):
    """An OpenAI API call failed. The message is safe to show to the user."""


def resolve_path(path: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path under the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and the secrets in .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(resolve_path(path or "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def openai_client(timeout: float = 120.0, max_retries: int = 5) -> openai.OpenAI:
    """Create an OpenAI client from OPENAI_API_KEY, which load_config() reads from .env.

    The SDK itself retries rate limits, timeouts, connection errors and server errors with backoff.
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise APIError("OPENAI_API_KEY is not set. Put OPENAI_API_KEY=sk-... in the .env file.")
    return openai.OpenAI(timeout=timeout, max_retries=max_retries)


@contextmanager
def openai_errors(model: str):
    """Turn OpenAI SDK exceptions into an APIError with a message the user can act on."""
    try:
        yield
    except openai.AuthenticationError as e:
        raise APIError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from e
    except openai.PermissionDeniedError as e:
        raise APIError(f"This API key is not allowed to use {model}: {e.message}") from e
    except openai.NotFoundError as e:
        raise APIError(f"Model {model!r} was not found. Check the model names in config.yaml.") from e
    except openai.RateLimitError as e:
        # Also what an account without credit gets ("You exceeded your current quota").
        raise APIError(f"OpenAI rate limit or quota reached: {e.message}") from e
    except openai.APIStatusError as e:
        raise APIError(f"OpenAI API error {e.status_code}: {e.message}") from e
    except openai.APIConnectionError as e:
        raise APIError("Could not reach the OpenAI API. Check your internet connection.") from e


def setup_logging(config: dict[str, Any]) -> None:
    """Log to the console and to the file named in config.yaml."""
    log_file = resolve_path(config["logging"]["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. uvicorn reload)
        return
    root.setLevel(config["logging"]["level"])
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    # These libraries log every HTTP request / telemetry event at INFO.
    for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "datasets", "numexpr", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def batched(items: list, size: int) -> Iterator[list]:
    """Yield consecutive slices of `items` with at most `size` elements."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def unique(items: Iterable) -> list:
    """Drop duplicates while keeping first-seen order."""
    return list(dict.fromkeys(items))
