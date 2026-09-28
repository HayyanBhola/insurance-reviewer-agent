"""
One place that decides which LLM the whole project uses.
Change LLM_PROVIDER in .env to switch between "openai" and "gemini".
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def _models(temperature=0, openai_model=None):
    """Return a list of chat models: [main, backup, ...]."""
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "openai":
        return [init_chat_model(
            openai_model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            model_provider="openai",
            temperature=temperature,
            max_retries=3,
        )]

    return [
        init_chat_model(
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            model_provider="google_genai",
            temperature=temperature,
            max_retries=3,
        ),
        init_chat_model(
            os.getenv("GEMINI_BACKUP_MODEL", "gemini-3.1-flash-lite"),
            model_provider="google_genai",
            temperature=temperature,
            max_retries=3,
        ),
    ]


def _with_fallbacks(runnables):
    return runnables[0].with_fallbacks(runnables[1:]) if len(runnables) > 1 else runnables[0]


def get_llm(temperature=0):
    """A plain chat model (returns messages)."""
    return _with_fallbacks(_models(temperature))


def get_structured_llm(schema, temperature=0, openai_model=None):
    """A chat model that always returns an instance of the given Pydantic schema.
    openai_model: optionally use a different OpenAI model for this task."""
    return _with_fallbacks([m.with_structured_output(schema)
                            for m in _models(temperature, openai_model)])
