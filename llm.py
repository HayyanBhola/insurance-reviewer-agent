"""
One place that decides which LLM the whole project uses.
Change LLM_PROVIDER in .env to switch between "openai" and "gemini".

Every structured AI call goes through structured_call(), which adds a local cache:
the same model + same prompt is answered from outputs/llm_cache.sqlite for free.
Set LLM_CACHE=0 in .env to turn the cache off.
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

CACHE_FILE = Path("outputs/llm_cache.sqlite")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def provider():
    return os.getenv("LLM_PROVIDER", "gemini")


def model_id(openai_model=None):
    """A readable name for the model that will answer, e.g. 'openai:gpt-5.4-nano'."""
    if provider() == "openai":
        return "openai:" + (openai_model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    return "google_genai:" + os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def _models(temperature=0, openai_model=None):
    """Return a list of chat models: [main, backup, ...]."""
    if provider() == "openai":
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


# ---------------------------------------------------------------------------
# Cached structured calls
# ---------------------------------------------------------------------------
def _cache_on():
    return os.getenv("LLM_CACHE", "1") != "0"


def _db():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
    return con


def _cache_key(schema, messages, mid):
    parts = [mid, schema.__name__, json.dumps(schema.model_json_schema(), sort_keys=True)]
    for m in messages:
        parts.append(m.type)
        parts.append(json.dumps(m.content, sort_keys=True) if not isinstance(m.content, str) else m.content)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# simple counters so reports can show cache hits
CACHE_STATS = {"hits": 0, "misses": 0}


def structured_call(schema, messages, openai_model=None):
    """Call the LLM and return a `schema` instance, using the local cache when possible."""
    mid = model_id(openai_model)
    key = _cache_key(schema, messages, mid) if _cache_on() else None

    if key:
        with _db() as con:
            row = con.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            CACHE_STATS["hits"] += 1
            return schema.model_validate_json(row[0])

    CACHE_STATS["misses"] += 1
    result = get_structured_llm(schema, openai_model=openai_model).invoke(messages)

    if key:
        with _db() as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (key, result.model_dump_json(), time.time()))
    return result


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def embedding_model_name():
    """The embedding model that WOULD be used right now (from .env)."""
    default = ("openai:text-embedding-3-small" if provider() == "openai"
               else "google_genai:gemini-embedding-001")
    return os.getenv("EMBEDDING_MODEL", default)


def get_embeddings():
    """Embedding model for policy search. Returns None if it can't be created,
    in which case the policy search falls back to keyword (BM25) search only.

    Set EMBEDDING_MODEL in .env to override, e.g. "openai:text-embedding-3-small".
    """
    from langchain.embeddings import init_embeddings
    try:
        return init_embeddings(embedding_model_name())
    except Exception as exc:  # missing package, bad key, unknown model...
        print(f"[embeddings] not available ({exc}); using keyword search only.")
        return None
