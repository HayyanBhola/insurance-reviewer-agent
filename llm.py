import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def get_llm(temperature=0):
    """Return the LLM chosen in .env. Change LLM_PROVIDER to switch."""
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "openai":
        return init_chat_model(
            os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            model_provider="openai",
            temperature=temperature,
            max_retries=3,
        )

    main = init_chat_model(
        os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        model_provider="google_genai",
        temperature=temperature,
        max_retries=3,
    )
    backup = init_chat_model(
        os.getenv("GEMINI_BACKUP_MODEL", "gemini-3.1-flash-lite"),
        model_provider="google_genai",
        temperature=temperature,
        max_retries=3,
    )
    return main.with_fallbacks([backup])
