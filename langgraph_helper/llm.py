"""LLM provider factory.

Three providers are supported, all reachable through a plain LangChain
BaseChatModel interface so the rest of the graph never needs to know which
one is in use:

  - gemini      (default) langchain-google-genai, Google AI Studio free tier
  - openrouter  langchain-openai pointed at OpenRouter's OpenAI-compatible
                endpoint, many models with a free tier
  - ollama      langchain-ollama, fully local, no API key, no internet

Note: the LLM call itself still requires internet for gemini/openrouter
(they are hosted APIs). The assignment explicitly allows this in "offline"
mode -- offline refers to not depending on live web search / doc fetching,
not to running with zero network access. Use LLM_PROVIDER=ollama for a
setup with no network dependency at all.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings


def get_llm(provider: str | None = None, temperature: float = 0.1) -> BaseChatModel:
    provider = (provider or settings.llm_provider).lower()

    if provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a free key at "
                "https://openrouter.ai/keys and put it in your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}. Choose one of: gemini, openrouter, ollama."
    )
