"""Shared LangChain chat model factory.

Choose the active model by changing ACTIVE_MODEL_CONFIG below.
Keep API keys and base URLs in .env; keep provider choice in this file.
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel

from .config import settings


def use_ollama(model: str = "llama3.1:8b") -> Dict[str, Any]:
    return {"provider": "ollama", "model": model}


def use_groq(model: str = "openai/gpt-oss-120b") -> Dict[str, Any]:
    return {"provider": "groq", "model": model}


def use_gemini(model: str = "gemini-3.5-flash") -> Dict[str, Any]:
    return {"provider": "gemini", "model": model}


def use_cerebras(model: str = "gpt-oss-120b") -> Dict[str, Any]:
    return {"provider": "cerebras", "model": model}


def use_nvidia_nim(model: str = "nemotron-3-super-120b-a12b") -> Dict[str, Any]:
    return {"provider": "nvidia_nim", "model": model}


def use_mock() -> Dict[str, Any]:
    return {"provider": "mock", "model": "mock"}


# Model selection:
# Uncomment exactly one line below.
# ACTIVE_MODEL_CONFIG = use_ollama("llama3.1:8b")
# ACTIVE_MODEL_CONFIG = use_groq("openai/gpt-oss-120b")
# ACTIVE_MODEL_CONFIG = use_gemini("gemini-3.5-flash")
ACTIVE_MODEL_CONFIG = use_cerebras("gpt-oss-120b")
# ACTIVE_MODEL_CONFIG = use_nvidia_nim("nemotron-3-super-120b-a12b")
# ACTIVE_MODEL_CONFIG = use_groq("llama-3.3-70b-versatile")
# ACTIVE_MODEL_CONFIG = use_mock()


class _MockStructuredWrapper:
    """Thin wrapper returned by MockLLM.with_structured_output().
    
    Calls the underlying MockLLM and attempts to instantiate the target
    Pydantic schema from the text output.  Falls back to a bare-minimum
    schema instance so callers never crash during local development.
    """

    def __init__(self, llm: "MockLLM", schema: Type[BaseModel]):
        self._llm = llm
        self._schema = schema

    def invoke(self, prompt: Any, **kwargs: Any) -> BaseModel:
        raw = self._llm._call(str(prompt))
        try:
            import json
            # Try to extract JSON from the raw string
            json_start = raw.find("{")
            if json_start != -1:
                data = json.loads(raw[json_start:])
                return self._schema(**data)
        except Exception:
            pass
        # Return a minimal valid instance using field defaults
        try:
            return self._schema.model_construct()
        except Exception:
            return self._schema.model_construct(**{
                f: None for f in self._schema.model_fields
            })

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> BaseModel:
        return self.invoke(prompt, **kwargs)


class MockLLM(LLM):
    """Small fallback model for development without API keys or Ollama."""

    mode: str = "document"

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.mode == "review":
            return prompt.split("Content to review:", 1)[-1].strip()
        if self.mode == "qa":
            return (
                "I can answer with real retrieved context once an Ollama or online "
                "model is configured. Relevant context was retrieved successfully."
            )
        return (
            "# Generated Document\n\n"
            "## Overview\n"
            "This is a local fallback draft because no configured LLM was available.\n\n"
            "## Requirements\n"
            "- Add project-specific functional requirements.\n"
            "- Add non-functional requirements.\n"
            "- Add acceptance criteria.\n\n"
            "## Conclusion\n"
            "Review and refine this draft before use.\n"
        )

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> _MockStructuredWrapper:
        """Return a wrapper that mimics LangChain's with_structured_output API."""
        return _MockStructuredWrapper(self, schema)


def get_chat_model(
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    mode: str = "document",
):
    """Return the configured LangChain-compatible chat model."""
    provider = ACTIVE_MODEL_CONFIG["provider"]
    model_name = ACTIVE_MODEL_CONFIG["model"]
    temperature = settings.temperature if temperature is None else temperature
    max_tokens = settings.max_tokens if max_tokens is None else max_tokens

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        # Resolve whichever key is exported (GEMINI_API_KEY or GOOGLE_API_KEY)
        api_key = settings.gemini_api_key or settings.google_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "cerebras":
        from langchain_cerebras import ChatCerebras
        import os

        api_key = settings.cerebras_api_key or os.environ.get("CEREBRAS_API_KEY")

        return ChatCerebras(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "nvidia_nim":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import os

        api_key = settings.nvidia_api_key or os.environ.get("NVIDIA_API_KEY")
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if settings.nvidia_nim_base_url:
            kwargs["base_url"] = settings.nvidia_nim_base_url

        return ChatNVIDIA(**kwargs)

    return MockLLM(mode=mode)


def get_model_status() -> Dict[str, Any]:
    return {
        "provider": ACTIVE_MODEL_CONFIG["provider"],
        "model": ACTIVE_MODEL_CONFIG["model"],
        "ollama_base_url": settings.ollama_base_url,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "embedding_model": settings.embedding_model,
        "vector_store": "pinecone" if settings.pinecone_api_key else "faiss",
    }
