"""Utilities for initializing the default LLM for orchestration."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def build_default_llm() -> BaseChatModel:
    """Build a default chat model from environment configuration.

    Environment variables:
    - AGENTIC_DEFAULT_LLM_MODEL (default: gpt-4o-mini)
    - AGENTIC_DEFAULT_LLM_PROVIDER (default: openai)
    """

    model = os.getenv("AGENTIC_DEFAULT_LLM_MODEL", "gpt-4o-mini")
    provider = os.getenv("AGENTIC_DEFAULT_LLM_PROVIDER", "openai")
    return init_chat_model(model=model, model_provider=provider)

