"""Configuration loading and model initialization for the workflow engine."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


DEFAULT_CONFIG_PATH = "llm.config"


@dataclass
class LLMSettings:
    """Settings used to instantiate a LangChain BaseChatModel."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None


@dataclass
class RAGSettings:
    """Settings for connecting to the RAG datastore and retrieval behavior."""

    enabled: bool
    provider: str
    database_path: str
    collection: str
    semantic_search_threshold: float
    top_k: int


@dataclass
class AppConfig:
    """Top-level app config composed of LLM and RAG settings."""

    llm: LLMSettings
    rag: RAGSettings


def _require_section(parser: configparser.ConfigParser, name: str) -> configparser.SectionProxy:
    if not parser.has_section(name):
        raise ValueError(f"Missing required section '[{name}]' in llm.config")
    return parser[name]


def _require_str(section: configparser.SectionProxy, key: str) -> str:
    value = section.get(key)
    if value is None or value.strip() == "":
        section_name = section.name if hasattr(section, "name") else "unknown"
        raise ValueError(f"Missing required config value '{key}' in section '[{section_name}]'")
    return value.strip()


def _optional_float(section: configparser.SectionProxy, key: str) -> float | None:
    value = section.get(key)
    return float(value) if value not in (None, "") else None


def _optional_int(section: configparser.SectionProxy, key: str) -> int | None:
    value = section.get(key)
    return int(value) if value not in (None, "") else None


def load_app_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load app config from llm.config and fail fast for missing required values."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Required configuration file not found at '{path}'. Create llm.config with [llm] and [rag] sections."
        )

    parser = configparser.ConfigParser()
    parser.read(config_path)

    llm_section = _require_section(parser, "llm")
    rag_section = _require_section(parser, "rag")

    llm = LLMSettings(
        provider=_require_str(llm_section, "provider"),
        model=_require_str(llm_section, "model"),
        base_url=llm_section.get("base_url") or None,
        api_key=llm_section.get("api_key") or None,
        temperature=_optional_float(llm_section, "temperature"),
        max_tokens=_optional_int(llm_section, "max_tokens"),
        timeout=_optional_float(llm_section, "timeout"),
    )

    rag = RAGSettings(
        enabled=_require_str(rag_section, "enabled").lower() in {"1", "true", "yes", "on"},
        provider=_require_str(rag_section, "provider"),
        database_path=_require_str(rag_section, "database_path"),
        collection=_require_str(rag_section, "collection"),
        semantic_search_threshold=float(_require_str(rag_section, "semantic_search_threshold")),
        top_k=int(_require_str(rag_section, "top_k")),
    )

    return AppConfig(llm=llm, rag=rag)


def build_default_llm(config: AppConfig | None = None, config_path: str = DEFAULT_CONFIG_PATH) -> BaseChatModel:
    """Build a BaseChatModel from llm.config settings."""

    app_config = config or load_app_config(config_path)
    settings = app_config.llm

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "model_provider": settings.provider,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    if settings.timeout is not None:
        kwargs["timeout"] = settings.timeout

    return init_chat_model(**kwargs)
