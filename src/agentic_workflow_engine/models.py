"""Core pydantic models for orchestrator planning and agent execution."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Supported agent categories."""

    RAG = "rag"
    DB = "db"
    IMAGE = "image"
    GENERIC = "generic"


class AgentSpec(BaseModel):
    """A spawned agent configuration."""

    name: str = Field(..., description="Unique agent name")
    agent_type: AgentType = Field(default=AgentType.GENERIC)
    system_prompts: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    """Single step in an orchestrated plan."""

    id: str
    description: str
    agent: AgentSpec
    depends_on: list[str] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    """Plan with DAG-ready steps."""

    objective: str
    steps: list[PlanStep]


ToolCallable = Callable[[str], str]
