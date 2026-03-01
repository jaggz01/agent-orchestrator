"""Main orchestration agent that plans and triggers concurrent agent spawning."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .logging_utils import build_spoolable_logger
from .models import AgentSpec, AgentType, PlanStep, WorkflowPlan
from .spawner import AgentSpawner
from .tool_lib import AgentToolLibrary


@dataclass
class InMemoryPlanStore:
    """Simple in-memory objective->plan storage."""

    plans: dict[str, WorkflowPlan] = field(default_factory=dict)

    def put(self, plan: WorkflowPlan) -> None:
        self.plans[plan.objective] = plan

    def get(self, objective: str) -> WorkflowPlan | None:
        return self.plans.get(objective)


class AgentOrchestrator:
    """Orchestrator that uses LLM (or fallback logic) to create workflow plans."""

    def __init__(
        self,
        tool_library: AgentToolLibrary,
        llm: BaseChatModel | None = None,
        log_path: str = "logs/agent_workflow.log",
    ) -> None:
        self.tool_library = tool_library
        self.llm = llm
        self.logger = build_spoolable_logger(log_path)
        self.plan_store = InMemoryPlanStore()
        self.spawner = AgentSpawner(tool_library=tool_library, log_path=log_path)

    def _fallback_plan(self, objective: str) -> WorkflowPlan:
        """Rule-based default plan used when LLM is absent or malformed."""

        steps = [
            PlanStep(
                id="discover_context",
                description="Gather relevant context for objective",
                agent=AgentSpec(
                    name="rag_researcher",
                    agent_type=AgentType.RAG,
                    system_prompts=["Retrieve and summarize context."],
                    tool_names=["echo"],
                ),
            ),
            PlanStep(
                id="reason_and_execute",
                description="Reason over context and produce actionable answer",
                agent=AgentSpec(
                    name="generic_executor",
                    agent_type=AgentType.GENERIC,
                    system_prompts=["Synthesize and execute the task."],
                    tool_names=["word_count"],
                ),
                depends_on=["discover_context"],
            ),
        ]
        return WorkflowPlan(objective=objective, steps=steps)

    def _parse_plan_payload(self, objective: str, payload: str) -> WorkflowPlan:
        parsed = json.loads(payload)
        steps = [PlanStep.model_validate(step) for step in parsed.get("steps", [])]
        if not steps:
            raise ValueError("LLM returned an empty plan")
        return WorkflowPlan(objective=objective, steps=steps)

    async def plan(self, objective: str) -> WorkflowPlan:
        if not self.llm:
            self.logger.info("No LLM provided, using fallback planner")
            plan = self._fallback_plan(objective)
            self.plan_store.put(plan)
            return plan

        prompt = (
            "Create a JSON object with key 'steps'. "
            "Each step must include id, description, agent{name, agent_type, system_prompts, tool_names}, "
            "and optional depends_on list."
            f" Objective: {objective}"
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            plan = self._parse_plan_payload(objective, response.content)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("LLM planner failed (%s). Falling back.", exc)
            plan = self._fallback_plan(objective)

        self.plan_store.put(plan)
        self.logger.info("Plan created with %d steps", len(plan.steps))
        return plan

    async def execute_objective(self, objective: str) -> dict[str, Any]:
        """Plan and execute, spawning all step agents via the spawner."""

        plan = await self.plan(objective)
        result = await self.spawner.execute(plan)
        return result

    def execute_objective_sync(self, objective: str) -> dict[str, Any]:
        """Synchronous helper for quick local usage and scripts."""

        return asyncio.run(self.execute_objective(objective))
