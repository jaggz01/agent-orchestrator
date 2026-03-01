"""Main orchestration agent that plans and triggers concurrent agent spawning."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .logging_utils import build_spoolable_logger
from .models import AgentSpec, PlanStep, WorkflowPlan
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

    def _build_fallback_steps(self) -> list[PlanStep]:
        available = self.tool_library.available_capabilities()
        steps: list[PlanStep] = []

        if "rag" in available:
            steps.append(
                PlanStep(
                    id="retrieve_context",
                    description="Collect relevant context for objective",
                    agent=AgentSpec(
                        name="context_worker",
                        system_prompts=["Gather and summarize relevant context."],
                        required_capabilities=["rag"],
                    ),
                )
            )

        if "database" in available:
            steps.append(
                PlanStep(
                    id="fetch_structured_data",
                    description="Fetch structured data required for the objective",
                    agent=AgentSpec(
                        name="data_worker",
                        system_prompts=["Retrieve structured information from data sources."],
                        required_capabilities=["database"],
                    ),
                    depends_on=[steps[-1].id] if steps else [],
                )
            )

        if "image" in available:
            steps.append(
                PlanStep(
                    id="image_processing",
                    description="Process image requirements for the objective",
                    agent=AgentSpec(
                        name="image_worker",
                        system_prompts=["Handle image transformations and extraction."],
                        required_capabilities=["image"],
                    ),
                    depends_on=[steps[-1].id] if steps else [],
                )
            )

        previous_step = steps[-1].id if steps else None
        final_step = PlanStep(
            id="synthesize_answer",
            description="Synthesize outputs and return final result",
            agent=AgentSpec(
                name="synthesis_worker",
                system_prompts=["Combine outputs and respond clearly."],
                required_capabilities=["generic"],
            ),
            depends_on=[previous_step] if previous_step else [],
        )
        steps.append(final_step)
        return steps

    def _fallback_plan(self, objective: str) -> WorkflowPlan:
        """Capability-aware fallback plan when LLM is absent or malformed."""

        return WorkflowPlan(objective=objective, steps=self._build_fallback_steps())

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

        tool_catalog = self.tool_library.list_tool_catalog()
        prompt = (
            "Create a JSON object with key 'steps'. "
            "Each step must include id, description, "
            "agent{name, system_prompts, optional tool_names, optional required_capabilities}, "
            "and optional depends_on list. "
            "Do not use hardcoded agent types; infer required_capabilities from available tools. "
            f"Available tools: {json.dumps(tool_catalog)}. "
            f"Objective: {objective}"
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
