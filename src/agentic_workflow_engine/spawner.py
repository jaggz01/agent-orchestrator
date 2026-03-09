"""Agent spawner that materializes plan steps into a LangGraph DAG and executes nodes."""

from __future__ import annotations

from typing import Any

from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

from .logging_utils import build_spoolable_logger
from .models import PlanStep, WorkflowPlan
from .tool_lib import AgentToolLibrary


class SpawnState(TypedDict, total=False):
    objective: str
    results: dict[str, str]


class AgentSpawner:
    """Executor/spawner that runs child agents based on plan DAG dependencies."""

    def __init__(self, tool_library: AgentToolLibrary, log_path: str = "logs/agent_workflow.log") -> None:
        self.tool_library = tool_library
        self.logger = build_spoolable_logger(log_path)

    def _resolve_core_capabilities(self, step: PlanStep) -> dict[str, object]:
        context: dict[str, object] = {}
        for capability in step.agent.core_capabilities:
            if capability == "rag_connection":
                rag_db = self.tool_library.mcp_clients.get("rag_db")
                if rag_db is None:
                    raise RuntimeError("Agent requested 'rag_connection' but no rag_db client is registered")
                context[capability] = rag_db
            else:
                self.logger.warning("Unknown core capability requested: %s", capability)
        return context

    async def _execute_agent(self, step: PlanStep, state: SpawnState) -> dict[str, Any]:
        self.logger.info("Spawning agent %s (%s)", step.agent.name, step.agent.agent_type.value)
        tools = self.tool_library.resolve_tools(step.agent.tool_names)
        core_context = self._resolve_core_capabilities(step)

        content = f"{step.description} | objective={state.get('objective', '')}"
        tool_outputs = []
        if "rag_connection" in core_context:
            tool_outputs.append("core:rag_connection=connected")

        for tool_name, tool_callable in tools.items():
            tool_output = tool_callable(content)
            tool_outputs.append(f"{tool_name}: {tool_output}")
            self.logger.info("Agent=%s tool=%s output=%s", step.agent.name, tool_name, tool_output)

        result = "\n".join(tool_outputs) if tool_outputs else "No tools executed"
        return {"results": {step.id: result}}

    def build_graph(self, plan: WorkflowPlan):
        graph = StateGraph(SpawnState)

        for step in plan.steps:
            async def node(state: SpawnState, _step: PlanStep = step):
                return await self._execute_agent(_step, state)

            graph.add_node(step.id, node)

        for step in plan.steps:
            if not step.depends_on:
                graph.add_edge(START, step.id)
            for parent in step.depends_on:
                graph.add_edge(parent, step.id)

        leaf_ids = {
            step.id
            for step in plan.steps
            if step.id not in {dep for s in plan.steps for dep in s.depends_on}
        }
        for leaf in leaf_ids:
            graph.add_edge(leaf, END)

        return graph.compile()

    async def execute(self, plan: WorkflowPlan) -> SpawnState:
        self.logger.info("Executing plan for objective: %s", plan.objective)
        runnable = self.build_graph(plan)
        result = await runnable.ainvoke({"objective": plan.objective, "results": {}})
        self.logger.info("Execution complete")
        return result
