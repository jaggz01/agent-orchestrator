# Vibecoded: Agentic Workflow Engine

A Python starter project for an **agentic workflow engine** with three major components:

1. **Agent Orchestrator** (planning and coordination)
2. **Agent Spawner** (DAG execution and agent runtime)
3. **Agent Tool Library** (shared tools and MCP client registry)

This implementation uses **LangChain** and **LangGraph** as requested.

---

## Architecture

### 1) Agent Orchestrator (`orchestrator.py`)
- Main controller that:
  - Takes a user objective.
  - Generates a plan via LLM (or a robust fallback planner).
  - Stores the plan in an in-memory store.
  - Delegates execution to the spawner.
- Supports asynchronous execution and a synchronous helper.

### 2) Agent Spawner (`spawner.py`)
- Implements behavior similar to:
  - `executors.execute(new_agent, system_prompt, list_of_tools)`
- Converts plan steps into a **LangGraph DAG**.
- Executes each spawned agent step (with dependency ordering).
- Resolves and injects only the needed tools from the shared tool library.
- Emits detailed spoolable logs.

### 3) Agent Tool Library (`tool_lib.py`)
- Central registry for:
  - Tool callables
  - MCP clients
- Sub-agents receive tools through the spawner.
- Includes a `default_tool_library()` bootstrap with sample tools.

### Logging
- `logging_utils.py` provides a rotating file logger (`logs/agent_workflow.log`) for durable, spoolable step-by-step logs.

---

## Project Structure

```text
.
├── pyproject.toml
├── README.md
└── src/
    └── agentic_workflow_engine/
        ├── __init__.py
        ├── logging_utils.py
        ├── main.py
        ├── models.py
        ├── orchestrator.py
        ├── spawner.py
        └── tool_lib.py
```

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Run the sample CLI

```bash
python -m agentic_workflow_engine.main "Plan a weekly content strategy for AI blog"
```

You should get a JSON result and logs written to:

```text
logs/agent_workflow.log
```

---

## How planning works

- If an LLM is provided to `AgentOrchestrator`, it attempts to produce structured JSON plan steps.
- If no LLM is provided (or parsing fails), it falls back to a deterministic two-step plan:
  1. Context discovery (`RAG`-style step)
  2. Reasoning/execution step

This gives you a reliable scaffold while integrating your own model provider.

---

## Extending tools and MCP clients

```python
from agentic_workflow_engine.tool_lib import AgentToolLibrary

lib = AgentToolLibrary()
lib.register_tool("search", my_search_tool)
lib.register_mcp_client("jira", jira_mcp_client)
```

Then pass `lib` into the orchestrator.

---

## Next enhancements you can add

- Persist plans in Redis/Postgres instead of in-memory.
- Add retries, timeouts, and circuit breakers per agent step.
- Add richer typed tool contracts with Pydantic schemas.
- Add tenant/user-level configuration for system prompts and allowed tools.
- Introduce human-in-the-loop approval nodes in the LangGraph DAG.

---

## Notes

- This repo is intentionally modular so each component can evolve independently.
- The starter code is ready for you to integrate real LLM providers, RAG index clients, DB connectors, and image tooling.
