# Agentic Workflow Engine

A Python starter project for an **agentic workflow engine** with three major components:

1. **Agent Orchestrator** (planning and coordination)
2. **Agent Spawner** (DAG execution and agent runtime)
3. **Agent Tool Library** (shared tools and MCP client registry)

This implementation uses **LangChain** and **LangGraph** and is designed for **capability-driven agent spawning**.

---

## Architecture

### 1) Agent Orchestrator (`orchestrator.py`)
- Main controller that:
  - Takes a user objective.
  - Generates a plan via LLM (or capability-aware fallback planner).
  - Stores the plan in an in-memory store.
  - Delegates execution to the spawner.
- **No hard-coded agent types are required.**
  - The orchestrator infers what can be spawned from available tool capabilities such as `image`, `rag`, `database`, and `generic`.

### 2) Agent Spawner (`spawner.py`)
- Implements behavior similar to:
  - `executors.execute(new_agent, system_prompt, list_of_tools)`
- Converts plan steps into a **LangGraph DAG**.
- Executes each spawned agent step (with dependency ordering).
- Resolves tools in either of two ways:
  - explicit `tool_names`, or
  - inferred from `required_capabilities`.
- Emits detailed spoolable logs.

### 3) Agent Tool Library (`tool_lib.py`)
- Central registry for:
  - Capability-tagged tools
  - MCP clients
- Tools can have one or more capabilities, for example:
  - `image`
  - `generic`
  - `rag + database`
- Includes a `default_tool_library()` bootstrap with sample tagged tools.

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

## Capability-tagged tools

Register tools with capabilities so the orchestrator/spawner can infer agent profiles.

```python
from agentic_workflow_engine.tool_lib import AgentToolLibrary

lib = AgentToolLibrary()
lib.register_tool("search", my_search_tool, capabilities=["rag", "generic"])
lib.register_tool("img_edit", my_img_tool, capabilities=["image", "generic"])
lib.register_tool("sql_query", my_sql_tool, capabilities=["database", "rag"])
lib.register_mcp_client("jira", jira_mcp_client)
```

Then pass `lib` into the orchestrator.

---

## How planning works

- If an LLM is provided to `AgentOrchestrator`, it is prompted with the tool catalog (including capability tags) and asked to build steps with:
  - `tool_names` and/or
  - `required_capabilities`
- If no LLM is provided (or parsing fails), fallback planning builds steps from available capabilities in the tool library and appends a final synthesis step.

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
