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
  - Generates a plan via a configured LLM.
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

- `AgentOrchestrator` always uses an LLM for planning.
- If you pass `llm=...` to `AgentOrchestrator`, that model is used.
- If you do not pass `llm`, the orchestrator **requires** `llm.config` and builds a `BaseChatModel` strictly from the values in that file (`provider`, `model`, optional `base_url`, `api_key`, `temperature`, `max_tokens`, `timeout`). If the file or required keys are missing, startup fails with an explicit error message.

Example `llm.config`:

```ini
[llm]
provider = openai
model = gpt-4o-mini
base_url =
api_key =
temperature = 0.2
max_tokens = 1200
timeout = 60

[rag]
enabled = true
provider = local
database_path = data/rag_db.json
collection = default
semantic_search_threshold = 0.35
top_k = 5
```

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


## RAG capabilities and tools

- `default_tool_library(config=...)` now creates a configured RAG client when `[rag].enabled=true`.
- Core capability `rag_connection` is resolved by the spawner (non-tool capability) before tool execution.
- Tools provided:
  - `rag_upload_documents` (uploads docs to vector DB).
  - `rag_semantic_search` (semantic retrieval with configurable threshold/top-k).
