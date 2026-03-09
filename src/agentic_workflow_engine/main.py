"""CLI entrypoint for running the workflow engine."""

from __future__ import annotations

import argparse
import json

from .llm_config import load_app_config
from .orchestrator import AgentOrchestrator
from .tool_lib import default_tool_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic workflow engine")
    parser.add_argument("objective", help="Objective for orchestrator to execute")
    parser.add_argument("--config", default="llm.config", help="Path to llm.config")
    args = parser.parse_args()

    app_config = load_app_config(args.config)
    orchestrator = AgentOrchestrator(
        tool_library=default_tool_library(config=app_config),
        config=app_config,
        config_path=args.config,
    )
    result = orchestrator.execute_objective_sync(args.objective)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
