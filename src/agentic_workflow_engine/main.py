"""CLI entrypoint for running the workflow engine."""

from __future__ import annotations

import argparse
import json

from .orchestrator import AgentOrchestrator
from .tool_lib import default_tool_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic workflow engine")
    parser.add_argument("objective", help="Objective for orchestrator to execute")
    args = parser.parse_args()

    orchestrator = AgentOrchestrator(tool_library=default_tool_library())
    result = orchestrator.execute_objective_sync(args.objective)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
