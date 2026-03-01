"""Tool library and MCP client registry used by sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ToolCallable


@dataclass
class ToolDefinition:
    """Tool metadata used for capability-aware agent spawning."""

    name: str
    tool: ToolCallable
    capabilities: set[str] = field(default_factory=set)
    description: str = ""


@dataclass
class AgentToolLibrary:
    """In-memory registry for tools and MCP clients."""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    mcp_clients: dict[str, object] = field(default_factory=dict)

    def register_tool(
        self,
        name: str,
        tool: ToolCallable,
        capabilities: list[str] | set[str] | None = None,
        description: str = "",
    ) -> None:
        self.tools[name] = ToolDefinition(
            name=name,
            tool=tool,
            capabilities=set(capabilities or ["generic"]),
            description=description,
        )

    def register_mcp_client(self, name: str, client: object) -> None:
        self.mcp_clients[name] = client

    def available_capabilities(self) -> set[str]:
        capabilities: set[str] = set()
        for tool in self.tools.values():
            capabilities.update(tool.capabilities)
        return capabilities

    def list_tool_catalog(self) -> list[dict[str, str | list[str]]]:
        return [
            {
                "name": tool.name,
                "capabilities": sorted(tool.capabilities),
                "description": tool.description,
            }
            for tool in self.tools.values()
        ]

    def resolve_tools(self, names: list[str]) -> dict[str, ToolCallable]:
        missing = [name for name in names if name not in self.tools]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"Requested tools are not registered: {missing_text}")
        return {name: self.tools[name].tool for name in names}

    def resolve_tools_by_capabilities(self, capabilities: list[str]) -> dict[str, ToolCallable]:
        required = set(capabilities)
        if not required:
            return {}

        matched = {
            name: definition.tool
            for name, definition in self.tools.items()
            if required.issubset(definition.capabilities) or required.intersection(definition.capabilities)
        }
        if not matched:
            capabilities_text = ", ".join(sorted(required))
            raise KeyError(f"No tools found for required capabilities: {capabilities_text}")
        return matched

    def infer_agent_profile(self, tool_names: list[str]) -> str:
        capabilities: set[str] = set()
        for name in tool_names:
            definition = self.tools.get(name)
            if definition:
                capabilities.update(definition.capabilities)

        if not capabilities:
            return "generic"
        return "+".join(sorted(capabilities))


def default_tool_library() -> AgentToolLibrary:
    """Bootstrap with capability-tagged tools."""

    lib = AgentToolLibrary()
    lib.register_tool(
        "echo",
        lambda text: text,
        capabilities=["generic"],
        description="Echoes incoming text.",
    )
    lib.register_tool(
        "word_count",
        lambda text: str(len(text.split())),
        capabilities=["generic", "rag"],
        description="Counts words in text, useful for summarization metrics.",
    )
    lib.register_tool(
        "image_stub",
        lambda text: f"image-processed:{text[:40]}",
        capabilities=["image", "generic"],
        description="Placeholder image-processing adapter.",
    )
    lib.register_tool(
        "db_stub",
        lambda text: f"db-query-simulated:{text[:40]}",
        capabilities=["database", "rag"],
        description="Placeholder database retrieval adapter.",
    )
    return lib
