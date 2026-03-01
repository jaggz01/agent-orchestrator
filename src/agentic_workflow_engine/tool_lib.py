"""Tool library and MCP client registry used by sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ToolCallable


@dataclass
class AgentToolLibrary:
    """In-memory registry for tools and MCP clients."""

    tools: dict[str, ToolCallable] = field(default_factory=dict)
    mcp_clients: dict[str, object] = field(default_factory=dict)

    def register_tool(self, name: str, tool: ToolCallable) -> None:
        self.tools[name] = tool

    def register_mcp_client(self, name: str, client: object) -> None:
        self.mcp_clients[name] = client

    def resolve_tools(self, names: list[str]) -> dict[str, ToolCallable]:
        missing = [name for name in names if name not in self.tools]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"Requested tools are not registered: {missing_text}")
        return {name: self.tools[name] for name in names}


def default_tool_library() -> AgentToolLibrary:
    """Bootstrap with a few example tools."""

    lib = AgentToolLibrary()
    lib.register_tool("echo", lambda text: text)
    lib.register_tool("word_count", lambda text: str(len(text.split())))
    return lib
