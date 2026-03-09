"""Tool library and MCP client registry used by sub-agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm_config import AppConfig
from .models import ToolCallable
from .rag import LocalRagDatabase, RagDocument, connect_rag_database


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


def _build_rag_upload_tool(rag_db: LocalRagDatabase) -> ToolCallable:
    def rag_upload(payload: str) -> str:
        """Upload one or many docs. Payload: JSON {'documents':[{'id','text','metadata'}]}"""

        parsed = json.loads(payload)
        documents = parsed.get("documents", [])
        docs = [
            RagDocument(
                id=str(doc["id"]),
                text=str(doc["text"]),
                metadata={str(k): str(v) for k, v in doc.get("metadata", {}).items()},
            )
            for doc in documents
        ]
        uploaded = rag_db.upload_documents(docs)
        return f"Uploaded {uploaded} documents"

    return rag_upload


def _build_semantic_search_tool(rag_db: LocalRagDatabase, default_threshold: float, default_top_k: int) -> ToolCallable:
    def semantic_search(payload: str) -> str:
        """Run semantic search. Payload: plain text query or JSON with query/threshold/top_k."""

        query = payload
        threshold = default_threshold
        top_k = default_top_k
        if payload.strip().startswith("{"):
            parsed = json.loads(payload)
            query = str(parsed["query"])
            threshold = float(parsed.get("threshold", default_threshold))
            top_k = int(parsed.get("top_k", default_top_k))

        results = rag_db.semantic_search(query=query, threshold=threshold, top_k=top_k)
        return json.dumps(results)

    return semantic_search


def default_tool_library(config: AppConfig | None = None) -> AgentToolLibrary:
    """Bootstrap with default tools and optional RAG-backed capabilities."""

    lib = AgentToolLibrary()
    lib.register_tool("echo", lambda text: text)
    lib.register_tool("word_count", lambda text: str(len(text.split())))

    if config and config.rag.enabled:
        rag_db = connect_rag_database(config.rag)
        lib.register_mcp_client("rag_db", rag_db)
        lib.register_tool("rag_upload_documents", _build_rag_upload_tool(rag_db))
        lib.register_tool(
            "rag_semantic_search",
            _build_semantic_search_tool(
                rag_db,
                default_threshold=config.rag.semantic_search_threshold,
                default_top_k=config.rag.top_k,
            ),
        )

    return lib
