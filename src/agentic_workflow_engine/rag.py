"""Minimal configurable local RAG datastore and retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .llm_config import RAGSettings


@dataclass
class RagDocument:
    id: str
    text: str
    metadata: dict[str, str]


class LocalRagDatabase:
    """Simple JSON-backed vector store for semantic search and upload."""

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings
        self.path = Path(settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _embed(self, text: str, dim: int = 16) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for idx in range(dim):
            byte_val = seed[idx]
            vector.append((byte_val / 255.0) * 2 - 1)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def upload_documents(self, docs: list[RagDocument]) -> int:
        rows = self._load()
        for doc in docs:
            rows.append(
                {
                    "collection": self.settings.collection,
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "embedding": self._embed(doc.text),
                }
            )
        self._save(rows)
        return len(docs)

    def semantic_search(self, query: str, threshold: float | None = None, top_k: int | None = None) -> list[dict]:
        active_threshold = threshold if threshold is not None else self.settings.semantic_search_threshold
        active_top_k = top_k if top_k is not None else self.settings.top_k

        rows = self._load()
        query_vector = self._embed(query)
        matches = []
        for row in rows:
            if row.get("collection") != self.settings.collection:
                continue
            score = self._cosine_similarity(query_vector, row["embedding"])
            if score >= active_threshold:
                matches.append({"id": row["id"], "text": row["text"], "metadata": row["metadata"], "score": score})

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:active_top_k]



def connect_rag_database(settings: RAGSettings) -> LocalRagDatabase:
    if settings.provider != "local":
        raise ValueError(f"Unsupported RAG provider '{settings.provider}'. Only 'local' is currently implemented.")
    return LocalRagDatabase(settings)
