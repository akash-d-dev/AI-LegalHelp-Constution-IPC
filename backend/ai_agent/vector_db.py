from __future__ import annotations

"""Milvus vector search helpers for the legal agent."""

from typing import List
import os
from dotenv import load_dotenv

from backend.admin.scripts.milvus_client import MilvusDBClient
from backend.admin.scripts.embedding_generator import EmbeddingGenerator


class MilvusVectorDB:
    """Wrapper around one Milvus account and a set of collections."""

    def __init__(self, uri: str, token: str, collection_names: List[str]):
        load_dotenv()
        self.client = MilvusDBClient(uri=uri, token=token)
        self.collection_names = collection_names
        self.embedder = EmbeddingGenerator()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search the configured collections for the query."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        results: List[dict] = []
        for name in self.collection_names:
            try:
                hits = self.client.search_similar(name, embedding, top_k=top_k)
                for hit_list in hits:
                    for hit in hit_list:
                        hit["collection"] = name
                results.extend(hits)
            except Exception:
                continue
        return results
