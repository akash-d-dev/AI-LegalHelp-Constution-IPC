
# Vector DB utilities using Milvus
from __future__ import annotations

from typing import List

from backend.admin.scripts.milvus_client import MilvusDBClient
from backend.admin.scripts.embedding_generator import EmbeddingGenerator


class MilvusVectorDB:
    """Wrapper around MilvusDBClient for search operations."""

    def __init__(self, collection_prefix: str = "constitution_of_india"):
        self.client = MilvusDBClient()
        self.collection_prefix = collection_prefix
        self.embedder = EmbeddingGenerator()

    def _collection_names(self) -> List[str]:
        """Return all collection names with the given prefix."""
        client = self.client._get_client()
        all_collections = client.list_collections()
        return [c for c in all_collections if c.startswith(self.collection_prefix)]

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search across all collections for the query."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        results = []
        for name in self._collection_names():
            try:
                hits = self.client.search_similar(name, embedding, top_k=top_k)
                for hit_list in hits:
                    for hit in hit_list:
                        hit["collection"] = name
                results.extend(hits)
            except Exception:
                continue
        return results

