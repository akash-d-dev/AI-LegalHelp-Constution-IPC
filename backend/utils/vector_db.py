from __future__ import annotations

"""Milvus vector search helpers for the legal agent."""

from typing import List
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType

from utils.embedding_generator import EmbeddingGenerator


class MilvusClientWrapper:
    """Wrapper around Milvus client for vector search operations."""
    
    def __init__(self, uri: str, token: str):
        """Initialize the Milvus client with URI and token."""
        self.client = MilvusClient(
            uri=uri,
            token=token
        )
    
    def search_similar(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[List[dict]]:
        """Search for similar vectors in the collection."""
        try:
            # Load the collection before searching
            self.client.load_collection(collection_name)
            
            results = self.client.search(
                collection_name=collection_name,
                anns_field="embedding",
                data=[query_embedding],
                limit=top_k,
                search_params={"metric_type": "L2"}
            )
            
            # Process and return results in a structured format
            processed_results = []
            for hits in results:
                hit_list = []
                for hit in hits:
                    hit_list.append({
                        "id": hit.get('id'),
                        "distance": hit.get('distance'),
                        "entity": hit.get('entity', {})
                    })
                processed_results.append(hit_list)
            
            return processed_results
        except Exception as e:
            print(f"Failed to search: {str(e)}")
            raise


class MilvusVectorDB:
    """Wrapper around one Milvus account and a set of collections."""

    def __init__(self, uri: str, token: str, collection_names: List[str]):
        load_dotenv()
        self.client = MilvusClientWrapper(uri=uri, token=token)
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