from __future__ import annotations

"""Enhanced Milvus vector search helpers for the legal agent with hybrid and grouping search."""

from typing import List, Dict, Any, Optional, Union
import os
import logging
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker

from utils.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class MilvusClientWrapper:
    """Enhanced wrapper around Milvus client for vector search operations."""
    
    def __init__(self, uri: str, token: str):
        """Initialize the Milvus client with URI and token."""
        self.client = MilvusClient(
            uri=uri,
            token=token
        )
    
    def search_similar(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection with enhanced parameters."""
        try:
            # Load the collection before searching
            self.client.load_collection(collection_name)
            
            # Default search parameters
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {}}
            
            # Default output fields to get actual content
            if output_fields is None:
                output_fields = ["text", "content", "metadata", "article", "section"]
            
            logger.info(f"Searching collection {collection_name} with output_fields: {output_fields}")
            
            results = self.client.search(
                collection_name=collection_name,
                anns_field="embedding",
                data=[query_embedding],
                limit=top_k,
                search_params=search_params,
                output_fields=output_fields,
                filter=filter_expr if filter_expr else ""
            )
            
            # Process and return results in a structured format
            processed_results = []
            for hits in results:
                for hit in hits:
                    processed_results.append({
                        "id": hit.get('id'),
                        "distance": hit.get('distance'),
                        "entity": hit.get('entity', {}),
                        "collection": collection_name
                    })
            
            logger.info(f"Found {len(processed_results)} results from {collection_name}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {str(e)}")
            return []
    
    def hybrid_search(
        self,
        collection_name: str,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors."""
        try:
            # Load the collection before searching
            self.client.load_collection(collection_name)
            
            # Default output fields
            if output_fields is None:
                output_fields = ["text", "content", "metadata", "article", "section"]
            
            # Create search requests
            search_requests = []
            
            # Dense vector search request
            dense_request = AnnSearchRequest(
                data=[dense_vector],
                anns_field="embedding",  # Assuming dense vectors are stored in 'embedding' field
                param={"metric_type": "L2", "params": {}},
                limit=top_k
            )
            search_requests.append(dense_request)
            
            # Sparse vector search request (if available)
            if sparse_vector:
                sparse_request = AnnSearchRequest(
                    data=[sparse_vector],
                    anns_field="sparse_embedding",  # Assuming sparse vectors are stored separately
                    param={"metric_type": "IP", "params": {}},
                    limit=top_k
                )
                search_requests.append(sparse_request)
            
            # Choose reranking strategy
            if rerank_strategy == "weighted" and weights:
                ranker = WeightedRanker(*weights)
            else:
                ranker = RRFRanker(k=60)  # Default RRF with k=60
            
            # Perform hybrid search
            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=search_requests,
                ranker=ranker,
                limit=top_k,
                output_fields=output_fields
            )
            
            # Process results
            processed_results = []
            for hits in results:
                for hit in hits:
                    processed_results.append({
                        "id": hit.get('id'),
                        "distance": hit.get('distance'),
                        "entity": hit.get('entity', {}),
                        "collection": collection_name,
                        "search_type": "hybrid"
                    })
            
            logger.info(f"Hybrid search found {len(processed_results)} results from {collection_name}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search on {collection_name}: {str(e)}")
            # Fallback to regular search
            return self.search_similar(collection_name, dense_vector, top_k, output_fields)
    
    def grouping_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        group_by_field: str,
        top_k: int = 5,
        group_size: int = 2,
        strict_group_size: bool = True,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform grouping search to ensure diversity in results."""
        try:
            # Load the collection before searching
            self.client.load_collection(collection_name)
            
            # Default search parameters
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {}}
            
            # Default output fields
            if output_fields is None:
                output_fields = ["text", "content", "metadata", "article", "section", group_by_field]
            
            logger.info(f"Performing grouping search on {collection_name} grouped by {group_by_field}")
            
            results = self.client.search(
                collection_name=collection_name,
                anns_field="embedding",
                data=[query_embedding],
                limit=top_k,
                search_params=search_params,
                output_fields=output_fields,
                group_by_field=group_by_field,
                group_size=group_size,
                strict_group_size=strict_group_size
            )
            
            # Process results
            processed_results = []
            for hits in results:
                for hit in hits:
                    processed_results.append({
                        "id": hit.get('id'),
                        "distance": hit.get('distance'),
                        "entity": hit.get('entity', {}),
                        "collection": collection_name,
                        "search_type": "grouping",
                        "grouped_by": group_by_field
                    })
            
            logger.info(f"Grouping search found {len(processed_results)} results from {collection_name}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to perform grouping search on {collection_name}: {str(e)}")
            # Fallback to regular search
            return self.search_similar(collection_name, query_embedding, top_k, output_fields)


class MilvusVectorDB:
    """Enhanced wrapper around one Milvus account and a set of collections with hybrid and grouping search."""

    def __init__(self, uri: str, token: str, collection_names: List[str]):
        load_dotenv()
        self.client = MilvusClientWrapper(uri=uri, token=token)
        self.collection_names = collection_names
        self.embedder = EmbeddingGenerator()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Basic search across all configured collections."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        for collection_name in self.collection_names:
            try:
                results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=top_k
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                continue
        
        # Sort by distance and return top results
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:top_k]

    def enhanced_search(
        self, 
        query: str, 
        search_type: str = "both",
        top_k: int = 5,
        group_by_field: Optional[str] = None,
        group_size: int = 2,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhanced search with multiple search strategies.
        
        Args:
            query: The search query
            search_type: "basic", "hybrid", "grouping", or "both" (hybrid + grouping)
            top_k: Number of results to return
            group_by_field: Field to group by for grouping search
            group_size: Number of results per group
            rerank_strategy: "rrf" or "weighted"
            weights: Weights for weighted reranking
        
        Returns:
            Dictionary with search results from different strategies
        """
        logger.info(f"Performing enhanced search with type: {search_type}")
        
        # Generate embeddings
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        results = {
            "basic_search": [],
            "hybrid_search": [],
            "grouping_search": [],
            "combined_results": []
        }
        
        all_results = []
        
        for collection_name in self.collection_names:
            try:
                # Basic search (always performed)
                basic_results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=top_k
                )
                results["basic_search"].extend(basic_results)
                all_results.extend(basic_results)
                
                # Hybrid search
                if search_type in ["hybrid", "both"]:
                    hybrid_results = self.client.hybrid_search(
                        collection_name=collection_name,
                        dense_vector=embedding,
                        top_k=top_k,
                        rerank_strategy=rerank_strategy,
                        weights=weights
                    )
                    results["hybrid_search"].extend(hybrid_results)
                    all_results.extend(hybrid_results)
                
                # Grouping search
                if search_type in ["grouping", "both"] and group_by_field:
                    grouping_results = self.client.grouping_search(
                        collection_name=collection_name,
                        query_embedding=embedding,
                        group_by_field=group_by_field,
                        top_k=top_k,
                        group_size=group_size
                    )
                    results["grouping_search"].extend(grouping_results)
                    all_results.extend(grouping_results)
                    
            except Exception as e:
                logger.error(f"Error in enhanced search for collection {collection_name}: {e}")
                continue
        
        # Remove duplicates and sort combined results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get('id')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by distance
        unique_results.sort(key=lambda x: x.get('distance', float('inf')))
        results["combined_results"] = unique_results[:top_k]
        
        # Sort individual result lists
        for key in ["basic_search", "hybrid_search", "grouping_search"]:
            results[key].sort(key=lambda x: x.get('distance', float('inf')))
            results[key] = results[key][:top_k]
        
        logger.info(f"Enhanced search completed. Found {len(results['combined_results'])} unique results")
        return results

    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search across all collections."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        for collection_name in self.collection_names:
            try:
                results = self.client.hybrid_search(
                    collection_name=collection_name,
                    dense_vector=embedding,
                    top_k=top_k,
                    rerank_strategy=rerank_strategy,
                    weights=weights
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in hybrid search for collection {collection_name}: {e}")
                continue
        
        # Sort by distance and return top results
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:top_k]

    def grouping_search(
        self, 
        query: str, 
        group_by_field: str,
        top_k: int = 5,
        group_size: int = 2,
        strict_group_size: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform grouping search across all collections."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        for collection_name in self.collection_names:
            try:
                results = self.client.grouping_search(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    group_by_field=group_by_field,
                    top_k=top_k,
                    group_size=group_size,
                    strict_group_size=strict_group_size
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in grouping search for collection {collection_name}: {e}")
                continue
        
        # Sort by distance and return top results
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:top_k]

    def combined_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform both hybrid and basic search, combine results, and return top results.
        This method combines the strengths of both search approaches for better accuracy.
        """
        logger.info(f"🔍 Performing combined search (hybrid + basic) for query: '{query[:50]}...'")
        
        # Generate embeddings
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        
        for collection_name in self.collection_names:
            try:
                logger.info(f"📚 Searching collection: {collection_name}")
                
                # Perform basic search
                basic_results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=top_k
                )
                
                # Mark basic search results
                for result in basic_results:
                    result['search_type'] = 'basic'
                    result['collection'] = collection_name
                
                all_results.extend(basic_results)
                logger.info(f"✅ Basic search found {len(basic_results)} results from {collection_name}")
                
                # Perform hybrid search
                hybrid_results = self.client.hybrid_search(
                    collection_name=collection_name,
                    dense_vector=embedding,
                    top_k=top_k,
                    rerank_strategy="rrf"
                )
                
                # Mark hybrid search results
                for result in hybrid_results:
                    result['search_type'] = 'hybrid'
                    result['collection'] = collection_name
                
                all_results.extend(hybrid_results)
                logger.info(f"✅ Hybrid search found {len(hybrid_results)} results from {collection_name}")
                
            except Exception as e:
                logger.error(f"❌ Error searching collection {collection_name}: {e}")
                continue
        
        # Remove duplicates based on content similarity and ID
        unique_results = []
        seen_ids = set()
        seen_content = set()
        
        for result in all_results:
            result_id = result.get('id')
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            
            # Create a content hash for duplicate detection
            content_hash = hash(content[:200]) if content else hash(str(result_id))
            
            if result_id not in seen_ids and content_hash not in seen_content:
                seen_ids.add(result_id)
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by distance (lower is better)
        unique_results.sort(key=lambda x: x.get('distance', float('inf')))
        
        # Return top results
        final_results = unique_results[:top_k]
        
        logger.info(f"🎯 Combined search completed: {len(all_results)} total results -> {len(unique_results)} unique -> {len(final_results)} final results")
        
        return final_results 