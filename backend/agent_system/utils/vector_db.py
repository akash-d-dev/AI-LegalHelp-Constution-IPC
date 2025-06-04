from __future__ import annotations

"""Enhanced Milvus vector search helpers for the legal agent with hybrid and grouping search."""

from typing import List, Dict, Any, Optional, Union
import os
import logging
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker
from datetime import datetime

from agent_system.utils.embedding_generator import EmbeddingGenerator

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
            
            logger.info(f"🔄 Attempting hybrid search on {collection_name}")
            
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
            else:
                logger.info(f"⚠️ No sparse vector provided for hybrid search on {collection_name}")
            
            # Choose reranking strategy
            if rerank_strategy == "weighted" and weights:
                ranker = WeightedRanker(*weights)
            else:
                ranker = RRFRanker(k=60)  # Default RRF with k=60
            
            # Perform hybrid search
            logger.info(f"🔍 Executing hybrid search with {len(search_requests)} search requests")
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
            
            logger.info(f"✅ True hybrid search found {len(processed_results)} results from {collection_name}")
            return processed_results
            
        except Exception as e:
            logger.warning(f"⚠️ Hybrid search failed on {collection_name}: {str(e)}")
            logger.info(f"🔄 Falling back to basic search for {collection_name}")
            # Fallback to regular search but mark results as hybrid_fallback
            fallback_results = self.search_similar(collection_name, dense_vector, top_k, output_fields)
            
            # Mark fallback results to distinguish from pure basic search
            for result in fallback_results:
                result['search_type'] = 'hybrid_fallback'
                result['collection'] = collection_name
                result['fallback_reason'] = str(e)
            
            logger.info(f"✅ Hybrid fallback search found {len(fallback_results)} results from {collection_name}")
            return fallback_results
    
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
        
        Note: If sparse embeddings are not available, this will fall back to enhanced basic search only.
        """
        logger.info(f"🔍 Performing combined search (hybrid + basic) for query: '{query[:50]}...'")
        
        # Generate embeddings
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        basic_count = 0
        hybrid_count = 0
        
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
                basic_count += len(basic_results)
                logger.info(f"✅ Basic search found {len(basic_results)} results from {collection_name}")
                
                # Only attempt hybrid search if we expect it to provide different results
                # For now, skip hybrid search since collections don't have sparse embeddings
                logger.info(f"⚠️ Skipping hybrid search for {collection_name} (no sparse embeddings configured)")
                
                # You could enable this once sparse embeddings are set up:
                # hybrid_results = self.client.hybrid_search(...)
                
            except Exception as e:
                logger.error(f"❌ Error searching collection {collection_name}: {e}")
                continue
        
        logger.info(f"📊 Search summary: {basic_count} basic results (hybrid search disabled)")
        
        # Since we're only using basic search, no deduplication needed
        unique_results = all_results
        
        # Sort by distance (lower is better)
        unique_results.sort(key=lambda x: x.get('distance', float('inf')))
        
        # Return top results
        final_results = unique_results[:top_k]
        
        logger.info(f"🎯 Combined search completed: {len(all_results)} total -> {len(final_results)} final")
        logger.info(f"🏆 Final result types: {{'basic': {len(final_results)}}}")
        
        # Save results to log file
        self._save_search_results_to_log(query, final_results, len(all_results), len(unique_results))
        
        return final_results

    def combined_search_enhanced(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Enhanced combined search that uses multiple search strategies to get diverse results
        even without sparse embeddings. This method uses different parameters and approaches
        to maximize result diversity and quality.
        """
        logger.info(f"🔍 Performing enhanced combined search for query: '{query[:50]}...'")
        
        # Generate embeddings
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        search_strategies = []
        
        for collection_name in self.collection_names:
            try:
                logger.info(f"📚 Searching collection: {collection_name}")
                
                # Strategy 1: Standard L2 distance search
                l2_results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=top_k,
                    search_params={"metric_type": "L2", "params": {}}
                )
                for result in l2_results:
                    result['search_type'] = 'l2_standard'
                    result['collection'] = collection_name
                all_results.extend(l2_results)
                search_strategies.append(f"L2 Standard: {len(l2_results)}")
                
                # Strategy 2: Cosine similarity search (if supported)
                try:
                    cosine_results = self.client.search_similar(
                        collection_name=collection_name,
                        query_embedding=embedding,
                        top_k=top_k,
                        search_params={"metric_type": "COSINE", "params": {}}
                    )
                    for result in cosine_results:
                        result['search_type'] = 'cosine'
                        result['collection'] = collection_name
                    all_results.extend(cosine_results)
                    search_strategies.append(f"Cosine: {len(cosine_results)}")
                except Exception as e:
                    logger.debug(f"Cosine search not available for {collection_name}: {e}")
                
                # Strategy 3: Higher top_k with different filtering
                expanded_results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=min(top_k * 2, 10),  # Get more results for diversity
                    search_params={"metric_type": "L2", "params": {}}
                )
                for result in expanded_results:
                    result['search_type'] = 'expanded'
                    result['collection'] = collection_name
                all_results.extend(expanded_results)
                search_strategies.append(f"Expanded: {len(expanded_results)}")
                
                logger.info(f"✅ Multi-strategy search completed for {collection_name}")
                
            except Exception as e:
                logger.error(f"❌ Error searching collection {collection_name}: {e}")
                continue
        
        logger.info(f"📊 Search strategies: {', '.join(search_strategies)}")
        
        # Smart deduplication that preserves diversity
        unique_results = []
        seen_ids = set()
        seen_content_hashes = set()
        strategy_counts = {}
        
        # Sort by distance first to prioritize best matches
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        
        for result in all_results:
            result_id = result.get('id')
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            search_type = result.get('search_type', 'unknown')
            
            # Create content hash for similarity detection
            content_hash = hash(content[:300]) if content else hash(str(result_id))
            
            # Keep result if it's truly unique
            if result_id not in seen_ids and content_hash not in seen_content_hashes:
                seen_ids.add(result_id)
                seen_content_hashes.add(content_hash)
                unique_results.append(result)
                strategy_counts[search_type] = strategy_counts.get(search_type, 0) + 1
                
                logger.debug(f"✅ Kept {search_type} result (distance: {result.get('distance', 'N/A'):.4f})")
                
                # Stop when we have enough unique results
                if len(unique_results) >= top_k:
                    break
        
        logger.info(f"🧹 Deduplication: {len(all_results)} total -> {len(unique_results)} unique")
        logger.info(f"🏆 Final strategy distribution: {strategy_counts}")
        
        # Return top results
        final_results = unique_results[:top_k]
        
        logger.info(f"🎯 Enhanced combined search completed: {len(final_results)} final results")
        
        # Save results to log file
        self._save_search_results_to_log(query, final_results, len(all_results), len(unique_results))
        
        return final_results

    def _save_search_results_to_log(self, query: str, results: List[Dict[str, Any]], total_results: int, unique_results: int):
        """Save search results to a log file in the generated directory."""
        try:
            # Create generated directory if it doesn't exist
            os.makedirs("generated", exist_ok=True)
            
            log_file = os.path.join("generated", "vector_db_results.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"VECTOR DATABASE SEARCH LOG\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Query: '{query}'\n")
                f.write(f"Total Results: {total_results} -> Unique: {unique_results} -> Final: {len(results)}\n")
                f.write(f"{'='*80}\n\n")
                
                for i, result in enumerate(results, 1):
                    entity = result.get('entity', {})
                    content = entity.get('text') or entity.get('content', 'No content available')
                    distance = result.get('distance', 'Unknown')
                    search_type = result.get('search_type', 'Unknown')
                    collection = result.get('collection', 'Unknown')
                    article = entity.get('article', 'Unknown Article')
                    section = entity.get('section', 'Unknown Section')
                    
                    f.write(f"Result {i}:\n")
                    f.write(f"  Distance: {distance:.4f}\n")
                    f.write(f"  Search Type: {search_type}\n")
                    f.write(f"  Collection: {collection}\n")
                    f.write(f"  Article/Section: {article if article != 'Unknown Article' else section}\n")
                    f.write(f"  Content: {content}\n")
                    f.write(f"{'-'*40}\n")
                
                f.write(f"\n{'='*80}\n\n")
                
        except Exception as e:
            logger.error(f"❌ Error saving search results to log: {e}") 