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

    def enhanced_cross_domain_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced cross-domain search specifically designed for legal queries that may span
        multiple legal domains (e.g., constitutional + criminal law). This method uses
        query decomposition, domain-specific search variants, and intelligent result fusion.
        
        This works with existing schema and doesn't require Milvus 2.5 upgrades.
        """
        logger.info(f"🔍 Performing enhanced cross-domain search for: '{query[:50]}...'")
        
        # Step 1: Generate legal domain-specific query variants
        query_variants = self._generate_legal_query_variants(query)
        logger.info(f"📝 Generated {len(query_variants)} query variants for enhanced search")
        
        # Step 2: Execute multiple search strategies
        all_results = []
        strategy_counts = {}
        
        # Generate embeddings for all variants
        variant_embeddings = {}
        for variant in query_variants:
            _, embedding = self.embedder.generate_embeddings([variant])
            variant_embeddings[variant] = embedding[0]
        
        # Strategy 1: Original query with multiple distance metrics
        logger.info("🎯 Strategy 1: Multi-metric search with original query")
        original_results = self._multi_metric_search(query, top_k)
        for result in original_results:
            result['search_strategy'] = 'multi_metric_original'
            result['query_variant'] = query
        all_results.extend(original_results)
        strategy_counts['multi_metric_original'] = len(original_results)
        
        # Strategy 2: Domain-specific variant searches
        logger.info("🎯 Strategy 2: Domain-specific variant searches")
        for i, variant in enumerate(query_variants[:3]):  # Limit to avoid too many searches
            logger.info(f"  Variant {i+1}: '{variant[:50]}...'")
            variant_results = self._targeted_variant_search(variant, variant_embeddings[variant], top_k)
            for result in variant_results:
                result['search_strategy'] = f'domain_variant_{i+1}'
                result['query_variant'] = variant
            all_results.extend(variant_results)
            strategy_counts[f'domain_variant_{i+1}'] = len(variant_results)
        
        # Strategy 3: Expanded search with relaxed parameters
        logger.info("🎯 Strategy 3: Expanded parameter search")
        expanded_results = self._expanded_parameter_search(query, top_k * 2)
        for result in expanded_results:
            result['search_strategy'] = 'expanded_parameters'
            result['query_variant'] = query
        all_results.extend(expanded_results)
        strategy_counts['expanded_parameters'] = len(expanded_results)
        
        logger.info(f"📊 Search strategies executed: {strategy_counts}")
        
        # Step 3: Intelligent result fusion with domain scoring
        logger.info("🔀 Step 3: Applying intelligent result fusion...")
        fused_results = self._intelligent_result_fusion(all_results, query, top_k)
        
        # Step 4: Legal domain analysis
        domain_analysis = self._analyze_legal_domains(fused_results, query)
        
        # Add domain analysis to results
        for result in fused_results:
            result['domain_analysis'] = domain_analysis
        
        logger.info(f"✅ Enhanced cross-domain search completed: {len(fused_results)} final results")
        logger.info(f"🏆 Domain coverage: {domain_analysis.get('domain_coverage', {})}")
        
        # Save to log with enhanced details
        self._save_enhanced_search_log(query, fused_results, all_results, strategy_counts, domain_analysis)
        
        return fused_results
    
    def _generate_legal_query_variants(self, query: str) -> List[str]:
        """Generate legal domain-specific query variants for enhanced search coverage."""
        query_lower = query.lower()
        variants = [query]  # Always include original
        
        # Constitutional law indicators
        constitutional_terms = [
            'constitutional', 'article', 'fundamental rights', 'freedom', 'speech', 
            'expression', 'liberty', 'equality', 'directive principles', 'amendment'
        ]
        
        # Criminal law indicators  
        criminal_terms = [
            'criminal', 'punishment', 'imprisonment', 'section', 'offense', 'defamation',
            'hate speech', 'liability', 'conviction', 'ipc', 'penal', 'crime'
        ]
        
        # Check domain relevance
        const_relevance = sum(1 for term in constitutional_terms if term in query_lower)
        criminal_relevance = sum(1 for term in criminal_terms if term in query_lower)
        
        # Generate constitutional variants
        if const_relevance > 0:
            variants.extend([
                f"fundamental rights {query}",
                f"constitutional protection {query}",
                f"Article 19 {query}" if 'speech' in query_lower or 'expression' in query_lower else f"constitutional provision {query}"
            ])
        
        # Generate criminal law variants  
        if criminal_relevance > 0:
            variants.extend([
                f"criminal law {query}",
                f"IPC section {query}",
                f"punishment {query}"
            ])
        
        # Generate cross-domain variants for complex queries
        if const_relevance > 0 and criminal_relevance > 0:
            variants.extend([
                f"legal framework {query}",
                f"constitutional criminal law {query}",
                f"rights and restrictions {query}"
            ])
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant not in seen:
                unique_variants.append(variant)
                seen.add(variant)
        
        return unique_variants[:6]  # Limit to avoid excessive searches
    
    def _multi_metric_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using multiple distance metrics for diverse results."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        metrics = [("L2", "L2"), ("COSINE", "COSINE"), ("IP", "IP")]
        
        for metric_name, metric_type in metrics:
            for collection_name in self.collection_names:
                try:
                    results = self.client.search_similar(
                        collection_name=collection_name,
                        query_embedding=embedding,
                        top_k=top_k,
                        search_params={"metric_type": metric_type, "params": {}}
                    )
                    for result in results:
                        result['search_type'] = f'multi_metric_{metric_name.lower()}'
                        result['collection'] = collection_name
                    all_results.extend(results)
                except Exception as e:
                    logger.debug(f"Metric {metric_name} not supported for {collection_name}: {e}")
                    continue
        
        return all_results
    
    def _targeted_variant_search(self, variant: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Perform targeted search with a specific query variant."""
        all_results = []
        
        for collection_name in self.collection_names:
            try:
                results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=top_k,
                    search_params={"metric_type": "L2", "params": {}}
                )
                for result in results:
                    result['search_type'] = 'targeted_variant'
                    result['collection'] = collection_name
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in targeted search for {collection_name}: {e}")
                continue
        
        return all_results
    
    def _expanded_parameter_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search with expanded parameters for broader coverage."""
        _, embedding = self.embedder.generate_embeddings([query])
        embedding = embedding[0]
        
        all_results = []
        
        for collection_name in self.collection_names:
            try:
                # Relaxed search with higher top_k
                results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=embedding,
                    top_k=min(top_k, 15),  # Expanded but reasonable
                    search_params={"metric_type": "L2", "params": {}}
                )
                for result in results:
                    result['search_type'] = 'expanded_parameters'
                    result['collection'] = collection_name
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in expanded search for {collection_name}: {e}")
                continue
        
        return all_results
    
    def _intelligent_result_fusion(self, all_results: List[Dict[str, Any]], original_query: str, top_k: int) -> List[Dict[str, Any]]:
        """Apply intelligent fusion with domain relevance scoring and deduplication."""
        logger.info(f"🧠 Fusing {len(all_results)} results with domain intelligence...")
        
        # Step 1: Calculate domain relevance for each result
        for result in all_results:
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            
            # Domain relevance scoring
            result['domain_score'] = self._calculate_domain_relevance_score(content, original_query)
            
            # Strategy diversity bonus
            strategy = result.get('search_strategy', 'unknown')
            result['strategy_bonus'] = self._get_strategy_diversity_bonus(strategy)
            
            # Calculate composite score
            distance = result.get('distance', 1.0)
            similarity_score = max(0, 1.0 - distance)  # Convert distance to similarity
            
            composite_score = (
                similarity_score * 0.6 +  # Semantic similarity (60%)
                result['domain_score'] * 0.3 +  # Domain relevance (30%)
                result['strategy_bonus'] * 0.1   # Strategy diversity (10%)
            )
            
            result['composite_score'] = composite_score
        
        # Step 2: Advanced deduplication
        unique_results = self._advanced_deduplication(all_results)
        
        # Step 3: Sort by composite score and return top results
        unique_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        final_results = unique_results[:top_k]
        
        logger.info(f"🎯 Fusion complete: {len(all_results)} -> {len(unique_results)} unique -> {len(final_results)} final")
        return final_results
    
    def _calculate_domain_relevance_score(self, content: str, query: str) -> float:
        """Calculate how relevant content is to the legal domain of the query."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Constitutional law terms
        const_terms = ['constitutional', 'article', 'fundamental', 'rights', 'freedom', 'liberty']
        const_score = sum(1 for term in const_terms if term in content_lower) / len(const_terms)
        
        # Criminal law terms
        criminal_terms = ['criminal', 'punishment', 'section', 'offense', 'liable', 'imprisonment']
        criminal_score = sum(1 for term in criminal_terms if term in content_lower) / len(criminal_terms)
        
        # Query alignment
        query_terms = query_lower.split()
        content_alignment = sum(1 for term in query_terms if term in content_lower) / max(len(query_terms), 1)
        
        # Combined score
        domain_score = max(const_score, criminal_score) * 0.6 + content_alignment * 0.4
        return min(domain_score, 1.0)
    
    def _get_strategy_diversity_bonus(self, strategy: str) -> float:
        """Assign diversity bonus based on search strategy to promote result diversity."""
        strategy_bonuses = {
            'multi_metric_original': 0.8,
            'domain_variant_1': 0.9,
            'domain_variant_2': 0.7,
            'domain_variant_3': 0.6,
            'expanded_parameters': 0.5,
            'targeted_variant': 0.7
        }
        return strategy_bonuses.get(strategy, 0.3)
    
    def _advanced_deduplication(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced deduplication that preserves the best result from each unique content."""
        unique_results = []
        seen_content_hashes = set()
        seen_ids = set()
        
        # Sort by composite score first to prioritize best results
        sorted_results = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        for result in sorted_results:
            result_id = result.get('id')
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            
            # Create content hash for duplicate detection
            content_hash = hash(content[:200]) if content else hash(str(result_id))
            
            # Keep if truly unique
            if result_id not in seen_ids and content_hash not in seen_content_hashes:
                seen_ids.add(result_id)
                seen_content_hashes.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _analyze_legal_domains(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Analyze the legal domains covered in the search results."""
        domain_analysis = {
            'query_classification': {},
            'result_domains': {},
            'domain_coverage': {},
            'cross_domain_relevance': 0.0
        }
        
        # Classify original query
        query_lower = query.lower()
        const_indicators = sum(1 for term in ['constitutional', 'article', 'fundamental', 'rights'] if term in query_lower)
        criminal_indicators = sum(1 for term in ['criminal', 'punishment', 'section', 'ipc'] if term in query_lower)
        
        domain_analysis['query_classification'] = {
            'constitutional_strength': min(const_indicators / 4.0, 1.0),
            'criminal_strength': min(criminal_indicators / 4.0, 1.0),
            'is_cross_domain': const_indicators > 0 and criminal_indicators > 0
        }
        
        # Analyze result domains
        const_results = 0
        criminal_results = 0
        
        for result in results:
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            content_lower = content.lower()
            
            is_constitutional = any(term in content_lower for term in ['constitutional', 'article', 'fundamental'])
            is_criminal = any(term in content_lower for term in ['criminal', 'punishment', 'section'])
            
            if is_constitutional:
                const_results += 1
            if is_criminal:
                criminal_results += 1
        
        total_results = len(results)
        if total_results > 0:
            domain_analysis['domain_coverage'] = {
                'constitutional_percentage': (const_results / total_results) * 100,
                'criminal_percentage': (criminal_results / total_results) * 100,
                'balanced_coverage': abs(const_results - criminal_results) <= 2
            }
            
            # Cross-domain relevance score
            if const_results > 0 and criminal_results > 0:
                domain_analysis['cross_domain_relevance'] = min(const_results, criminal_results) / total_results
        
        return domain_analysis
    
    def _save_enhanced_search_log(self, query: str, results: List[Dict[str, Any]], all_results: List[Dict[str, Any]], 
                                 strategy_counts: Dict[str, int], domain_analysis: Dict[str, Any]):
        """Save enhanced search results with detailed analysis to log file."""
        try:
            os.makedirs("generated", exist_ok=True)
            log_file = os.path.join("generated", "enhanced_search_results.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"ENHANCED CROSS-DOMAIN SEARCH LOG\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Query: '{query}'\n")
                f.write(f"{'='*100}\n\n")
                
                # Search strategy summary
                f.write(f"SEARCH STRATEGY SUMMARY:\n")
                f.write(f"Total Raw Results: {len(all_results)}\n")
                f.write(f"Strategy Breakdown: {strategy_counts}\n")
                f.write(f"Final Unique Results: {len(results)}\n\n")
                
                # Domain analysis
                f.write(f"DOMAIN ANALYSIS:\n")
                f.write(f"Query Classification: {domain_analysis.get('query_classification', {})}\n")
                f.write(f"Domain Coverage: {domain_analysis.get('domain_coverage', {})}\n")
                f.write(f"Cross-Domain Relevance: {domain_analysis.get('cross_domain_relevance', 0):.2f}\n\n")
                
                # Detailed results
                f.write(f"DETAILED RESULTS:\n")
                for i, result in enumerate(results, 1):
                    entity = result.get('entity', {})
                    content = entity.get('text') or entity.get('content', 'No content available')
                    
                    f.write(f"Result {i}:\n")
                    f.write(f"  Composite Score: {result.get('composite_score', 0):.4f}\n")
                    f.write(f"  Original Distance: {result.get('distance', 'N/A'):.4f}\n")
                    f.write(f"  Domain Score: {result.get('domain_score', 0):.4f}\n")
                    f.write(f"  Strategy: {result.get('search_strategy', 'Unknown')}\n")
                    f.write(f"  Collection: {result.get('collection', 'Unknown')}\n")
                    f.write(f"  Query Variant: {result.get('query_variant', 'Unknown')[:50]}...\n")
                    f.write(f"  Content: {content[:300]}...\n")
                    f.write(f"{'-'*50}\n")
                
                f.write(f"\n{'='*100}\n\n")
                
        except Exception as e:
            logger.error(f"❌ Error saving enhanced search log: {e}") 