from __future__ import annotations

"""Enhanced Milvus vector search helpers for the legal agent with concurrent execution and fuzzy deduplication."""

from typing import List, Dict, Any, Optional, Union
import os
import logging
import concurrent.futures
from dataclasses import dataclass
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker
from datetime import datetime
import numpy as np
from simhash import Simhash

from agent_system.utils.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """Structured search result with metadata."""
    id: str
    distance: float
    entity: Dict[str, Any]
    collection: str
    strategy: str


class MilvusClientWrapper:
    """Enhanced wrapper around Milvus client for vector search operations."""
    
    EMBED_FIELD = "embedding"
    OUTPUT_FIELDS = ["text", "content", "metadata", "article", "section"]
    
    def __init__(self, uri: str, token: str):
        """Initialize the Milvus client with URI and token."""
        self.client = MilvusClient(
            uri=uri,
            token=token
        )
        self._loaded_collections = set()
    
    def _ensure_collection_loaded(self, collection_name: str):
        """Ensure collection is loaded (one-time operation)."""
        if collection_name not in self._loaded_collections:
            try:
                self.client.load_collection(collection_name)
                self._loaded_collections.add(collection_name)
                logger.debug(f"✅ Collection {collection_name} loaded")
            except Exception as e:
                logger.warning(f"⚠️ Collection {collection_name} load failed: {e}")
    
    def _milvus_search(
        self,
        collection: str,
        query_vec: np.ndarray,
        limit: int,
        params: dict,
        tag: str,
    ) -> List[Hit]:
        """Execute single Milvus search with structured results."""
        try:
            self._ensure_collection_loaded(collection)
            
            logger.debug(f"🔍 Executing {tag} search on {collection} with params: {params}")
            
            results = self.client.search(
                collection_name=collection,
                anns_field=self.EMBED_FIELD,
                data=[query_vec.tolist()],
                limit=limit,
                search_params=params,
                output_fields=self.OUTPUT_FIELDS,
            )
            
            # Log the raw results structure for debugging
            logger.debug(f"🔍 Raw Milvus results for {tag} search on {collection}:")
            logger.debug(f"   Type: {type(results)}")
            logger.debug(f"   Length: {len(results) if results else 0}")
            if results and len(results) > 0:
                logger.debug(f"   First result type: {type(results[0])}")
                logger.debug(f"   First result: {results[0]}")
                if len(results[0]) > 0:
                    logger.debug(f"   First hit type: {type(results[0][0])}")
                    logger.debug(f"   First hit: {results[0][0]}")
            
            hits: List[Hit] = []
            
            # Handle the results structure - Milvus returns a list of lists
            if results and len(results) > 0:
                for hit in results[0]:
                    # Log the hit structure for debugging
                    logger.debug(f"🔍 Processing hit: {hit}")
                    logger.debug(f"   Hit type: {type(hit)}")
                    logger.debug(f"   Hit keys: {hit.keys() if isinstance(hit, dict) else 'Not a dict'}")
                    
                    try:
                        # Handle both dictionary and object formats
                        if isinstance(hit, dict):
                            hit_id = str(hit.get('id', 'unknown'))
                            hit_distance = float(hit.get('distance', 0.0))
                            hit_entity = hit.get('entity', {})
                        else:
                            # Try object attributes as fallback
                            hit_id = str(getattr(hit, 'id', 'unknown'))
                            hit_distance = float(getattr(hit, 'distance', 0.0))
                            hit_entity = getattr(hit, 'entity', {})
                        
                        logger.debug(f"🔍 Extracted hit data - ID: {hit_id}, Distance: {hit_distance}")
                        
                        hits.append(
                            Hit(
                                id=hit_id,
                                distance=hit_distance,
                                entity=hit_entity,
                                collection=collection,
                                strategy=tag,
                            )
                        )
                    except Exception as hit_error:
                        logger.error(f"❌ Failed to process hit in {tag} search on {collection}: {hit_error}")
                        logger.error(f"   Hit data: {hit}")
                        logger.error(f"   Hit type: {type(hit)}")
                        continue
            
            logger.debug(f"✅ {tag} search: {len(hits)} hits from {collection}")
            return hits
            
        except Exception as exc:
            logger.error(f"❌ {tag} search failed on {collection}: {exc}")
            logger.error(f"   Collection: {collection}")
            logger.error(f"   Query vector shape: {query_vec.shape}")
            logger.error(f"   Limit: {limit}")
            logger.error(f"   Params: {params}")
            logger.error(f"   Tag: {tag}")
            return []

    def search_similar(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        top_k: int = 2,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection with enhanced parameters."""
        try:
            self._ensure_collection_loaded(collection_name)
            
            # Default search parameters - using L2 metric (compatible with existing collections)
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
            
            # Default output fields to get actual content
            if output_fields is None:
                output_fields = self.OUTPUT_FIELDS
            
            logger.info(f"Searching collection {collection_name} with L2 metric")
            
            # Use embedding as-is for L2 metric
            query_vec = np.array(query_embedding)
            
            results = self.client.search(
                collection_name=collection_name,
                anns_field=self.EMBED_FIELD,
                data=[query_vec.tolist()],
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
        top_k: int = 2,
        output_fields: Optional[List[str]] = None,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors."""
        try:
            self._ensure_collection_loaded(collection_name)
            
            # Default output fields
            if output_fields is None:
                output_fields = self.OUTPUT_FIELDS
            
            logger.info(f"🔄 Attempting hybrid search on {collection_name}")
            
            # Create search requests
            search_requests = []
            
            # Use dense vector as-is for L2 metric
            dense_vec = np.array(dense_vector)
            
            # Dense vector search request
            dense_request = AnnSearchRequest(
                data=[dense_vec.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 8}},
                limit=top_k
            )
            search_requests.append(dense_request)
            
            # Sparse vector search request (if available)
            if sparse_vector:
                sparse_request = AnnSearchRequest(
                    data=[sparse_vector],
                    anns_field="sparse_embedding",
                    param={"metric_type": "L2", "params": {}},
                    limit=top_k
                )
                search_requests.append(sparse_request)
            else:
                logger.info(f"⚠️ No sparse vector provided for hybrid search on {collection_name}")
            
            # Choose reranking strategy
            if rerank_strategy == "weighted" and weights:
                ranker = WeightedRanker(*weights)
            else:
                ranker = RRFRanker(k=60)
            
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
            # Fallback to regular search
            fallback_results = self.search_similar(collection_name, dense_vector, top_k, output_fields)
            
            # Mark fallback results
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
        top_k: int = 2,
        group_size: int = 2,
        strict_group_size: bool = True,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform grouping search to ensure diversity in results."""
        try:
            self._ensure_collection_loaded(collection_name)
            
            # Default search parameters - using L2 metric (compatible with existing collections)
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
            
            # Default output fields
            if output_fields is None:
                output_fields = self.OUTPUT_FIELDS + [group_by_field]
            
            logger.info(f"Performing grouping search on {collection_name} grouped by {group_by_field}")
            
            # Use embedding as-is for L2 metric
            query_vec = np.array(query_embedding)
            
            results = self.client.search(
                collection_name=collection_name,
                anns_field=self.EMBED_FIELD,
                data=[query_vec.tolist()],
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
    """Enhanced wrapper around one Milvus account and a set of collections with concurrent execution."""

    def __init__(self, uri: str, token: str, collection_names: List[str]):
        load_dotenv()
        self.client = MilvusClientWrapper(uri=uri, token=token)
        self.collection_names = collection_names
        self.embedder = EmbeddingGenerator()
        
        # Pre-load all collections once
        logger.info(f"🔄 Pre-loading {len(collection_names)} collections...")
        for name in collection_names:
            self.client._ensure_collection_loaded(name)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for search."""
        _, emb = self.embedder.generate_embeddings([text])
        embedding = np.array(emb[0])
        # Return embedding as-is for L2 metric
        return embedding

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Basic search across all configured collections."""
        query_vec = self._get_embedding(query)
        
        all_results = []
        for collection_name in self.collection_names:
            try:
                results = self.client.search_similar(
                    collection_name=collection_name,
                    query_embedding=query_vec.tolist(),
                    top_k=top_k
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                continue
        
        # Sort by distance (L2: lower is better)
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:top_k]

    def combined_search_enhanced(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Enhanced combined search with concurrent execution and fuzzy deduplication.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of search results as dictionaries
        """
        logger.info(f"🔍 Enhanced search → '{query[:60]}…'")

        # Generate normalized embedding
        qvec = self._get_embedding(query)
        search_tasks: List[tuple[str, str, dict, int]] = []

        # Strategy definitions (L2 metric for existing collections)
        base_params = {"metric_type": "L2", "params": {"nprobe": 8}}
        for col in self.collection_names:
            search_tasks += [
                (col, "base", base_params, top_k),
                (col, "nprobe12", {"metric_type": "L2", "params": {"nprobe": 12}}, top_k),
                (col, "expanded", base_params, min(top_k * 2, 8)),
            ]

        # Execute searches concurrently
        hits: List[Hit] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_tasks), 8)) as pool:
            futures = [
                pool.submit(self.client._milvus_search, c, qvec, lim, p, tag)
                for c, tag, p, lim in search_tasks
            ]
            
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result_hits = fut.result()
                    logger.debug(f"🔍 Received {len(result_hits)} hits from search task")
                    
                    # Validate each hit before adding
                    for hit in result_hits:
                        try:
                            # Verify hit has required attributes
                            if not hasattr(hit, 'id'):
                                logger.error(f"❌ Hit missing 'id' attribute: {hit}")
                                logger.error(f"   Hit type: {type(hit)}")
                                logger.error(f"   Hit dir: {dir(hit) if hasattr(hit, '__dict__') else 'No __dict__'}")
                                continue
                            
                            if not hasattr(hit, 'distance'):
                                logger.error(f"❌ Hit missing 'distance' attribute: {hit}")
                                logger.error(f"   Hit type: {type(hit)}")
                                logger.error(f"   Hit dir: {dir(hit) if hasattr(hit, '__dict__') else 'No __dict__'}")
                                continue
                            
                            if not hasattr(hit, 'entity'):
                                logger.error(f"❌ Hit missing 'entity' attribute: {hit}")
                                logger.error(f"   Hit type: {type(hit)}")
                                logger.error(f"   Hit dir: {dir(hit) if hasattr(hit, '__dict__') else 'No __dict__'}")
                                continue
                            
                            # Log successful hit validation
                            logger.debug(f"✅ Validated hit: ID={hit.id}, Distance={hit.distance}, Strategy={hit.strategy}")
                            hits.append(hit)
                            
                        except Exception as hit_validation_error:
                            logger.error(f"❌ Hit validation failed: {hit_validation_error}")
                            logger.error(f"   Hit object: {hit}")
                            logger.error(f"   Hit type: {type(hit)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"❌ Search task failed: {e}")
                    logger.error(f"   Exception type: {type(e)}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")

        logger.info(f"📊 Total hits from all strategies: {len(hits)}")

        # Fuzzy deduplication with SimHash
        seen_simhashes: set[int] = set()
        deduped: List[Hit] = []
        
        # Sort by distance (L2: lower is better)
        try:
            sorted_hits = sorted(hits, key=lambda x: x.distance)
            logger.debug(f"✅ Sorted {len(sorted_hits)} hits by distance")
        except Exception as sort_error:
            logger.error(f"❌ Failed to sort hits by distance: {sort_error}")
            logger.error(f"   Hits sample: {hits[:3] if hits else 'No hits'}")
            sorted_hits = hits  # Use unsorted as fallback
        
        for h in sorted_hits:
            try:
                content = (h.entity.get("text") or h.entity.get("content", ""))[:300]
                if not content:
                    logger.debug(f"⚠️ Skipping hit with no content: ID={h.id}")
                    continue
                    
                # Generate SimHash for fuzzy duplicate detection
                sig = Simhash(content).value
                
                # Check if similar content already exists (Hamming distance < 5)
                is_duplicate = any(
                    bin(sig ^ s).count("1") < 5 for s in seen_simhashes
                )
                
                if not is_duplicate:
                    seen_simhashes.add(sig)
                    deduped.append(h)
                    logger.debug(f"✅ Added unique hit: ID={h.id}, Distance={h.distance}")
                else:
                    logger.debug(f"🔄 Skipped duplicate hit: ID={h.id}, Distance={h.distance}")
                    
            except Exception as dedup_error:
                logger.error(f"❌ Failed to process hit for deduplication: {dedup_error}")
                logger.error(f"   Hit: {h}")
                logger.error(f"   Hit type: {type(h)}")
                continue

        logger.info(f"🧹 Deduplication: {len(hits)} → {len(deduped)} unique hits")

        # Reciprocal rank fusion with strategy bonus
        try:
            rr_scored = sorted(
                deduped,
                key=lambda h: (
                    1 / (1 + h.distance),  # Reciprocal rank score (L2: lower distance is better)
                    -0.1 if h.strategy == "base" else 0  # Base strategy gets small boost
                ),
                reverse=True,
            )
            logger.debug(f"✅ Applied reciprocal rank fusion to {len(rr_scored)} hits")
        except Exception as rrf_error:
            logger.error(f"❌ Failed to apply reciprocal rank fusion: {rrf_error}")
            logger.error(f"   Deduped hits sample: {deduped[:3] if deduped else 'No hits'}")
            rr_scored = deduped  # Use unsorted as fallback

        # Apply distance threshold for quality control
        final = []
        for h in rr_scored:
            try:
                # For L2 metric, we want distance < threshold (e.g., 1.0)
                if h.distance < 1.0:
                    final.append(h)
                    logger.debug(f"✅ Added final hit: ID={h.id}, Distance={h.distance}, Strategy={h.strategy}")
                else:
                    logger.debug(f"🚫 Filtered out low-quality hit: ID={h.id}, Distance={h.distance:.3f}")
            except Exception as filter_error:
                logger.error(f"❌ Failed to filter hit: {filter_error}")
                logger.error(f"   Hit: {h}")
                logger.error(f"   Hit type: {type(h)}")
                continue

        final = final[:top_k]
        
        logger.info(f"🏁 Final results: {len(final)} / {len(hits)} hits after processing")

        # Convert to dictionary format for compatibility
        result_dicts = []
        for h in final:
            try:
                result_dict = {
                    "id": h.id,
                    "distance": h.distance,
                    "entity": h.entity,
                    "collection": h.collection,
                    "search_type": h.strategy
                }
                result_dicts.append(result_dict)
                logger.debug(f"✅ Converted hit to dict: ID={h.id}")
            except Exception as convert_error:
                logger.error(f"❌ Failed to convert hit to dict: {convert_error}")
                logger.error(f"   Hit: {h}")
                logger.error(f"   Hit type: {type(h)}")
                continue

        # Save results to log file
        self._save_search_results_to_log(query, result_dicts, len(hits), len(deduped))
        
        return result_dicts

    def enhanced_search(
        self, 
        query: str, 
        search_type: str = "both",
        top_k: int = 2,
        group_by_field: Optional[str] = None,
        group_size: int = 2,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Enhanced search with multiple search strategies - now uses refactored search."""
        logger.info(f"Performing enhanced search with type: {search_type}")
        
        # Use the new combined_search_enhanced for all search types
        combined_results = self.combined_search_enhanced(query, top_k)
        
        results = {
            "basic_search": combined_results,
            "hybrid_search": combined_results,
            "grouping_search": combined_results,
            "combined_results": combined_results
        }
        
        return results

    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 2,
        rerank_strategy: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def grouping_search(
        self, 
        query: str, 
        group_by_field: str,
        top_k: int = 2,
        group_size: int = 2,
        strict_group_size: bool = True
    ) -> List[Dict[str, Any]]:
        """Grouping search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def combined_search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Combined search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def enhanced_cross_domain_search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Enhanced cross-domain search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def _save_search_results_to_log(self, query: str, results: List[Dict[str, Any]], total_results: int, unique_results: int):
        """Save search results to a log file in the generated directory."""
        try:
            # Create generated directory if it doesn't exist
            os.makedirs("generated", exist_ok=True)
            
            log_file = os.path.join("generated", "vector_db_results.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"REFACTORED VECTOR DATABASE SEARCH LOG\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Query: '{query}'\n")
                f.write(f"Total Results: {total_results} -> Unique: {unique_results} -> Final: {len(results)}\n")
                f.write(f"Search Method: Concurrent L2-based with SimHash deduplication\n")
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
                    f.write(f"  L2 Distance: {distance:.4f}\n")
                    f.write(f"  Search Strategy: {search_type}\n")
                    f.write(f"  Collection: {collection}\n")
                    f.write(f"  Article/Section: {article if article != 'Unknown Article' else section}\n")
                    f.write(f"  Content: {content[:200]}...\n")
                    f.write(f"{'-'*40}\n")
                
                f.write(f"\n{'='*80}\n\n")
                
        except Exception as e:
            logger.error(f"❌ Error saving search results to log: {e}")

    # Legacy methods for compatibility - all use enhanced search now
    def _generate_legal_query_variants(self, query: str) -> List[str]:
        """Generate query variants - simplified for refactored version."""
        return [query]  # Enhanced search handles diversity internally

    def _multi_metric_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Multi-metric search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def _targeted_variant_search(self, variant: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Targeted variant search - now uses enhanced search."""
        return self.combined_search_enhanced(variant, top_k)

    def _expanded_parameter_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Expanded parameter search - now uses enhanced search."""
        return self.combined_search_enhanced(query, top_k)

    def _intelligent_result_fusion(self, all_results: List[Dict[str, Any]], original_query: str, top_k: int) -> List[Dict[str, Any]]:
        """Intelligent result fusion - now handled by enhanced search internally."""
        return all_results[:top_k]

    def _calculate_domain_relevance_score(self, content: str, query: str) -> float:
        """Calculate domain relevance - simplified for refactored version."""
        return 0.5  # Enhanced search handles relevance internally

    def _get_strategy_diversity_bonus(self, strategy: str) -> float:
        """Strategy diversity bonus - handled internally by enhanced search."""
        return 0.0

    def _advanced_deduplication(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced deduplication - now handled by SimHash in enhanced search."""
        return results

    def _analyze_legal_domains(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Analyze legal domains - simplified for refactored version."""
        return {"analysis": "handled_by_enhanced_search"}

    def _save_enhanced_search_log(self, query: str, results: List[Dict[str, Any]], all_results: List[Dict[str, Any]], 
                                 strategy_counts: Dict[str, int], domain_analysis: Dict[str, Any]):
        """Save enhanced search log - now handled by main logging."""
        pass 