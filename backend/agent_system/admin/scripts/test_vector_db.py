"""Test script for Milvus vector databases with enhanced search capabilities."""

import os
import logging
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import necessary modules
from agent_system.utils.Constants import Constants
from agent_system.utils.vector_db import MilvusVectorDB
from agent_system.utils.embedding_generator import EmbeddingGenerator



class VectorDBTester:
    """Test class for Milvus vector databases with enhanced search capabilities."""
    
    def __init__(self):
        """Initialize the tester with environment variables."""
        logger.info("🔄 Initializing VectorDBTester...")
        
        # Load environment variables
        load_dotenv()
        Constants.set_env_variables()
        
        # Initialize embedding generator
        logger.info("🔄 Initializing embedding generator...")
        self.embedder = EmbeddingGenerator()
        logger.info("✅ Embedding generator initialized")
        
        # Initialize Constitution database
        logger.info("🔄 Initializing Constitution database...")
        try:
            self.constitution_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_COI,
                token=Constants.MILVUS_TOKEN_DB_COI,
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_CONSTITUTION}_{i}" 
                                 for i in range(1, Constants.MILVUS_COLLECTION_COUNT_CONSTITUTION + 1)],
            )
            logger.info("✅ Constitution database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Could not initialize Constitution database: {e}")
            self.constitution_db = None
        
        # Initialize IPC database
        logger.info("🔄 Initializing IPC database...")
        try:
            self.ipc_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_IPC,
                token=Constants.MILVUS_TOKEN_DB_IPC,
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_IPC}_{i}" 
                                 for i in range(1, Constants.MILVUS_COLLECTION_COUNT_IPC + 1)],
            )
            logger.info("✅ IPC database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Could not initialize IPC database: {e}")
            self.ipc_db = None
    
    def test_basic_search(self, db, db_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test basic search functionality."""
        logger.info(f"🔍 Testing basic search on {db_name} database with query: '{query}'")
        
        if db is None:
            logger.error(f"❌ {db_name} database is not available")
            return []
        
        try:
            results = db.search(query, top_k=top_k)
            
            if results:
                logger.info(f"✅ Found {len(results)} results from {db_name} database")
                for i, result in enumerate(results):
                    logger.info(f"📄 Result {i+1}: ID={result.get('id')}, Distance={result.get('distance'):.4f}")
                    entity = result.get('entity', {})
                    if entity:
                        # Try to get content from different possible fields
                        content = entity.get('text') or entity.get('content') or 'No content available'
                        logger.info(f"📝 Content preview: {content[:200]}...")
                    else:
                        logger.warning(f"⚠️ No entity data found for result {i+1}")
            else:
                logger.info(f"📭 No results found in {db_name} database")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error testing {db_name} database: {e}")
            logger.exception("Detailed error information:")
            return []
    
    def test_hybrid_search(self, db, db_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test hybrid search functionality."""
        logger.info(f"🔍 Testing hybrid search on {db_name} database with query: '{query}'")
        
        if db is None:
            logger.error(f"❌ {db_name} database is not available")
            return []
        
        try:
            results = db.hybrid_search(
                query=query, 
                top_k=top_k,
                rerank_strategy="rrf"
            )
            
            if results:
                logger.info(f"✅ Found {len(results)} hybrid search results from {db_name} database")
                for i, result in enumerate(results):
                    logger.info(f"📄 Hybrid Result {i+1}: ID={result.get('id')}, Distance={result.get('distance'):.4f}")
                    entity = result.get('entity', {})
                    if entity:
                        content = entity.get('text') or entity.get('content') or 'No content available'
                        logger.info(f"📝 Content preview: {content[:200]}...")
                    else:
                        logger.warning(f"⚠️ No entity data found for hybrid result {i+1}")
            else:
                logger.info(f"📭 No hybrid search results found in {db_name} database")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid search for {db_name} database: {e}")
            logger.exception("Detailed error information:")
            return []
    
    def test_grouping_search(self, db, db_name: str, query: str, group_by_field: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test grouping search functionality."""
        logger.info(f"🔍 Testing grouping search on {db_name} database with query: '{query}', grouped by: '{group_by_field}'")
        
        if db is None:
            logger.error(f"❌ {db_name} database is not available")
            return []
        
        try:
            results = db.grouping_search(
                query=query,
                group_by_field=group_by_field,
                top_k=top_k,
                group_size=2,
                strict_group_size=False
            )
            
            if results:
                logger.info(f"✅ Found {len(results)} grouping search results from {db_name} database")
                for i, result in enumerate(results):
                    logger.info(f"📄 Grouped Result {i+1}: ID={result.get('id')}, Distance={result.get('distance'):.4f}")
                    entity = result.get('entity', {})
                    if entity:
                        content = entity.get('text') or entity.get('content') or 'No content available'
                        group_value = entity.get(group_by_field, 'Unknown')
                        logger.info(f"📝 Content preview: {content[:200]}...")
                        logger.info(f"🏷️ Group ({group_by_field}): {group_value}")
                    else:
                        logger.warning(f"⚠️ No entity data found for grouped result {i+1}")
            else:
                logger.info(f"📭 No grouping search results found in {db_name} database")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in grouping search for {db_name} database: {e}")
            logger.exception("Detailed error information:")
            return []
    
    def test_enhanced_search(self, db, db_name: str, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Test enhanced search with multiple strategies."""
        logger.info(f"🔍 Testing enhanced search on {db_name} database with query: '{query}'")
        
        if db is None:
            logger.error(f"❌ {db_name} database is not available")
            return {}
        
        try:
            # Test with different group by fields for different databases
            group_by_field = "article" if "Constitution" in db_name else "section"
            
            results = db.enhanced_search(
                query=query,
                search_type="both",  # Both hybrid and grouping
                top_k=top_k,
                group_by_field=group_by_field,
                group_size=2,
                rerank_strategy="rrf"
            )
            
            logger.info(f"✅ Enhanced search completed for {db_name} database")
            
            # Log results for each search type
            for search_type, search_results in results.items():
                if search_results:
                    logger.info(f"📊 {search_type}: {len(search_results)} results")
                    for i, result in enumerate(search_results[:3]):  # Show first 3 results
                        entity = result.get('entity', {})
                        content = entity.get('text') or entity.get('content') or 'No content available'
                        logger.info(f"  📄 {search_type} Result {i+1}: {content[:100]}...")
                else:
                    logger.info(f"📊 {search_type}: No results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced search for {db_name} database: {e}")
            logger.exception("Detailed error information:")
            return {}
    
    def test_enhanced_complex_search(
        self, 
        constitution_db, 
        ipc_db, 
        query: str, 
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Enhanced search method for complex legal queries spanning multiple databases.
        Implements multi-stage hybrid search with cross-database fusion.
        """
        logger.info(f"🔍 Testing enhanced complex search with query: '{query}'")
        start_time = time.time()
        
        try:
            # Stage 1: Query decomposition for legal domain
            legal_query_variants = self._generate_legal_query_variants(query)
            
            # Stage 2: Parallel search across databases with different strategies
            search_tasks = []
            
            # Constitution database - focus on constitutional aspects
            if constitution_db:
                constitution_tasks = [
                    lambda: constitution_db.hybrid_search(
                        query=variant,
                        top_k=top_k * 2,  # Get more results for fusion
                        rerank_strategy="rrf"
                    ) for variant in legal_query_variants['constitution']
                ]
                search_tasks.extend(constitution_tasks)
            
            # IPC database - focus on criminal law aspects  
            if ipc_db:
                ipc_tasks = [
                    lambda: ipc_db.hybrid_search(
                        query=variant,
                        top_k=top_k * 2,
                        rerank_strategy="rrf" 
                    ) for variant in legal_query_variants['ipc']
                ]
                search_tasks.extend(ipc_tasks)
            
            # Execute searches in parallel (simulated)
            constitution_results = []
            ipc_results = []
            
            # Constitution searches
            for variant in legal_query_variants['constitution']:
                if constitution_db:
                    try:
                        results = constitution_db.hybrid_search(
                            query=variant,
                            top_k=top_k,
                            rerank_strategy="rrf"
                        )
                        for result in results:
                            result['source_db'] = 'constitution'
                            result['query_variant'] = variant
                        constitution_results.extend(results)
                    except Exception as e:
                        logger.error(f"Constitution search failed for variant '{variant}': {e}")
            
            # IPC searches
            for variant in legal_query_variants['ipc']:
                if ipc_db:
                    try:
                        results = ipc_db.hybrid_search(
                            query=variant,
                            top_k=top_k,
                            rerank_strategy="rrf"
                        )
                        for result in results:
                            result['source_db'] = 'ipc'
                            result['query_variant'] = variant
                        ipc_results.extend(results)
                    except Exception as e:
                        logger.error(f"IPC search failed for variant '{variant}': {e}")
            
            # Stage 3: Advanced result fusion with domain relevance
            fused_results = self._fuse_cross_database_results(
                constitution_results, 
                ipc_results, 
                query
            )
            
            # Stage 4: Legal cross-reference analysis
            cross_references = self._analyze_legal_cross_references(fused_results)
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Enhanced complex search completed in {execution_time:.2f} seconds")
            logger.info(f"📊 Found {len(constitution_results)} constitution results, {len(ipc_results)} IPC results")
            logger.info(f"🔗 Generated {len(fused_results)} fused results")
            
            return {
                'query': query,
                'strategy': 'enhanced_complex_search',
                'results': fused_results[:top_k],
                'raw_results': {
                    'constitution': constitution_results,
                    'ipc': ipc_results
                },
                'cross_references': cross_references,
                'execution_time': execution_time,
                'metadata': {
                    'total_constitution_results': len(constitution_results),
                    'total_ipc_results': len(ipc_results),
                    'fusion_applied': True,
                    'legal_analysis_applied': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced complex search: {e}")
            return {
                'query': query,
                'error': str(e),
                'results': [],
                'execution_time': time.time() - start_time
            }

    def _generate_legal_query_variants(self, query: str) -> Dict[str, List[str]]:
        """Generate domain-specific query variants for legal search."""
        query_lower = query.lower()
        
        # Legal domain classification
        constitution_keywords = [
            'constitutional', 'article', 'fundamental rights', 'freedom', 
            'speech', 'expression', 'directive principles', 'amendment'
        ]
        
        ipc_keywords = [
            'criminal', 'punishment', 'imprisonment', 'section', 'offense',
            'defamation', 'hate speech', 'liability', 'conviction'
        ]
        
        variants = {
            'constitution': [query],  # Always include original query
            'ipc': [query]           # Always include original query
        }
        
        # Add constitution-focused variants
        if any(kw in query_lower for kw in constitution_keywords):
            variants['constitution'].extend([
                f"Article 19 {query}",
                f"fundamental rights {query}",
                f"constitutional protection {query}",
                f"reasonable restrictions {query}"
            ])
        
        # Add IPC-focused variants
        if any(kw in query_lower for kw in ipc_keywords):
            variants['ipc'].extend([
                f"IPC section {query}",
                f"criminal law {query}",
                f"legal consequences {query}",
                f"punishment provisions {query}"
            ])
        
        # Add cross-domain variants for complex queries
        if (any(kw in query_lower for kw in constitution_keywords) and 
            any(kw in query_lower for kw in ipc_keywords)):
            cross_variants = [
                f"constitutional criminal law interaction {query}",
                f"legal framework balance {query}"
            ]
            variants['constitution'].extend(cross_variants)
            variants['ipc'].extend(cross_variants)
        
        return variants

    def _fuse_cross_database_results(
        self, 
        constitution_results: List[Dict[str, Any]], 
        ipc_results: List[Dict[str, Any]], 
        original_query: str
    ) -> List[Dict[str, Any]]:
        """Fuse results from multiple databases with domain relevance scoring."""
        all_results = []
        
        # Process constitution results
        for result in constitution_results:
            enhanced_result = result.copy()
            enhanced_result['domain_relevance'] = self._calculate_domain_relevance(
                original_query, 'constitution'
            )
            enhanced_result['adjusted_score'] = self._calculate_adjusted_score(
                result, enhanced_result['domain_relevance']
            )
            all_results.append(enhanced_result)
        
        # Process IPC results
        for result in ipc_results:
            enhanced_result = result.copy()
            enhanced_result['domain_relevance'] = self._calculate_domain_relevance(
                original_query, 'ipc'
            )
            enhanced_result['adjusted_score'] = self._calculate_adjusted_score(
                result, enhanced_result['domain_relevance']
            )
            all_results.append(enhanced_result)
        
        # Remove duplicates based on content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by adjusted score (higher is better)
        unique_results.sort(key=lambda x: x.get('adjusted_score', 0), reverse=True)
        
        return unique_results

    def _calculate_domain_relevance(self, query: str, domain: str) -> float:
        """Calculate how relevant the query is to a specific legal domain."""
        query_lower = query.lower()
        
        domain_keywords = {
            'constitution': ['constitutional', 'article', 'fundamental', 'rights', 'freedom'],
            'ipc': ['criminal', 'punishment', 'section', 'offense', 'liability']
        }
        
        if domain not in domain_keywords:
            return 0.5  # Neutral relevance
        
        keywords = domain_keywords[domain]
        relevance_score = sum(1 for kw in keywords if kw in query_lower) / len(keywords)
        
        return min(max(relevance_score, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def _calculate_adjusted_score(self, result: Dict[str, Any], domain_relevance: float) -> float:
        """Calculate adjusted score considering domain relevance."""
        # Get original distance (lower is better for similarity)
        original_distance = result.get('distance', 1.0)
        
        # Convert distance to similarity score (higher is better)
        similarity_score = 1.0 - min(original_distance, 1.0)
        
        # Apply domain relevance boost
        adjusted_score = similarity_score * (0.7 + 0.3 * domain_relevance)
        
        return adjusted_score

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', '')
            
            # Create a simple hash of the content
            content_hash = hash(content[:200])  # Use first 200 chars for comparison
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results

    def _analyze_legal_cross_references(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze legal cross-references and identify important connections."""
        constitution_results = [r for r in results if r.get('source_db') == 'constitution']
        ipc_results = [r for r in results if r.get('source_db') == 'ipc']
        
        analysis = {
            'constitution_articles_found': [],
            'ipc_sections_found': [],
            'potential_conflicts': [],
            'complementary_provisions': [],
            'legal_principles': []
        }
        
        # Extract article/section references
        for result in constitution_results:
            entity = result.get('entity', {})
            article = entity.get('article', 'Unknown')
            if article not in analysis['constitution_articles_found']:
                analysis['constitution_articles_found'].append(article)
        
        for result in ipc_results:
            entity = result.get('entity', {})
            section = entity.get('section', 'Unknown')
            if section not in analysis['ipc_sections_found']:
                analysis['ipc_sections_found'].append(section)
        
        # Identify key legal principles (simplified heuristic)
        if constitution_results and ipc_results:
            analysis['legal_principles'] = [
                'Balance between fundamental rights and criminal law',
                'Constitutional protection vs. legal consequences',
                'Reasonable restrictions framework'
            ]
        
        return analysis

    def format_results(self, results: List[Dict[str, Any]], db_type: str) -> str:
        """Format the search results for display."""
        if not results:
            return f"No results found in {db_type} database."
        
        formatted_results = []
        for i, result in enumerate(results):
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content') or 'No content available'
            distance = result.get('distance', 'Unknown')
            collection = result.get('collection', 'Unknown')
            
            if db_type == "Constitution":
                article = entity.get('article', 'Unknown Article')
                formatted_results.append(
                    f"Result {i+1}:\n"
                    f"Collection: {collection}\n"
                    f"Article: {article}\n"
                    f"Distance: {distance}\n"
                    f"Content: {content}\n"
                )
            else:  # IPC
                section = entity.get('section', 'Unknown Section')
                formatted_results.append(
                    f"Result {i+1}:\n"
                    f"Collection: {collection}\n"
                    f"Section: {section}\n"
                    f"Distance: {distance}\n"
                    f"Content: {content}\n"
                )
        
        return "\n".join(formatted_results)


def save_to_log(results: Dict[str, Any], query: str, db_name: str):
    """Save search results to the log file."""
    log_dir = os.path.join(os.path.dirname(__file__), 'generated')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'search_results.log')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Search Results Log ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"=== {db_name.upper()} DATABASE SEARCH RESULTS ===\n")
        f.write(f'Query: "{query}"\n\n')
        
        for search_type, search_results in results.items():
            f.write(f"--- {search_type.replace('_', ' ').title()} Results ---\n")
            if isinstance(search_results, list):
                for i, result in enumerate(search_results, 1):
                    f.write(f"{i}. Distance: {result.get('distance', 'N/A'):.4f}\n")
                    entity = result.get('entity', {})
                    content = entity.get('text') or entity.get('content') or 'No content available'
                    f.write(f"Content: {content}\n\n")
            f.write("\n")
        
        f.write("="*50 + "\n")


def main():
    """Main function to test the enhanced vector databases."""
    print("=== Enhanced Vector Database Test ===")
    
    # Initialize the tester
    tester = VectorDBTester()
    
    # Test Constitution database
    constitution_query = "DECLRATION UNDER ARTICLE 370(3) OF THE CONSTITUTION"
    print("\n" + "="*80)
    print("TESTING CONSTITUTION DATABASE")
    print("="*80)
    print(f"Query: {constitution_query}")
    print("-" * 80)
    
    # Basic search
    print("\n--- Basic Search ---")
    basic_results = tester.test_basic_search(tester.constitution_db, "Constitution", constitution_query)
    
    # Hybrid search
    print("\n--- Hybrid Search ---")
    hybrid_results = tester.test_hybrid_search(tester.constitution_db, "Constitution", constitution_query)
    
    # Grouping search
    print("\n--- Grouping Search ---")
    grouping_results = tester.test_grouping_search(tester.constitution_db, "Constitution", constitution_query, "article")
    
    # Save Constitution results to log
    constitution_results = {
        'basic_search': basic_results,
        'hybrid_search': hybrid_results,
        'grouping_search': grouping_results
    }
    save_to_log(constitution_results, constitution_query, "Constitution")
    
    # Test IPC database
    ipc_query = "Punishment  for  attempting  to  commit  offences  punishable  with  imprisonment  for  life  or other imprisonment"
    print("\n" + "="*80)
    print("TESTING IPC DATABASE")
    print("="*80)
    print(f"Query: {ipc_query}")
    print("-" * 80)
    
    # Basic search
    print("\n--- Basic Search ---")
    ipc_basic_results = tester.test_basic_search(tester.ipc_db, "IPC", ipc_query)
    
    # Hybrid search
    print("\n--- Hybrid Search ---")
    ipc_hybrid_results = tester.test_hybrid_search(tester.ipc_db, "IPC", ipc_query)
    
    # Grouping search
    print("\n--- Grouping Search ---")
    ipc_grouping_results = tester.test_grouping_search(tester.ipc_db, "IPC", ipc_query, "section")
    
    # Save IPC results to log
    ipc_results = {
        'basic_search': ipc_basic_results,
        'hybrid_search': ipc_hybrid_results,
        'grouping_search': ipc_grouping_results
    }
    save_to_log(ipc_results, ipc_query, "IPC")
    
    # Test the new Enhanced Complex Search for cross-domain queries
    complex_legal_query = """
    What are the constitutional protections for freedom of speech and expression under Article 19, 
    and how do they interact with IPC provisions on hate speech and defamation? Specifically, 
    what are the reasonable restrictions on free speech, and what are the potential legal 
    consequences for violating these restrictions?
    """
    
    print("\n" + "="*80)
    print("TESTING ENHANCED COMPLEX SEARCH (RECOMMENDED STRATEGY)")
    print("="*80)
    print(f"Complex Query: {complex_legal_query.strip()}")
    print("-" * 80)
    
    # Test the enhanced complex search method
    enhanced_complex_results = tester.test_enhanced_complex_search(
        tester.constitution_db, 
        tester.ipc_db, 
        complex_legal_query.strip()
    )
    
    # Save enhanced results to log
    if enhanced_complex_results.get('results'):
        enhanced_log_data = {
            'enhanced_complex_search': enhanced_complex_results
        }
        save_to_log(enhanced_log_data, complex_legal_query.strip(), "Enhanced_Complex")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nConstitution Database Results:")
    if basic_results:
        print(f"✅ Basic Search: {len(basic_results)} results")
    else:
        print("❌ Basic Search: No results")
    
    if hybrid_results:
        print(f"✅ Hybrid Search: {len(hybrid_results)} results")
    else:
        print("❌ Hybrid Search: No results")
    
    if grouping_results:
        print(f"✅ Grouping Search: {len(grouping_results)} results")
    else:
        print("❌ Grouping Search: No results")
    
    print("\nIPC Database Results:")
    if ipc_basic_results:
        print(f"✅ Basic Search: {len(ipc_basic_results)} results")
    else:
        print("❌ Basic Search: No results")
    
    if ipc_hybrid_results:
        print(f"✅ Hybrid Search: {len(ipc_hybrid_results)} results")
    else:
        print("❌ Hybrid Search: No results")
    
    if ipc_grouping_results:
        print(f"✅ Grouping Search: {len(ipc_grouping_results)} results")
    else:
        print("❌ Grouping Search: No results")
    
    print("\nEnhanced Complex Search Results:")
    if enhanced_complex_results.get('results'):
        print(f"✅ Enhanced Complex Search: {len(enhanced_complex_results['results'])} results")
        print(f"⚡ Execution time: {enhanced_complex_results.get('execution_time', 0):.2f} seconds")
        metadata = enhanced_complex_results.get('metadata', {})
        print(f"📊 Constitution results: {metadata.get('total_constitution_results', 0)}")
        print(f"📊 IPC results: {metadata.get('total_ipc_results', 0)}")
        print(f"🔗 Cross-references found: {len(enhanced_complex_results.get('cross_references', {}).get('legal_principles', []))}")
    else:
        print("❌ Enhanced Complex Search: No results")
    
    print("\n✅ Enhanced test completed!")
    print(f"Results have been saved to: generated/search_results.log")


if __name__ == "__main__":
    main() 
    
    
    
    #hybrid search and group search
