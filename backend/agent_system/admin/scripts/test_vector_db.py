"""Test script for Milvus vector databases with enhanced search capabilities."""

import os
import logging
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
    
    print("\n✅ Enhanced test completed!")
    print(f"Results have been saved to: generated/search_results.log")


if __name__ == "__main__":
    main() 
    
    
    
    #hybrid search and group search
