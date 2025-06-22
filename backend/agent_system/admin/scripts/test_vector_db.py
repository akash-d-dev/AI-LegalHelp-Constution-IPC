#!/usr/bin/env python3
"""
Test script for vector database search functionality.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_system.utils.vector_db import MilvusVectorDB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_vector_search():
    """Test the vector search functionality."""
    
    # Get environment variables
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN", "")
    
    # Test collections
    test_collections = ["ipc_1", "ipc_2"]
    
    print("🔍 Testing Vector Database Search")
    print(f"URI: {uri}")
    print(f"Collections: {test_collections}")
    print("=" * 50)
    
    try:
        # Initialize the vector database
        vector_db = MilvusVectorDB(uri, token, test_collections)
        print("✅ Vector database initialized successfully")
        
        # Test query
        test_query = "murder"
        print(f"\n🔍 Testing search with query: '{test_query}'")
        
        # Perform search
        results = vector_db.combined_search_enhanced(test_query, top_k=3)
        
        print(f"\n📊 Search Results:")
        print(f"Total results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  ID: {result.get('id', 'N/A')}")
            print(f"  Distance: {result.get('distance', 'N/A')}")
            print(f"  Collection: {result.get('collection', 'N/A')}")
            print(f"  Search Type: {result.get('search_type', 'N/A')}")
            
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', 'No content')
            print(f"  Content: {content[:100]}...")
        
        if results:
            print(f"\n✅ Search test PASSED - Found {len(results)} results")
        else:
            print(f"\n⚠️ Search test completed but no results found")
            
    except Exception as e:
        print(f"\n❌ Search test FAILED: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_vector_search()
    sys.exit(0 if success else 1) 
    
    
    
    #hybrid search and group search
