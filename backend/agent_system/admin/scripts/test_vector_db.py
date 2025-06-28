#!/usr/bin/env python3
"""
Interactive Vector Database Testing Tool
Allows testing different search strategies on Constitution and IPC databases
"""

import sys
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_system.utils.vector_db import MilvusVectorDB
from agent_system.utils.Constants import Constants
from agent_system.utils.embedding_generator import EmbeddingGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDBTester:
    """Interactive testing tool for vector database search strategies."""
    
    def __init__(self):
        """Initialize the tester with database connections and embedding generator."""
        self.setup_environment()
        self.embedding_generator = EmbeddingGenerator()
        self.constitution_db = None
        self.ipc_db = None
        self.results_dir = "generated/db_query_results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Available search strategies
        self.search_strategies = {
            "1": ("basic_search", "Basic Search"),
            "2": ("combined_search_enhanced", "Enhanced Combined Search (Recommended)"),
            "3": ("enhanced_search", "Enhanced Search with Multiple Strategies"),
            "4": ("hybrid_search", "Hybrid Search"),
            "5": ("grouping_search", "Grouping Search"),
            "6": ("combined_search", "Combined Search"),
            "7": ("enhanced_cross_domain_search", "Enhanced Cross-Domain Search"),
            "8": ("all_strategies", "Test All Strategies")
        }
        
        # Available databases
        self.databases = {
            "1": ("constitution", "Constitution of India Database"),
            "2": ("ipc", "Indian Penal Code Database"),
            "3": ("both", "Both Databases")
        }
    
    def setup_environment(self):
        """Set up environment variables."""
        try:
            Constants.set_env_variables()
            print("✅ Environment variables loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load environment variables: {e}")
            sys.exit(1)
    
    def initialize_databases(self, db_choice: str):
        """Initialize the selected database(s)."""
        try:
            if db_choice in ["constitution", "both"]:
                print("🔄 Initializing Constitution database...")
                if not Constants.MILVUS_URI_DB_COI or not Constants.MILVUS_TOKEN_DB_COI:
                    raise ValueError("Constitution database credentials not found in environment variables")
                
                self.constitution_db = MilvusVectorDB(
                    uri=Constants.MILVUS_URI_DB_COI,
                    token=Constants.MILVUS_TOKEN_DB_COI,
                    collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_CONSTITUTION}_{i}" 
                                    for i in range(1, Constants.MILVUS_COLLECTION_COUNT_CONSTITUTION + 1)]
                )
                print("✅ Constitution database initialized")
            
            if db_choice in ["ipc", "both"]:
                print("🔄 Initializing IPC database...")
                if not Constants.MILVUS_URI_DB_IPC or not Constants.MILVUS_TOKEN_DB_IPC:
                    raise ValueError("IPC database credentials not found in environment variables")
                
                self.ipc_db = MilvusVectorDB(
                    uri=Constants.MILVUS_URI_DB_IPC,
                    token=Constants.MILVUS_TOKEN_DB_IPC,
                    collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_IPC}_{i}" 
                                    for i in range(1, Constants.MILVUS_COLLECTION_COUNT_IPC + 1)]
                )
                print("✅ IPC database initialized")
                
        except Exception as e:
            print(f"❌ Failed to initialize database(s): {e}")
            sys.exit(1)
    
    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*60)
        print("🔍 VECTOR DATABASE TESTING TOOL")
        print("="*60)
        
        print("\n📊 Available Databases:")
        for key, (_, description) in self.databases.items():
            print(f"  {key}. {description}")
        
        print("\n🛠️ Available Search Strategies:")
        for key, (_, description) in self.search_strategies.items():
            print(f"  {key}. {description}")
        
        print("\n" + "="*60)
    
    def get_user_input(self) -> tuple[str, str, str, int]:
        """Get user input for database, strategy, query, and top_k."""
        # Get database choice
        while True:
            db_choice = input("\nSelect database (1-3): ").strip()
            if db_choice in self.databases:
                db_name = self.databases[db_choice][0]
                break
            print("❌ Invalid choice. Please select 1, 2, or 3.")
        
        # Get search strategy
        while True:
            strategy_choice = input("\nSelect search strategy (1-8): ").strip()
            if strategy_choice in self.search_strategies:
                strategy_name = self.search_strategies[strategy_choice][0]
                break
            print("❌ Invalid choice. Please select 1-8.")
        
        # Get query
        query = input("\nEnter your search query: ").strip()
        if not query:
            print("❌ Query cannot be empty.")
            return self.get_user_input()
        
        # Get top_k
        while True:
            try:
                top_k = int(input("\nEnter number of results to retrieve (default 3): ").strip() or "3")
                if top_k > 0:
                    break
                print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        return db_name, strategy_name, query, top_k
    
    def execute_search(self, db_name: str, strategy_name: str, query: str, top_k: int) -> Dict[str, Any]:
        """Execute the search with the specified parameters."""
        results = {}
        
        if strategy_name == "all_strategies":
            # Test all strategies
            for strategy_key, (actual_strategy, strategy_desc) in self.search_strategies.items():
                if actual_strategy == "all_strategies":
                    continue
                print(f"\n🔍 Testing {strategy_desc}...")
                strategy_results = self._execute_single_search(db_name, actual_strategy, query, top_k)
                results[actual_strategy] = strategy_results
        else:
            # Single strategy
            results[strategy_name] = self._execute_single_search(db_name, strategy_name, query, top_k)
        
        return results
    
    def _execute_single_search(self, db_name: str, strategy_name: str, query: str, top_k: int) -> Dict[str, Any]:
        """Execute a single search strategy."""
        strategy_results = {}
        
        try:
            if db_name in ["constitution", "both"] and self.constitution_db:
                print(f"  🔍 Searching Constitution database...")
                constitution_results = self._search_database(self.constitution_db, strategy_name, query, top_k)
                strategy_results["constitution"] = constitution_results
            
            if db_name in ["ipc", "both"] and self.ipc_db:
                print(f"  🔍 Searching IPC database...")
                ipc_results = self._search_database(self.ipc_db, strategy_name, query, top_k)
                strategy_results["ipc"] = ipc_results
                
        except Exception as e:
            error_msg = f"Error executing {strategy_name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            strategy_results["error"] = error_msg
        
        return strategy_results
    
    def _search_database(self, db: MilvusVectorDB, strategy_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Execute search on a specific database."""
        try:
            if strategy_name == "basic_search":
                return db.search(query, top_k)
            elif strategy_name == "combined_search_enhanced":
                return db.combined_search_enhanced(query, top_k)
            elif strategy_name == "enhanced_search":
                result = db.enhanced_search(query, top_k=top_k)
                return result.get("combined_results", [])
            elif strategy_name == "hybrid_search":
                return db.hybrid_search(query, top_k)
            elif strategy_name == "grouping_search":
                # Try grouping by common fields, fallback to basic search
                try:
                    return db.grouping_search(query, "article", top_k)
                except:
                    return db.search(query, top_k)
            elif strategy_name == "combined_search":
                return db.combined_search(query, top_k)
            elif strategy_name == "enhanced_cross_domain_search":
                return db.enhanced_cross_domain_search(query, top_k)
            else:
                return db.search(query, top_k)  # Fallback
                
        except Exception as e:
            logger.error(f"Search failed for strategy {strategy_name}: {e}")
            return []
    
    def save_results(self, query: str, db_name: str, strategy_name: str, results: Dict[str, Any], top_k: int):
        """Save search results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        result_data = {
            "timestamp": timestamp,
            "query": query,
            "database": db_name,
            "strategy": strategy_name,
            "top_k": top_k,
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the search results."""
        summary = {
            "total_strategies_tested": len(results),
            "strategies": {},
            "total_results": 0
        }
        
        for strategy, strategy_results in results.items():
            strategy_summary = {
                "databases_searched": list(strategy_results.keys()) if isinstance(strategy_results, dict) else [],
                "total_results": 0,
                "has_error": "error" in strategy_results
            }
            
            if isinstance(strategy_results, dict) and "error" not in strategy_results:
                for db_name, db_results in strategy_results.items():
                    if isinstance(db_results, list):
                        strategy_summary["total_results"] += len(db_results)
                        summary["total_results"] += len(db_results)
            
            summary["strategies"][strategy] = strategy_summary
        
        return summary
    
    def display_results(self, results: Dict[str, Any]):
        """Display search results in a formatted way."""
        print("\n" + "="*60)
        print("📊 SEARCH RESULTS")
        print("="*60)
        
        for strategy, strategy_results in results.items():
            print(f"\n🛠️ Strategy: {strategy.upper()}")
            print("-" * 40)
            
            if "error" in strategy_results:
                print(f"❌ Error: {strategy_results['error']}")
                continue
            
            for db_name, db_results in strategy_results.items():
                print(f"\n📚 Database: {db_name.upper()}")
                
                if not db_results:
                    print("  ⚠️ No results found")
                    continue
                
                for i, result in enumerate(db_results[:3], 1):  # Show top 3
                    entity = result.get('entity', {})
                    content = entity.get('text') or entity.get('content', 'No content')
                    distance = result.get('distance', 'Unknown')
                    collection = result.get('collection', 'Unknown')
                    
                    print(f"\n  Result {i}:")
                    print(f"    Distance: {distance}")
                    print(f"    Collection: {collection}")
                    print(f"    Content: {content[:150]}...")
    
    def run(self):
        """Run the interactive testing tool."""
        print("🚀 Starting Vector Database Testing Tool...")
        
        while True:
            try:
                self.display_menu()
                
                # Get user input
                db_name, strategy_name, query, top_k = self.get_user_input()
                
                # Initialize databases
                self.initialize_databases(db_name)
                
                print(f"\n🔍 Executing search...")
                print(f"   Query: '{query}'")
                print(f"   Database: {db_name}")
                print(f"   Strategy: {strategy_name}")
                print(f"   Top K: {top_k}")
                
                # Execute search
                results = self.execute_search(db_name, strategy_name, query, top_k)
                
                # Display results
                self.display_results(results)
                
                # Save results
                self.save_results(query, db_name, strategy_name, results, top_k)
                
                # Ask if user wants to continue
                while True:
                    continue_choice = input("\nDo you want to perform another search? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes']:
                        break
                    elif continue_choice in ['n', 'no']:
                        print("👋 Thank you for using the Vector Database Testing Tool!")
                        return
                    else:
                        print("❌ Please enter 'y' or 'n'.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Testing tool interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again.")

def main():
    """Main function to run the testing tool."""
    tester = VectorDBTester()
    tester.run()

if __name__ == "__main__":
    main() 
    
    
    
    #hybrid search and group search
