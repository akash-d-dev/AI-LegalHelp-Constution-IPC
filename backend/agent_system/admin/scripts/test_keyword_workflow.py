"""Test script to verify that the agent now uses generate_keywords first before searching."""

import logging
from dotenv import load_dotenv

# Configure logging to see tool execution order
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from agent_system.utils.Constants import Constants
from agent_system.ai_agent.agent_graph import run_agent

def test_keyword_workflow():
    """Test that the agent uses generate_keywords before search operations."""
    
    print("🧪 Testing Agent Keyword Workflow")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    Constants.set_env_variables()
    
    # Test query that should trigger search
    test_query = "What are the fundamental rights in Indian Constitution?"
    
    print(f"🔍 Testing with query: '{test_query}'")
    print("-" * 30)
    
    try:
        result = run_agent(test_query)
        
        print(f"\n✅ Agent completed successfully!")
        print(f"📝 Response preview: {result[:200]}...")
        
        print(f"\n💡 Check the agent_execution.log in generated/ folder to see if:")
        print(f"   1. generate_keywords was called first")
        print(f"   2. search_constitution was called after generate_keywords")
        print(f"   3. The agent followed the proper workflow")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_keyword_workflow() 