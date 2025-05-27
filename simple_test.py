"""Simple test script for the Legal AI Agent."""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ai_agent.agent_graph import run_agent


def main():
    """Test the legal AI agent with a simple query."""
    
    query = "What are the fundamental rights guaranteed by the Indian Constitution?"
    
    print("=== Legal AI Agent Simple Test ===")
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        response = run_agent(query)
        print(f"Response: {response}")
        print("\n✅ Agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 