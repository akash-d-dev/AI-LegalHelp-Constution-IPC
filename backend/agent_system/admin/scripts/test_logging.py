"""Test script to verify logging functionality."""


from agent_system.utils.Constants import Constants
from agent_system.ai_agent.agent_graph import run_agent


def main():
    """Test the legal AI agent with logging."""
    
    query = "What is Section 511 of IPC about attempting to commit offences?"
    
    print("=== Testing Agent with Logging ===")
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        response = run_agent(query)
        print(f"Response: {response}")
        print("\n✅ Agent test completed successfully!")
        print("\n📁 Check the following log files in ./generated/:")
        print("   - agent_execution.log (detailed agent execution logs)")
        print("   - agent_conversations.log (conversation summaries)")
        print("   - vector_db_results.log (vector database search results)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    Constants.set_env_variables()
    main() 