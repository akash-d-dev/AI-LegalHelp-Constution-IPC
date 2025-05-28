"""Test script for the Legal AI Agent."""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ai_agent.agent_graph import run_agent, stream_agent
from backend.utils.Constants import Constants

def test_agent():
    """Test the legal AI agent with sample queries."""
    
    test_queries = [
        "What are the fundamental rights guaranteed by the Indian Constitution?",
        "What is the punishment for theft under IPC?",
        "Can the government restrict freedom of speech? Under what circumstances?",
        "What happens if someone commits murder in India?"
    ]
    
    print("=== Legal AI Agent Test ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 50)
        
        try:
            # Test the agent
            response = run_agent(query)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*80 + "\n")


def interactive_test():
    """Interactive test mode."""
    print("=== Interactive Legal AI Agent ===")
    print("Ask questions about Indian Constitution and IPC.")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            print("\nThinking...")
            response = run_agent(query)
            print(f"\nAnswer: {response}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # Run interactive test by default
    Constants.set_env_variables()
    interactive_test() 