"""Test script for the Legal AI Agent."""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from agent_system.ai_agent.agent_graph import run_agent, stream_agent, get_conversation_history_format
from agent_system.utils.Constants import Constants

def test_agent():
    """Test the legal AI agent with sample queries."""
    
    test_queries = [
        # "What are the fundamental rights guaranteed by the Indian Constitution?",
        # "What is the punishment for theft under IPC?",
        "What are the constitutional protections for freedom of speech and expression under Article 19, and how do they interact with IPC provisions on hate speech and defamation? Specifically, what are the reasonable restrictions on free speech, and what are the potential legal consequences for violating these restrictions?"
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


def test_agent_with_memory():
    """Test the agent with conversation memory functionality."""
    
    print("=== Legal AI Agent Memory Test ===\n")
    
    # Simulate a conversation with memory
    chat_history = []
    
    # First interaction
    query1 = "What are fundamental rights in Indian Constitution?"
    print(f"Query 1: {query1}")
    print("-" * 50)
    
    try:
        response1 = run_agent(query1, chat_history)
        print(f"Response 1: {response1}\n")
        
        # Add to chat history
        chat_history.extend([
            {"role": "user", "content": query1},
            {"role": "assistant", "content": response1}
        ])
        
    except Exception as e:
        print(f"Error in query 1: {e}\n")
        return
    
    # Second interaction with context
    query2 = "Can you explain Article 21 from the previous topic?"
    print(f"Query 2 (with context): {query2}")
    print("-" * 50)
    
    try:
        response2 = run_agent(query2, chat_history)
        print(f"Response 2: {response2}\n")
        
        # Add to chat history
        chat_history.extend([
            {"role": "user", "content": query2},
            {"role": "assistant", "content": response2}
        ])
        
    except Exception as e:
        print(f"Error in query 2: {e}\n")
        return
    
    # Third interaction with full context
    query3 = "How does this relate to the punishment for violating these rights?"
    print(f"Query 3 (with full context): {query3}")
    print("-" * 50)
    
    try:
        response3 = run_agent(query3, chat_history)
        print(f"Response 3: {response3}\n")
        
    except Exception as e:
        print(f"Error in query 3: {e}\n")
    
    print(f"Final chat history length: {len(chat_history)} messages")
    print("="*80 + "\n")


def interactive_test():
    """Interactive test mode."""
    print("=== Interactive Legal AI Agent ===")
    print("Ask questions about Indian Constitution and IPC.")
    print("Type 'exit' to quit, 'clear' to clear chat history.\n")
    
    chat_history = []
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'clear':
            chat_history = []
            print("Chat history cleared.\n")
            continue
        
        if not query:
            continue
        
        try:
            print("\nThinking...")
            response = run_agent(query, chat_history)
            print(f"\nAnswer: {response}\n")
            
            # Add to chat history
            chat_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ])
            
            print(f"[Chat history: {len(chat_history)} messages]")
            
        except Exception as e:
            print(f"Error: {e}\n")


def interactive_test_with_streaming():
    """Interactive test mode with streaming responses."""
    print("=== Interactive Legal AI Agent (Streaming) ===")
    print("Ask questions about Indian Constitution and IPC.")
    print("Type 'exit' to quit, 'clear' to clear chat history.\n")
    
    chat_history = []
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'clear':
            chat_history = []
            print("Chat history cleared.\n")
            continue
        
        if not query:
            continue
        
        try:
            print("\nStreaming response...")
            print("Answer: ", end="", flush=True)
            
            full_response = ""
            for chunk in stream_agent(query, chat_history):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print("\n")
            
            # Add to chat history
            chat_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": full_response}
            ])
            
            print(f"[Chat history: {len(chat_history)} messages]")
            
        except Exception as e:
            print(f"Error: {e}\n")


def show_chat_history_format():
    """Display the expected chat history format."""
    print("=== Chat History Format ===")
    format_info = get_conversation_history_format()
    
    print(f"Description: {format_info['description']}")
    print(f"Supported roles: {format_info['supported_roles']}")
    print(f"Note: {format_info['note']}\n")
    
    print("Example format:")
    for i, msg in enumerate(format_info['format'], 1):
        print(f"  {i}. {msg}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Setup environment
    Constants.set_env_variables()
    
    # Show menu
    print("Choose test mode:")
    print("1. Basic agent test")
    print("2. Memory functionality test") 
    print("3. Interactive mode (with memory)")
    print("4. Interactive streaming mode (with memory)")
    print("5. Show chat history format")
    print("6. Run all tests")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        test_agent()
    elif choice == "2":
        test_agent_with_memory()
    elif choice == "3":
        interactive_test()
    elif choice == "4":
        interactive_test_with_streaming()
    elif choice == "5":
        show_chat_history_format()
    elif choice == "6":
        show_chat_history_format()
        test_agent()
        test_agent_with_memory()
    else:
        print("Invalid choice. Running memory test by default...")
        test_agent_with_memory()