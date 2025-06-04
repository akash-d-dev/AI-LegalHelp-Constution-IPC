"""
Example of how to use the Legal AI Agent with memory in an API context.

This demonstrates the expected format for chat history when calling the agent
from your API endpoints.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_system.ai_agent.agent_graph import run_agent, stream_agent
from agent_system.utils.Constants import Constants


def api_chat_example():
    """Example of how to use the agent with chat history in an API context."""
    
    # Setup environment
    Constants.set_env_variables()
    
    print("=== API Chat Example ===")
    
    # Example 1: First message (no chat history)
    print("\n1. First API call (no chat history):")
    current_query = "What are fundamental rights in India?"
    
    # In your API, you would receive this as a parameter
    chat_history = None  # or []
    
    response1 = run_agent(current_query, chat_history)
    print(f"Query: {current_query}")
    print(f"Response: {response1}")
    
    # Example 2: Second message (with chat history)
    print("\n2. Second API call (with chat history):")
    
    # In your API, you would receive this chat history from the frontend
    chat_history = [
        {"role": "user", "content": "What are fundamental rights in India?"},
        {"role": "assistant", "content": response1}
    ]
    
    current_query = "Can you tell me more about Article 21?"
    
    response2 = run_agent(current_query, chat_history)
    print(f"Query: {current_query}")
    print(f"Chat history: {len(chat_history)} messages")
    print(f"Response: {response2}")
    
    # Example 3: Third message (with full chat history)
    print("\n3. Third API call (with full chat history):")
    
    # Update chat history with the previous interaction
    chat_history.extend([
        {"role": "user", "content": current_query},
        {"role": "assistant", "content": response2}
    ])
    
    current_query = "How is this enforced in Indian courts?"
    
    response3 = run_agent(current_query, chat_history)
    print(f"Query: {current_query}")
    print(f"Chat history: {len(chat_history)} messages")
    print(f"Response: {response3}")


def api_streaming_example():
    """Example of using streaming with memory in API context."""
    
    # Setup environment
    Constants.set_env_variables()
    
    print("\n=== API Streaming Example ===")
    
    # Simulate chat history from previous interactions
    chat_history = [
        {"role": "user", "content": "What are fundamental rights?"},
        {"role": "assistant", "content": "Fundamental rights are basic human rights guaranteed by the Indian Constitution..."},
        {"role": "user", "content": "Tell me about Article 21"},
        {"role": "assistant", "content": "Article 21 of the Indian Constitution guarantees the right to life and personal liberty..."}
    ]
    
    current_query = "What are the exceptions to this article?"
    
    print(f"Query: {current_query}")
    print(f"Chat history: {len(chat_history)} messages")
    print("Streaming response: ", end="", flush=True)
    
    full_response = ""
    for chunk in stream_agent(current_query, chat_history):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print(f"\n\nFull response received: {len(full_response)} characters")


# FastAPI example (pseudo-code)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str
    chat_history: List[ChatMessage]

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    # Convert Pydantic models to dict format expected by agent
    chat_history_dict = []
    if request.chat_history:
        chat_history_dict = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.chat_history
        ]
    
    # Call the agent with memory
    response = run_agent(request.query, chat_history_dict)
    
    # Update chat history with new interaction
    updated_history = chat_history_dict + [
        {"role": "user", "content": request.query},
        {"role": "assistant", "content": response}
    ]
    
    return ChatResponse(
        response=response,
        chat_history=[ChatMessage(**msg) for msg in updated_history]
    )

@app.post("/chat/stream")
async def stream_chat_with_agent(request: ChatRequest):
    # Convert to dict format
    chat_history_dict = []
    if request.chat_history:
        chat_history_dict = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.chat_history
        ]
    
    # Stream the response
    async def generate():
        full_response = ""
        for chunk in stream_agent(request.query, chat_history_dict):
            full_response += chunk
            yield f"data: {chunk}\n\n"
        
        # Send final message with updated chat history
        yield f"data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")
"""


if __name__ == "__main__":
    # Run the examples
    api_chat_example()
    api_streaming_example()
    
    print("\n" + "="*80)
    print("API Integration Notes:")
    print("1. Always pass chat_history as a list of dicts with 'role' and 'content' keys")
    print("2. Supported roles: 'user', 'assistant', 'system'")
    print("3. Chat history is optional - agent works without it")
    print("4. Remember to update chat history after each interaction")
    print("5. Consider limiting chat history length for performance")
    print("="*80) 