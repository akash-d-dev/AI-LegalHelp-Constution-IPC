# AI Agent Memory Functionality

## Overview

The Legal AI Agent now supports conversation memory, allowing it to maintain context across multiple interactions. This is essential for API integrations where chat history needs to be preserved between requests.

## Key Features

### 1. **Optional Chat History Parameter**
Both `run_agent()` and `stream_agent()` functions now accept an optional `chat_history` parameter:

```python
from agent_system.ai_agent.agent_graph import run_agent, stream_agent

# Without memory (fresh conversation)
response = run_agent("What are fundamental rights?")

# With memory (conversation context)
chat_history = [
    {"role": "user", "content": "What are fundamental rights?"},
    {"role": "assistant", "content": "Fundamental rights are..."},
]
response = run_agent("Tell me about Article 21", chat_history)
```

### 2. **Supported Message Roles**
- `user`: User messages/queries
- `assistant`: AI agent responses
- `system`: System messages (optional)

### 3. **Backwards Compatibility**
The functions remain backwards compatible - existing code without chat history will continue to work unchanged.

## API Integration Example

### FastAPI Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from agent_system.ai_agent.agent_graph import run_agent

app = FastAPI()

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = None

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # Convert to expected format
    chat_history_dict = []
    if request.chat_history:
        chat_history_dict = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.chat_history
        ]
    
    # Call agent with memory
    response = run_agent(request.query, chat_history_dict)
    
    # Return updated chat history
    updated_history = chat_history_dict + [
        {"role": "user", "content": request.query},
        {"role": "assistant", "content": response}
    ]
    
    return {
        "response": response,
        "chat_history": updated_history
    }
```

### Example API Request

```json
{
  "query": "Can you explain Article 21 from the previous topic?",
  "chat_history": [
    {
      "role": "user",
      "content": "What are fundamental rights in India?"
    },
    {
      "role": "assistant", 
      "content": "Fundamental rights are basic human rights guaranteed by the Indian Constitution..."
    }
  ]
}
```

## Chat History Format

### Required Format
```python
chat_history = [
    {"role": "user", "content": "Your question here"},
    {"role": "assistant", "content": "Agent's response here"},
    {"role": "user", "content": "Follow-up question"},
    {"role": "assistant", "content": "Agent's follow-up response"}
]
```

### Best Practices

1. **Always include both user and assistant messages** in the chat history
2. **Maintain chronological order** of messages
3. **Consider limiting chat history length** for performance (recommend max 20-30 messages)
4. **Include only relevant context** - not all conversation history may be necessary

## Function Signatures

### run_agent()
```python
def run_agent(query: str, chat_history: Optional[List[dict]] = None) -> str:
    """
    Run the agent with a query and optional chat history.
    
    Args:
        query: The current user question/query
        chat_history: Optional list of previous messages
        
    Returns:
        str: The agent's response to the current query
    """
```

### stream_agent()
```python
def stream_agent(query: str, chat_history: Optional[List[dict]] = None):
    """
    Stream the agent execution with optional chat history.
    
    Args:
        query: The current user question/query  
        chat_history: Optional list of previous messages
        
    Yields:
        str: Chunks of the agent's response as they become available
    """
```

## Logging and Monitoring

The agent now provides enhanced logging for memory usage:

- Chat history presence and length
- Message processing details
- Context building information
- Conversation history saved to `generated/agent_conversations.log`

## Testing

Use the updated test script to verify memory functionality:

```bash
python agent_system/admin/scripts/test_agent.py
```

Choose option 2 for "Memory functionality test" to see the agent maintaining context across multiple queries.

## Error Handling

- Unknown message roles are treated as user messages with a warning
- Empty chat history is handled gracefully
- Invalid message format logs errors but doesn't crash the agent

## Performance Considerations

1. **Chat History Length**: Longer chat histories increase processing time and token usage
2. **Message Size**: Large messages in chat history consume more memory
3. **Database Connections**: Each agent call initializes fresh database connections
4. **Recommendation**: Implement chat history trimming in your API for optimal performance

## Migration Guide

### Existing Code (No Changes Required)
```python
# This continues to work as before
response = run_agent("What are fundamental rights?")
```

### New Code with Memory
```python
# Add chat history for context
chat_history = [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
]
response = run_agent("Follow-up question", chat_history)
```

## Troubleshooting

### Common Issues

1. **"Chat history format error"**: Ensure each message has `role` and `content` keys
2. **"Context too long"**: Reduce chat history length or message sizes  
3. **"Memory not working"**: Verify chat history is passed as a list of dictionaries

### Debug Logging

Enable debug logging to see detailed memory processing:

```python
import logging
logging.getLogger('agent_system.ai_agent').setLevel(logging.DEBUG)
```

## Examples

See `agent_system/examples/api_usage_example.py` for complete working examples of:
- Basic chat with memory
- Streaming responses with memory
- FastAPI integration patterns
- Error handling examples 