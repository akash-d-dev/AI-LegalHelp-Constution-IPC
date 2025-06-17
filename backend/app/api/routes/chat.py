"""
Chat API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any

from ...models.chat import ChatRequest, ChatResponse, ErrorResponse
from ...services.chat_service import chat_service
from ...core.config import settings
from ...core.logging_config import get_logger

logger = get_logger("chat_api")

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="Send Chat Message")
async def send_message(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the AI Legal Assistant agent
    
    Args:
        request: Chat request containing message and history
        
    Returns:
        Agent response
        
    Raises:
        HTTPException: If message processing fails
    """
    try:
        logger.info(f"Chat request received - Message length: {len(request.message)}")
        logger.debug(f"Chat history length: {len(request.chat_history)}")
        
        # Validate message
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Process message through chat service
        response = await chat_service.process_message(
            message=request.message,
            chat_history=request.chat_history
        )
        
        logger.info(f"Chat response generated successfully in {response.processing_time:.2f}s")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message"
        )

@router.get("/chat/history/format", summary="Get Chat History Format")
async def get_chat_format() -> Dict[str, Any]:
    """
    Get information about the expected chat history format
    
    Returns:
        Format specification for chat history
    """
    return {
        "description": "Expected format for chat history",
        "format": [
            {
                "content": "string - The message content",
                "sender": "string - Either 'user' or 'agent'",
                "timestamp": "datetime - ISO format timestamp"
            }
        ],
        "example": [
            {
                "content": "What are fundamental rights?",
                "sender": "user",
                "timestamp": "2024-01-01T12:00:00Z"
            },
            {
                "content": "Fundamental rights are basic human rights...",
                "sender": "agent", 
                "timestamp": "2024-01-01T12:00:05Z"
            }
        ],
        "limits": {
            "max_message_length": 5000,
            "max_history_length": 100,
            "agent_history_limit": settings.MAX_CHAT_HISTORY
        }
    }

# Future endpoints for chat management
@router.get("/chat/sessions", summary="Get Chat Sessions (Future)")
async def get_chat_sessions():
    """
    Get user's chat sessions (future implementation)
    
    Note: This endpoint is prepared for future MongoDB integration
    """
    return {
        "message": "Chat sessions endpoint - coming soon",
        "feature": "mongodb_integration",
        "status": "planned"
    }

@router.post("/chat/sessions", summary="Create Chat Session (Future)")
async def create_chat_session():
    """
    Create a new chat session (future implementation)
    
    Note: This endpoint is prepared for future MongoDB integration
    """
    return {
        "message": "Create chat session endpoint - coming soon",
        "feature": "mongodb_integration", 
        "status": "planned"
    }

@router.get("/chat/sessions/{session_id}", summary="Get Chat Session (Future)")
async def get_chat_session(session_id: str):
    """
    Get a specific chat session (future implementation)
    
    Args:
        session_id: Unique session identifier
        
    Note: This endpoint is prepared for future MongoDB integration
    """
    return {
        "message": f"Get chat session {session_id} endpoint - coming soon",
        "feature": "mongodb_integration",
        "status": "planned"
    }

@router.delete("/chat/sessions/{session_id}", summary="Delete Chat Session (Future)")
async def delete_chat_session(session_id: str):
    """
    Delete a specific chat session (future implementation)
    
    Args:
        session_id: Unique session identifier
        
    Note: This endpoint is prepared for future MongoDB integration
    """
    return {
        "message": f"Delete chat session {session_id} endpoint - coming soon",
        "feature": "mongodb_integration",
        "status": "planned"
    } 