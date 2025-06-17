"""
Chat service for handling agent interactions
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..core.config import settings
from ..core.logging_config import get_logger
from ..models.chat import ChatMessage, ChatResponse
from agent_system.ai_agent.agent_graph import run_agent

logger = get_logger("chat_service")

class ChatService:
    """Service for handling chat interactions with AI agents"""
    
    def __init__(self):
        """Initialize chat service"""
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("Chat service initialized")
    
    def _prepare_chat_history(self, chat_history: List[ChatMessage]) -> List[Dict[str, str]]:
        """
        Convert frontend chat history to agent format and limit to last N interactions
        
        Args:
            chat_history: List of chat messages from frontend
            
        Returns:
            Formatted chat history for agent
        """
        # Limit to last N interactions (user-agent pairs)
        max_messages = settings.MAX_CHAT_HISTORY * 2  # Each interaction = user + agent message
        
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]
            logger.info(f"Limited chat history to last {max_messages} messages")
        
        # Convert to agent format
        agent_history = []
        for msg in chat_history:
            role = "user" if msg.sender == "user" else "assistant"
            agent_history.append({
                "role": role,
                "content": msg.content
            })
        
        logger.debug(f"Prepared chat history with {len(agent_history)} messages")
        return agent_history
    
    def _run_agent_sync(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Synchronous wrapper for agent execution
        
        Args:
            message: User message
            chat_history: Formatted chat history
            
        Returns:
            Agent response
        """
        try:
            logger.info(f"Running agent for message: {message[:100]}...")
            response = run_agent(message, chat_history)
            logger.info("Agent response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise
    
    async def process_message(
        self, 
        message: str, 
        chat_history: List[ChatMessage]
    ) -> ChatResponse:
        """
        Process a user message and get AI agent response
        
        Args:
            message: User message
            chat_history: Previous conversation history
            
        Returns:
            Agent response
        """
        start_time = time.time()
        
        try:
            # Prepare chat history for agent
            agent_history = self._prepare_chat_history(chat_history)
            
            # Run agent in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._run_agent_sync,
                message,
                agent_history
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Message processed successfully in {processing_time:.2f} seconds")
            
            return ChatResponse(
                message=response,
                metadata={
                    "model": "constitutional_ai_agent",
                    "chat_history_length": len(chat_history),
                    "agent_history_length": len(agent_history)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process message after {processing_time:.2f} seconds: {e}")
            
            # Return error response
            return ChatResponse(
                message=f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                metadata={
                    "error": True,
                    "error_type": type(e).__name__
                },
                processing_time=processing_time
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the chat service and agent system
        
        Returns:
            Health status information
        """
        try:
            # Test a simple agent query to verify functionality
            test_message = "Test health check"
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            test_response = await loop.run_in_executor(
                self.executor,
                self._run_agent_sync,
                test_message,
                []
            )
            
            response_time = time.time() - start_time
            
            return {
                "agent_status": "healthy",
                "response_time": f"{response_time:.2f}s",
                "test_successful": True
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "agent_status": "unhealthy",
                "error": str(e),
                "test_successful": False
            }
    
    def __del__(self):
        """Cleanup thread pool on service destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Create singleton instance
chat_service = ChatService() 