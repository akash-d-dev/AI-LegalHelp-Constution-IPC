"""
Pydantic models for chat API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class ChatMessage(BaseModel):
    """Individual chat message model"""
    content: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender (user/agent)")
    timestamp: datetime = Field(..., description="Message timestamp")
    
    @validator("sender")
    def validate_sender(cls, v):
        """Validate sender field"""
        if v not in ["user", "agent", "assistant"]:
            raise ValueError("Sender must be 'user', 'agent', or 'assistant'")
        return v

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    chat_history: List[ChatMessage] = Field(default=[], description="Previous chat messages")
    timestamp: Optional[datetime] = Field(default=None, description="Request timestamp")
    
    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        """Set timestamp if not provided"""
        return v or datetime.utcnow()
    
    @validator("chat_history")
    def validate_chat_history(cls, v):
        """Validate chat history length"""
        if len(v) > 100:  # Reasonable limit for chat history
            raise ValueError("Chat history too long")
        return v

class ChatResponse(BaseModel):
    """Chat response model"""
    message: str = Field(..., description="Agent response message")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(default="ok", description="Service status")
    message: str = Field(default="Service is running", description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    agent_status: Optional[str] = Field(default=None, description="Agent system status")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")

# Future models for authentication and user management
class UserCreate(BaseModel):
    """User creation model (future)"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    """User login model (future)"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")

class Token(BaseModel):
    """Authentication token model (future)"""
    access_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None

# Future models for chat persistence
class ChatSession(BaseModel):
    """Chat session model (future)"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = Field(default=[], description="Session messages")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Session metadata") 