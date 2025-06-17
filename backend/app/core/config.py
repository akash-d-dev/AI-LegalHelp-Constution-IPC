"""
Configuration settings for the Constitutional AI Chat API
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_NAME: str = "Constitutional AI Chat API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ]
    
    # Agent settings
    MAX_CHAT_HISTORY: int = 5  # Only send last 5 interactions to AI
    AGENT_TIMEOUT: int = 30  # Timeout for agent responses in seconds
    
    # Database settings (for future MongoDB integration)
    DATABASE_URL: str = "mongodb://localhost:27017/constitutional_ai"
    
    # Authentication settings (for future)
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        """Validate and process CORS origins"""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting"""
        if v not in ["development", "staging", "production"]:
            return "development"
        return v
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables from agent system

# Create settings instance
settings = Settings()

# Update debug mode based on environment
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False 