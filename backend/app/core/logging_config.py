"""
Logging configuration for the AI Legal Assistant Chat API
"""

import logging
import sys
from typing import Optional
from .config import settings

def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Override the default log level
        
    Returns:
        Logger instance
    """
    
    # Use provided log level or default from settings
    level = log_level or settings.LOG_LEVEL
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Get logger for this application
    logger = logging.getLogger("constitutional_ai_api")
    
    # Set log levels for specific modules
    if settings.ENVIRONMENT == "production":
        # Reduce noise in production
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    else:
        # More verbose in development
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
    
    # Configure agent system logging
    logging.getLogger("agent_system").setLevel(logging.INFO)
    
    logger.info(f"Logging configured - Level: {level}, Environment: {settings.ENVIRONMENT}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"constitutional_ai_api.{name}") 