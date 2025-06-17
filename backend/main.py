"""
FastAPI backend for Constitutional AI Chat Application
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Add the backend directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging_config import setup_logging
from agent_system.utils.Constants import Constants

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Constitutional AI Chat API...")
    
    # Initialize environment variables for agent system
    Constants.set_env_variables()
    logger.info("Agent system environment initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Constitutional AI Chat API...")

# Create FastAPI application
app = FastAPI(
    title="Constitutional AI Chat API",
    description="API for interacting with Indian Constitution and IPC AI agents",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Constitutional AI Chat API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.ENVIRONMENT == "development" else "disabled"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    ) 