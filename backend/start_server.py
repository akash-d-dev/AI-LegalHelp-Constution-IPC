#!/usr/bin/env python3
"""
Startup script for AI Legal Assistant Chat API
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    """Start the FastAPI server"""
    
    # Set environment variables
    os.environ.setdefault('ENVIRONMENT', 'development')
    
    print("🚀 Starting AI Legal Assistant Chat API...")
    print("📚 Backend for Indian Constitution & IPC AI Agent")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 