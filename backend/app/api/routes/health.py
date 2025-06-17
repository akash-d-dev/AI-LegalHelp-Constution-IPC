"""
Health check API endpoints
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...models.chat import HealthResponse
from ...services.chat_service import chat_service
from ...core.config import settings
from ...core.logging_config import get_logger

logger = get_logger("health_api")

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Check the health of the API and agent system
    
    Returns:
        Health status information
    """
    try:
        logger.info("Health check requested")
        
        # Check agent system health
        agent_health = await chat_service.health_check()
        
        response = HealthResponse(
            status="ok",
            message="Service is running",
            version=settings.VERSION,
            agent_status=agent_health.get("agent_status", "unknown")
        )
        
        logger.info(f"Health check completed - Status: {response.status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        
        return HealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}",
            version=settings.VERSION,
            agent_status="unhealthy"
        )

@router.get("/health/detailed", summary="Detailed Health Check")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Get detailed health information
    
    Returns:
        Detailed health status
    """
    try:
        logger.info("Detailed health check requested")
        
        # Get agent system health
        agent_health = await chat_service.health_check()
        
        health_info = {
            "status": "ok",
            "timestamp": "2024-01-01T00:00:00Z",  # Will be set by Pydantic
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "agent_system": agent_health,
            "configuration": {
                "max_chat_history": settings.MAX_CHAT_HISTORY,
                "agent_timeout": settings.AGENT_TIMEOUT,
                "cors_enabled": len(settings.ALLOWED_ORIGINS) > 0
            }
        }
        
        logger.info("Detailed health check completed successfully")
        return health_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}", exc_info=True)
        
        return {
            "status": "error",
            "message": str(e),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        } 