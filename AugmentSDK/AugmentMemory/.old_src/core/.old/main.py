#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment SDK - FastAPI Entry Point

This module serves as the main entry point for the Augment SDK API server,
providing HTTP endpoints for interacting with the SDK's hierarchical memory
system. It initializes the FastAPI application, configures middleware,
sets up routing, and handles application lifecycle events.

The server exposes endpoints for all memory operations including:
- Memory storage across different memory layers (ephemeral, working, semantic, etc.)
- Memory retrieval with context-aware recall
- Meta-cognitive operations for self-reflection and memory refinement
- Memory analytics and monitoring
- Dynamic domain adaptation for specialized memory contexts

Usage:
    Run directly: python -m augment_sdk.api.main
    Via uvicorn: uvicorn augment_sdk.api.main:app --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union, Any, Callable

import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

# Internal imports from Augment SDK
from augment_sdk.config import settings
from augment_sdk.core.logging import setup_logging, get_logger
from augment_sdk.memory.components.memory_manager import MemoryManager
from augment_sdk.memory.utils.config import load_config
from augment_sdk.memory.api.routes import (
    memory_router,
    analytics_router, 
    meta_cognition_router
)
from augment_sdk.memory.api.models.api_models import (
    StatusResponse,
    ErrorResponse
)
from augment_sdk.version import __version__

# Configure logging
logger = get_logger(__name__)

# Create memory manager instance to be used across the application
config = load_config()
memory_manager = MemoryManager(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle events.
    
    Handles startup and shutdown events, initializing necessary services
    on startup and cleaning up resources on shutdown.
    
    Args:
        app: The FastAPI application instance
    """
    # Startup operations
    logger.info("Starting Augment SDK API server")
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize memory systems
    try:
        await memory_manager.initialize()
        logger.info("Memory systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory systems: {str(e)}")
        # Continue anyway to allow partial functionality
    
    yield  # Application runs here
    
    # Shutdown operations
    logger.info("Shutting down Augment SDK API server")
    try:
        await memory_manager.shutdown()
        logger.info("Memory systems shut down gracefully")
    except Exception as e:
        logger.error(f"Error during memory systems shutdown: {str(e)}")


# Initialize FastAPI application
app = FastAPI(
    title="Augment SDK API",
    description="API for interacting with Augment SDK's hierarchical memory systems",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.SHOW_DOCS else None,
    redoc_url="/redoc" if settings.SHOW_DOCS else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add gzip compression for API responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """
    Middleware to track and log request processing time.
    
    Args:
        request: The incoming HTTP request
        call_next: The next middleware in the chain
        
    Returns:
        The HTTP response
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    logger.debug(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Process time: {process_time:.4f}s"
    )
    return response


# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler to catch and format all unhandled exceptions.
    
    Args:
        request: The incoming HTTP request
        exc: The raised exception
        
    Returns:
        A formatted JSON response with error details
    """
    error_id = f"err_{int(time.time())}"
    logger.exception(f"Unhandled exception: {str(exc)} [{error_id}]")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=True,
            message="An unexpected error occurred",
            details=str(exc) if settings.DEBUG else None,
            error_id=error_id
        ).dict()
    )


# Include routers for different API endpoints
app.include_router(
    memory_router,
    prefix="/memory",
    tags=["Memory Operations"],
)

app.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["Memory Analytics"],
)

app.include_router(
    meta_cognition_router,
    prefix="/metacognition",
    tags=["Meta-Cognitive Operations"],
)


# Health check endpoint
@app.get("/health", response_model=StatusResponse, tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        JSON response with system status information
    """
    # Check memory system health
    memory_status = await memory_manager.health_check()
    
    return StatusResponse(
        status="ok",
        version=__version__,
        timestamp=int(time.time()),
        environment=settings.ENVIRONMENT,
        memory_system=memory_status
    ).dict()


# Root endpoint for API information
@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    """
    Root endpoint providing API information.
    
    Returns:
        JSON response with API information
    """
    return {
        "name": "Augment SDK API",
        "version": __version__,
        "description": "API for Augment SDK's hierarchical memory systems",
        "documentation": f"{request.base_url}docs" if settings.SHOW_DOCS else None
    }


# Memory layers information endpoint
@app.get("/memory-layers", tags=["System"])
async def memory_layers() -> Dict[str, Any]:
    """
    Get information about available memory layers in the system.
    
    Returns:
        JSON response with details about each memory layer
    """
    layers = {
        "ephemeral": {
            "description": "Short-term, temporary data (e.g., active chat context)",
            "ttl": "Session-based",
            "purpose": "Immediate context awareness"
        },
        "working": {
            "description": "Mid-term retention for project-focused tasks",
            "ttl": "Hours to days",
            "purpose": "Task completion and ongoing projects"
        },
        "semantic": {
            "description": "Long-term storage of facts, concepts, and structured knowledge",
            "ttl": "Persistent with decay",
            "purpose": "Knowledge base and conceptual understanding"
        },
        "procedural": {
            "description": "Step-by-step processes and workflows",
            "ttl": "Persistent",
            "purpose": "Reusable workflows and methods"
        },
        "reflective": {
            "description": "AI's self-analysis and past decisions",
            "ttl": "Persistent with prioritization",
            "purpose": "Self-improvement and learning"
        },
        "predictive": {
            "description": "Anticipation of future responses and needs",
            "ttl": "Dynamic based on accuracy",
            "purpose": "Proactive assistance and optimization"
        }
    }
    
    return {
        "available_layers": list(layers.keys()),
        "details": layers,
        "default_layer": "semantic"
    }


def start():
    """
    Function to start the Augment SDK API server directly.
    
    This is used when the module is run directly rather than through uvicorn.
    """
    # Set up logging
    setup_logging()
    
    # Load configuration from environment variables
    port = int(os.environ.get("AUGMENT_API_PORT", 8000))
    host = os.environ.get("AUGMENT_API_HOST", "0.0.0.0")
    reload = os.environ.get("AUGMENT_API_RELOAD", "false").lower() == "true"
    
    # Start the uvicorn server
    logger.info(f"Starting Augment SDK API server on {host}:{port}")
    uvicorn.run(
        "augment_sdk.api.main:app", 
        host=host, 
        port=port,
        reload=reload,
        log_level=os.environ.get("AUGMENT_API_LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    start()