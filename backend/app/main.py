"""
FastAPI application entry point for the Clinical Trial Site Activation
Response Intelligence POC.

Run from the project root:
    uvicorn main:app --reload

Expected project layout:
    main.py
    src/
      __init__.py
      schemas.py
      Site_Activation_Router.py
      Site_Activation_Service.py
      Site_Activation_Implementation.py
      Site_Activation_Orchestrator.py
      Document_Intelligence_Agent.py
      Site_Response_Intelligence_Agent.py
      Action_Prioritization_Followup_Agent.py
      llm_config.py

This POC uses synthetic data. It does not activate clinical trial sites or
send external emails. Human review remains required.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.Site_Activation_Router import router as site_activation_router
from app.observability.tracing_langsmith import tracing_is_configured

from app.config.settings import settings
logger = settings.get_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown hooks."""
    logger.info(
            "Starting Clinical Trial Site Activation Intelligence API | langsmith_tracing_configured=%s",
            tracing_is_configured(),
        )
    try:
        # Add lightweight startup checks here if required.
        # Avoid initializing external services that are not configured.
        yield
    finally:
        logger.info("Stopping Clinical Trial Site Activation Intelligence API")


app = FastAPI(
    title="Clinical Trial Site Activation & Response Intelligence",
    description=(
        "POC API for identifying site activation document gaps, "
        "classifying site responses, and prioritizing follow-up actions. "
        "Outputs require human review and do not authorize site activation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS only for explicitly configured origins. For local development,
# set CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000 if needed.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {
        "application": "Clinical Trial Site Activation & Response Intelligence",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "healthy"}


app.include_router(site_activation_router)