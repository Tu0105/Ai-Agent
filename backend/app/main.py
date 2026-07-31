"""FastAPI application entrypoint with bounded, validated public APIs."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .agent import evict_agent, get_or_create_agent
from .memory import get_memory_manager
from .schemas import (
    CONVERSATION_ID_PATTERN,
    MAX_CONVERSATION_ID_LENGTH,
    MAX_MESSAGE_LENGTH,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
)


def cors_allow_origins() -> list[str]:
    """Return a comma-delimited origin allowlist; wildcards are never accepted."""
    configured = os.getenv("CORS_ALLOW_ORIGINS")
    candidates = configured.split(",") if configured else DEFAULT_CORS_ORIGINS
    origins: list[str] = []
    for raw_origin in candidates:
        origin = raw_origin.strip().rstrip("/")
        parsed = urlparse(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            logger.warning("Ignoring invalid CORS origin configuration")
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def require_safe_conversation_id(conversation_id: str) -> str:
    if (
        len(conversation_id) > MAX_CONVERSATION_ID_LENGTH
        or not CONVERSATION_ID_PATTERN.fullmatch(conversation_id)
    ):
        raise HTTPException(status_code=422, detail="Invalid conversation ID.")
    return conversation_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("AI Agent service started")
    yield
    logger.info("AI Agent service stopped")


app = FastAPI(
    title="AI Agent Customer Service",
    description="A bounded FastAPI service for an AI customer-service agent.",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", version="1.1.0", timestamp=datetime.now())


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        agent = get_or_create_agent(conversation_id)
        result = await asyncio.to_thread(agent.invoke, request.message)
        if not result.get("success"):
            logger.warning("Agent invocation failed for conversation %s", conversation_id)
            raise HTTPException(status_code=502, detail="Unable to process the message.")
        return ChatResponse(
            message=result["message"],
            conversation_id=result["conversation_id"],
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected chat failure for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Internal server error.")


async def stream_chat_response(message: str, conversation_id: str):
    try:
        agent = get_or_create_agent(conversation_id)
        yield {"event": "conversation", "data": conversation_id}
        # The agent's synchronous streaming API is intentionally not accumulated in memory.
        for chunk in agent.stream(message):
            if chunk:
                yield {"event": "message", "data": chunk}
                await asyncio.sleep(0)
        yield {"event": "done", "data": "stream_complete"}
    except Exception:
        logger.exception("Streaming chat failed for conversation %s", conversation_id)
        yield {"event": "error", "data": "Unable to process the message."}


@app.get("/api/chat/stream", tags=["chat"])
async def chat_stream(
    message: str = Query(..., min_length=1, max_length=MAX_MESSAGE_LENGTH),
    conversation_id: Optional[str] = Query(default=None, max_length=MAX_CONVERSATION_ID_LENGTH),
):
    if not message.strip():
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    safe_conversation_id = require_safe_conversation_id(conversation_id) if conversation_id else str(uuid.uuid4())
    return EventSourceResponse(stream_chat_response(message, safe_conversation_id))


@app.post("/api/chat/stream", tags=["chat"])
async def chat_stream_post(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    return EventSourceResponse(stream_chat_response(request.message, conversation_id))


@app.get("/api/history/{conversation_id}", tags=["history"])
async def get_history(
    conversation_id: str = Path(..., max_length=MAX_CONVERSATION_ID_LENGTH),
):
    safe_conversation_id = require_safe_conversation_id(conversation_id)
    try:
        history = get_memory_manager().get_history(safe_conversation_id)
        return {"conversation_id": safe_conversation_id, "messages": history, "count": len(history)}
    except Exception:
        logger.exception("History lookup failed for conversation %s", safe_conversation_id)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.delete("/api/history/{conversation_id}", tags=["history"])
async def clear_history(
    conversation_id: str = Path(..., max_length=MAX_CONVERSATION_ID_LENGTH),
):
    safe_conversation_id = require_safe_conversation_id(conversation_id)
    try:
        success = get_memory_manager().clear_memory(safe_conversation_id)
        evict_agent(safe_conversation_id)
        return {
            "success": success,
            "conversation_id": safe_conversation_id,
            "message": "Conversation history cleared." if success else "Conversation history does not exist.",
        }
    except Exception:
        logger.exception("History cleanup failed for conversation %s", safe_conversation_id)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=str(exc.detail), timestamp=datetime.now()).model_dump(mode="json"),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    logger.info("Request validation failed: %s error(s)", len(exc.errors()))
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Invalid request.", timestamp=datetime.now()).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled application error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error.", timestamp=datetime.now()).model_dump(mode="json"),
    )


@app.get("/", tags=["root"])
async def root():
    return {"message": "AI Agent Customer Service", "version": "1.1.0", "docs": "/docs", "health": "/health"}
