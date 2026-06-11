"""Pydantic schemas shared by the WebSocket protocol and the consumer API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── WebSocket protocol (orchestrator ↔ provider daemon) ─────────────────────

class Handshake(BaseModel):
    """First frame a daemon sends after receiving the server nonce."""

    type: Literal["handshake"] = "handshake"
    wallet_address: str
    signature: str                      # sig over f"grid-auth:{nonce}"
    models: list[str]                   # active local models
    max_concurrency: int = Field(ge=1, le=64, default=1)
    engine: Literal["ollama", "vllm", "aphrodite"] = "ollama"
    display_name: Optional[str] = None


class JobAssignment(BaseModel):
    """Orchestrator → daemon: run this inference."""

    type: Literal["job"] = "job"
    job_id: str
    model: str
    payload: dict[str, Any]             # sanitized OpenAI-style body
    deadline_seconds: int = 300


class JobResult(BaseModel):
    """Daemon → orchestrator: inference finished."""

    type: Literal["result"] = "result"
    job_id: str
    ok: bool
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


class Heartbeat(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    running_jobs: list[str] = []


# ─── Consumer API ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class CompletionResponse(BaseModel):
    job_id: str
    model: str
    output: dict[str, Any]
    served_by: str                      # provider DID
    tokens_in: int
    tokens_out: int


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
