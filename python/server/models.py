"""Pydantic data models following the OpenAI Chat Completion API format."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r1-distill-qwen-1.5b"
    messages: List[ChatMessage]
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stream: bool = False
    # Extension: session management (for Phase 4)
    session_id: Optional[str] = None


# ── Response ─────────────────────────────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: str = ""


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessageResponse
    finish_reason: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "deepseek-r1-distill-qwen-1.5b"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "deepseek-r1-distill-qwen-1.5b"
    choices: List[ChatCompletionStreamChoice]
