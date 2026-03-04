"""
LLAISYS Chat Server — OpenAI-compatible Chat Completion API.

Usage:
    python -m server.app --model /path/to/model [--host 0.0.0.0] [--port 8000]

Or from the project root:
    cd python && uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
import uuid
from typing import AsyncGenerator

# Ensure the llaisys package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessageResponse,
    ChoiceDelta,
    UsageInfo,
)

import llaisys
from llaisys.libllaisys import DeviceType

# ── Globals (initialised in startup) ─────────────────────────────────

app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: llaisys.models.Qwen2 | None = None
TOKENIZER = None
MODEL_PATH: str = ""
DEVICE: DeviceType = DeviceType.CPU

# ── Model Management ─────────────────────────────────────────────────

def load_model(model_path: str, device: str = "cpu") -> None:
    """Load model and tokenizer (called once at startup)."""
    global MODEL, TOKENIZER, MODEL_PATH, DEVICE

    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    # Resolve model path
    if os.path.isdir(model_path):
        resolved_path = model_path
    else:
        print(f"Downloading model {model_path} ...")
        resolved_path = snapshot_download(model_path)

    MODEL_PATH = resolved_path
    DEVICE = DeviceType.NVIDIA if device == "nvidia" else DeviceType.CPU

    print(f"Loading tokenizer from {resolved_path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        trust_remote_code=True,
    )

    print(f"Loading LLAISYS model from {resolved_path} (device={device}) ...")
    MODEL = llaisys.models.Qwen2(resolved_path, DEVICE)
    print("Model ready.")


def _reset_model() -> None:
    """Reset KV-cache position (no weight reload needed)."""
    MODEL.reset_cache()


# ── Helpers ──────────────────────────────────────────────────────────

def _encode_messages(messages: list) -> list[int]:
    """Apply chat template and tokenize."""
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    text = TOKENIZER.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    return TOKENIZER.encode(text)


# ── Non-streaming endpoint ───────────────────────────────────────────

def _generate_full(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate a complete response (blocking)."""
    _reset_model()

    input_ids = _encode_messages(request.messages)
    prompt_tokens = len(input_ids)

    output_ids = MODEL.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    )

    # Decode only the generated tokens
    new_tokens = output_ids[prompt_tokens:]
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ── Streaming endpoint ───────────────────────────────────────────────

async def _generate_stream(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Yield SSE chunks, one per generated token."""
    _reset_model()

    input_ids = _encode_messages(request.messages)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Send initial role chunk
    initial_chunk = ChatCompletionStreamResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChoiceDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream tokens
    for token_id in MODEL.generate_stream(
        input_ids,
        max_new_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    ):
        if token_id == 151643:  # EOS
            break

        text = TOKENIZER.decode([token_id], skip_special_tokens=True)
        if not text:
            continue

        chunk = ChatCompletionStreamResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChoiceDelta(content=text),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ───────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        return StreamingResponse(
            _generate_stream(request),
            media_type="text/event-stream",
        )

    return _generate_full(request)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-r1-distill-qwen-1.5b",
                "object": "model",
                "owned_by": "llaisys",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


# ── Static files (Web UI) ───────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chat Server")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model path or HuggingFace repo id")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model, args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
