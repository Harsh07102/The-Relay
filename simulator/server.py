import json
from fastapi.responses import StreamingResponse
import asyncio
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="The Relay - Stand-in Model Service")

# Simulation configuration
MAX_CONCURRENCY = 2
OVERLOAD_PENALTY_MS = 300

# Shared state for this single-process simulator
request_slots = asyncio.Semaphore(MAX_CONCURRENCY)
active_requests = 0
peak_active_requests = 0


class ChatRequest(BaseModel):
    prompt: str
    delay_ms: int = Field(default=0, ge=0, le=10000)
    mode: str = "normal"
    response_text: str = "This is a simulated model response."


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "relay-stand-in",
        "max_concurrency": MAX_CONCURRENCY
    }


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    global active_requests, peak_active_requests

    request_id = str(uuid.uuid4())
    request_start = time.perf_counter()

    # Failure injection
    if request.mode == "rate_limit":
        raise HTTPException(
            status_code=429,
            detail="Simulated rate limit exceeded",
            headers={"Retry-After": "5"}
        )

    if request.mode == "server_error":
        raise HTTPException(
            status_code=500,
            detail="Simulated internal server error"
        )

    if request.mode != "normal":
        raise HTTPException(
            status_code=400,
            detail="Unknown simulation mode"
        )

    # Wait until a concurrency slot becomes available
    await request_slots.acquire()

    queue_wait_ms = (
        time.perf_counter() - request_start
    ) * 1000

    active_requests += 1
    peak_active_requests = max(
        peak_active_requests, active_requests
    )

    active_at_start = active_requests

    # Simulate additional latency when service is busy
    overload_penalty = (
        OVERLOAD_PENALTY_MS
        if active_at_start > 1
        else 0
    )

    processing_start = time.perf_counter()

    try:
        total_delay_ms = request.delay_ms + overload_penalty

        await asyncio.sleep(total_delay_ms / 1000)

        processing_ms = (
            time.perf_counter() - processing_start
        ) * 1000

    finally:
        active_requests -= 1
        request_slots.release()

    server_elapsed_ms = (
        time.perf_counter() - request_start
    ) * 1000

    return {
        "id": request_id,
        "object": "chat.completion",
        "simulated": True,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": request.response_text
                },
                "finish_reason": "stop"
            }
        ],
        "simulation_metrics": {
            "configured_delay_ms": request.delay_ms,
            "queue_wait_ms": round(queue_wait_ms, 2),
            "active_at_start": active_at_start,
            "overload_penalty_ms": overload_penalty,
            "processing_ms": round(processing_ms, 2),
            "server_elapsed_ms": round(server_elapsed_ms, 2),
            "peak_active_requests": peak_active_requests
        }
    }
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    async def generate():
        global active_requests, peak_active_requests

        await request_slots.acquire()

        active_requests += 1
        peak_active_requests = max(
            peak_active_requests, active_requests
        )

        try:
            overload_penalty = (
                OVERLOAD_PENALTY_MS
                if active_requests > 1
                else 0
            )

            # Simulate time before the first output
            first_delay = request.delay_ms + overload_penalty
            await asyncio.sleep(first_delay / 1000)

            words = request.response_text.split()

            for index, word in enumerate(words):
                chunk = {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": word + " "
                        }
                    }]
                }

                yield f"data: {json.dumps(chunk)}\n\n"

                # Simulate time between generated chunks
                if index < len(words) - 1:
                    await asyncio.sleep(0.1)

            yield "data: [DONE]\n\n"

        finally:
            active_requests -= 1
            request_slots.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
