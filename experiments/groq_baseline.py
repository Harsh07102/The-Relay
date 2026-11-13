
import asyncio
import json
import os
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

# Load credentials and configuration from .env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

URL = "https://api.groq.com/openai/v1/chat/completions"

NUMBER_OF_REQUESTS = 5
MAX_OUTPUT_TOKENS = 250
REQUEST_INTERVAL_SECONDS = 2

PROMPT = (
    "Explain retrieval augmented generation (RAG) "
    "in simple terms. Describe its two main stages "
    "and give one practical example. Keep your answer "
    "within 100 words."
)


async def run_one(client, request_number):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()

    first_token_ms = None
    finish_reason = None
    answer = ""
    usage = None

    print(f"\nStarting request {request_number}...")

    try:
        async with client.stream(
            "POST",
            URL,
            headers=headers,
            json=payload
        ) as response:

            status_code = response.status_code

            # Capture rate-limit headers
            rate_headers = {
                key: value
                for key, value in response.headers.items()
                if "ratelimit" in key.lower()
                or "retry-after" in key.lower()
            }

            # Handle unsuccessful responses
            if status_code != 200:

                body = await response.aread()

                total_ms = (
                    time.perf_counter() - start
                ) * 1000

                result = {
                    "request": request_number,
                    "status": status_code,
                    "first_token_ms": None,
                    "total_ms": round(total_ms, 2),
                    "rate_headers": rate_headers,
                    "error": body.decode(
                        "utf-8",
                        errors="replace"
                    ),
                }

                print(result)
                return result

            # Read streamed response
            async for line in response.aiter_lines():

                if not line.startswith("data: "):
                    continue

                raw = line[6:]

                if raw == "[DONE]":
                    break

                event = json.loads(raw)

                # Usage information may arrive separately
                if event.get("usage"):
                    usage = event["usage"]

                choices = event.get("choices", [])

                if choices:

                    choice = choices[0]

                    # Capture why generation stopped
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    content = (
                        choice.get("delta", {}).get("content")
                    )

                    if content:

                        # Time until first generated text
                        if first_token_ms is None:
                            first_token_ms = (
                                time.perf_counter() - start
                            ) * 1000

                        answer += content

            total_ms = (
                time.perf_counter() - start
            ) * 1000

            result = {
                "request": request_number,
                "status": status_code,
                "first_token_ms": (
                    round(first_token_ms, 2)
                    if first_token_ms is not None
                    else None
                ),
                "total_ms": round(total_ms, 2),
                "tokens_in": (
                    usage.get("prompt_tokens")
                    if usage else None
                ),
                "tokens_out": (
                    usage.get("completion_tokens")
                    if usage else None
                ),
                "total_tokens": (
                    usage.get("total_tokens")
                    if usage else None
                ),
                "finish_reason": finish_reason,
                "rate_headers": rate_headers,
                "answer": answer.strip(),
            }

            print(result)
            return result

    except httpx.TimeoutException:

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        result = {
            "request": request_number,
            "status": "timeout",
            "first_token_ms": (
                round(first_token_ms, 2)
                if first_token_ms is not None
                else None
            ),
            "total_ms": round(total_ms, 2),
        }

        print(result)
        return result

    except Exception as exc:

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        result = {
            "request": request_number,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "first_token_ms": (
                round(first_token_ms, 2)
                if first_token_ms is not None
                else None
            ),
            "total_ms": round(total_ms, 2),
        }

        print(result)
        return result


async def main():

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY missing. Check your .env file."
        )

    print("=" * 55)
    print("THE RELAY - REAL GROQ BASELINE EXPERIMENT")
    print("=" * 55)

    print("Model:", MODEL)
    print("Requests:", NUMBER_OF_REQUESTS)
    print("Maximum output tokens:", MAX_OUTPUT_TOKENS)
    print(
        "Start time:",
        datetime.now(timezone.utc).isoformat()
    )

    timeout = httpx.Timeout(
        60.0,
        connect=10.0
    )

    async with httpx.AsyncClient(timeout=timeout) as client:

        results = []

        for i in range(1, NUMBER_OF_REQUESTS + 1):

            result = await run_one(client, i)
            results.append(result)

            if i < NUMBER_OF_REQUESTS:
                await asyncio.sleep(
                    REQUEST_INTERVAL_SECONDS
                )

    print("\n" + "=" * 55)
    print("BASELINE SUMMARY")
    print("=" * 55)

    for result in results:
        print(result)

    successful = [
        r for r in results
        if r.get("status") == 200
    ]

    print("\nSUMMARY STATISTICS")
    print("Successful requests:", len(successful))
    print(
        "Failed requests:",
        len(results) - len(successful)
    )

    if successful:

        ttft_values = [
            r["first_token_ms"]
            for r in successful
            if r.get("first_token_ms") is not None
        ]

        total_values = [
            r["total_ms"]
            for r in successful
            if r.get("total_ms") is not None
        ]

        if ttft_values:
            print(
                "Average TTFT:",
                f"{sum(ttft_values) / len(ttft_values):.2f} ms"
            )

        if total_values:
            print(
                "Average total time:",
                f"{sum(total_values) / len(total_values):.2f} ms"
            )

    print("\nExperiment completed.")


if __name__ == "__main__":
    asyncio.run(main())
