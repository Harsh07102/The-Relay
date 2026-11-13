import asyncio
import time
import httpx

URL = "http://127.0.0.1:8001/chat/stream"


async def main():

    async with httpx.AsyncClient(timeout=20.0) as client:

        start = time.perf_counter()
        first_chunk_ms = None
        chunks_received = 0
        answer = ""

        async with client.stream(
            "POST",
            URL,
            json={
                "prompt": "Explain RAG",
                "delay_ms": 1000,
                "response_text": (
                    "Retrieval augmented generation "
                    "uses external knowledge to "
                    "improve language model answers."
                )
            }
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():

                if not line.startswith("data: "):
                    continue

                payload = line[6:]

                if payload == "[DONE]":
                    break

                import json
                data = json.loads(payload)

                content = (
                    data["choices"][0]["delta"].get("content", "")
                )

                if content:
                    if first_chunk_ms is None:
                        first_chunk_ms = (
                            time.perf_counter() - start
                        ) * 1000

                    chunks_received += 1
                    answer += content

        total_ms = (time.perf_counter() - start) * 1000

        print("\nSTREAMING RESULTS")
        print(f"First chunk time: {first_chunk_ms:.2f} ms")
        print(f"Total response time: {total_ms:.2f} ms")
        print(f"Chunks received: {chunks_received}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())