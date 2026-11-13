
import asyncio
import time
import httpx

URL = "http://127.0.0.1:8001/chat/completions"


async def main():
    async with httpx.AsyncClient(timeout=1.0) as client:

        start = time.perf_counter()

        try:
            response = await client.post(
                URL,
                json={
                    "prompt": "Explain Transformer attention",
                    "delay_ms": 5000,
                    "mode": "normal"
                }
            )

            print("Response:", response.status_code)

        except httpx.ReadTimeout:
            print("Result: Client-side read timeout")

        except httpx.TimeoutException as error:
            print("Other timeout:", type(error).__name__)

        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"Client elapsed time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
