
import asyncio
import time
import httpx

URL = "http://127.0.0.1:8001/chat/completions"
NUMBER_OF_REQUESTS = 6


async def send_request(client, request_number):
    start = time.perf_counter()

    response = await client.post(
        URL,
        json={
            "prompt": f"Test request {request_number}",
            "delay_ms": 1000,
            "mode": "normal"
        }
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()

    data = response.json()

    return {
        "request": request_number,
        "client_elapsed_ms": round(elapsed_ms, 2),
        "server_metrics": data["simulation_metrics"]
    }


async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:

        start = time.perf_counter()

        tasks = [
            send_request(client, i)
            for i in range(1, NUMBER_OF_REQUESTS + 1)
        ]

        results = await asyncio.gather(*tasks)

        total_elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        print("\nINDIVIDUAL REQUEST RESULTS")

        for result in results:
            print(result)

        print("\nTOTAL EXPERIMENT TIME:")
        print(f"{total_elapsed_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
