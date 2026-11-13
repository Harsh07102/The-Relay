"""Ask one question:  python main.py "How many attention heads does the base Transformer use?" [--backend standin|groq]
For --backend standin start the stand-in first:  uvicorn simulator.standin:app --port 8099"""
import argparse, asyncio
from app.config import Settings
from app.llm import LLMClient
from app.gateway import Gateway
from app.pipeline import Pipeline


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("question"); ap.add_argument("--backend", default=None)
    a = ap.parse_args(); s = Settings(backend=a.backend) if a.backend else Settings()
    c = LLMClient(s); gw = Gateway(s, c); gw.start()
    r = await Pipeline(s, gw, (350, 60, 4, True)).run("cli", a.question)
    print(r.answer); print(f"\n[{r.status} | first token {r.first_token_ms:.0f} ms | total {r.total_ms:.0f} ms | "
                           f"tokens {r.tokens_in}+{r.tokens_out} | degraded={r.degraded} | refused={r.refused}]")
    await gw.stop(); await c.close()

asyncio.run(main())
