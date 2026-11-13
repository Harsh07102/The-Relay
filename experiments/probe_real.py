"""Probe the REAL Groq service (run on your machine; needs GROQ_API_KEY in .env).
Uses ~150 requests of your daily budget. Writes:
  results/real_probe.json   latency samples, discovered limits, 429 bodies/headers
  results/results.csv       rows with scenario=real_probe_*, source=real
Phases: A latency (30 sequential), B concurrency ramp (1,2,4,8 at once), C per-minute burst until 429,
        D token-per-minute test (big prompts until 429).  Re-run calibrate.py afterwards."""
import asyncio, csv, json, time
from datetime import datetime, timezone
from pathlib import Path
from app.config import SETTINGS as S
from app.llm import LLMClient, RateLimited, ServerError, CallTimeout
from experiments.run_all import CSV, COLS, RES

PROMPT = "Explain in two sentences what a transformer's self-attention does. " * 1
BIG = "Summarise: " + ("The quick brown fox jumps over the lazy dog. " * 200)    # ~2000 tokens


async def one(c, rid, scn, content, max_tokens=60):
    ts = datetime.now(timezone.utc).isoformat(); st = "ok"; ttft = tot = None; tin = tout = 0; h = {}; body = ""
    t0 = time.perf_counter()
    try:
        r = await c.chat([{"role": "user", "content": content}], max_tokens, 30)
        ttft, tot, tin, tout, h = r.ttft_ms, r.total_ms, r.tokens_in, r.tokens_out, r.headers
    except RateLimited as e: st, h, body = "rate_limited", e.headers, e.body; tot = (time.perf_counter() - t0) * 1000
    except CallTimeout: st = "timeout"; tot = (time.perf_counter() - t0) * 1000
    except ServerError as e: st = "error"; body = str(e); tot = (time.perf_counter() - t0) * 1000
    return dict(request_id=rid, scenario=scn, source="real", start_time=ts, first_token_ms=round(ttft, 1) if ttft else "",
                total_ms=round(tot, 1), status=st, retries=0, tokens_in=tin, tokens_out=tout, retrieval_setting="n/a",
                answer_grade="", degraded=0), h, body


async def main():
    assert S.groq_key, "GROQ_API_KEY missing in .env"
    c = LLMClient(S); rows, out = [], dict(model=S.model, started=datetime.now(timezone.utc).isoformat(), headers_ok=None, limits={})
    print("A: latency"); lat = []
    for i in range(30):
        r, h, b = await one(c, f"probeA-{i:02d}", "real_probe_latency", PROMPT); rows.append(r); lat.append(r)
        if r["status"] == "ok" and out["headers_ok"] is None: out["headers_ok"] = h
        await asyncio.sleep(2.5)
    out["latency_samples"] = [(r["first_token_ms"], r["total_ms"], r["tokens_in"], r["tokens_out"]) for r in lat]
    print("B: concurrency ramp"); out["concurrency"] = {}
    for n in (1, 2, 4, 8):
        res = await asyncio.gather(*[one(c, f"probeB-{n}-{i}", "real_probe_concurrency", PROMPT) for i in range(n)])
        rows += [r for r, _, _ in res]; out["concurrency"][n] = [(r["status"], r["total_ms"]) for r, _, _ in res]
        await asyncio.sleep(15)
    print("C: burst until 429 (tiny requests)"); first429 = None; seen = 0; burst = []
    t0 = time.monotonic()
    for i in range(60):
        r, h, b = await one(c, f"probeC-{i:02d}", "real_probe_burst", "Say OK.", 5); rows.append(r); burst.append(r["status"]); seen += 1
        if r["status"] == "rate_limited" and first429 is None:
            first429 = dict(after_requests=seen, after_s=round(time.monotonic() - t0, 1), headers=h, body=b[:400]); break
    out["limits"]["burst_first_429"] = first429
    if first429:
        ra = float(first429["headers"].get("retry-after", 5)); out["limits"]["retry_after_s"] = ra
        t1 = time.monotonic(); await asyncio.sleep(ra)
        while True:
            r, h, b = await one(c, f"probeC-rec-{int(time.monotonic()-t1)}", "real_probe_burst", "Say OK.", 5); rows.append(r)
            if r["status"] == "ok": out["limits"]["recovered_after_s"] = round(time.monotonic() - t1, 1); break
            await asyncio.sleep(1)
    await asyncio.sleep(65)
    print("D: tokens-per-minute (big prompts)"); first = None
    for i in range(12):
        r, h, b = await one(c, f"probeD-{i:02d}", "real_probe_tokens", BIG, 20); rows.append(r)
        if r["status"] == "rate_limited": first = dict(after_requests=i + 1, headers=h, body=b[:400]); break
    out["limits"]["token_first_429"] = first
    (RES / "real_probe.json").write_text(json.dumps(out, indent=1, default=str))
    new = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new: w.writeheader()
        w.writerows(rows)
    await c.close(); print("wrote results/real_probe.json; now run: python -m experiments.calibrate")

if __name__ == "__main__": asyncio.run(main())
