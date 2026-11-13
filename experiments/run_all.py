"""Runs the whole experiment set against the stand-in (or --backend groq for a small real run).
   python -m experiments.run_all --phase eval|load|all [--backend standin|groq]
Writes results/results.csv (one row per request) and results/results_detail.jsonl (latency traces)."""
import argparse, asyncio, csv, json, os, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

from app.config import Settings
from app.llm import LLMClient
from app.gateway import Gateway
from app.pipeline import Pipeline
from experiments.eval_questions import QUESTIONS, grade

RES = Path(__file__).resolve().parents[1] / "results"; RES.mkdir(exist_ok=True)
CSV, DETAIL = RES / "results.csv", RES / "results_detail.jsonl"
COLS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
        "tokens_in", "tokens_out", "retrieval_setting", "answer_grade", "degraded"]
SETTINGS_GRID = {"R1": (150, 30, 6, False), "R2": (350, 60, 4, False), "R3": (350, 60, 4, True), "R4": (600, 100, 3, True)}
STANDIN = "http://127.0.0.1:8099"
BASE = dict(rpm=300, tpm=80000, rpd=10000, tpd=2_000_000)     # stand-in profile ~10x the assumed free tier

# name: (n, rate_per_min or None for burst, server overrides, events [(t, cfg)], policy, client overrides)
SCEN = {
 "steady":            dict(n=20, rate=20, server={}, events=[], policy="managed"),
 "burst":             dict(n=40, rate=None, server={}, events=[], policy="managed"),
 "sustained_overload": dict(n=80, rate=80, server={}, events=[], policy="managed"),
 "rpm_halved":        dict(n=24, rate=24, server={}, events=[(10, dict(rpm=150, tpm=40000))], policy="managed"),
 "daily_exhausted":   dict(n=40, rate=40, server=dict(tpd=25000), events=[], policy="managed"),
 "overload_naive_retry": dict(n=80, rate=80, server={}, events=[], policy="naive"),
}


def start_server():
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "simulator.standin:app", "--port", "8099", "--log-level", "warning"],
                         cwd=str(RES.parent))
    for _ in range(50):
        try:
            if httpx.get(STANDIN + "/health", timeout=1).status_code == 200: return p
        except Exception: time.sleep(0.2)
    raise RuntimeError("stand-in did not start")


def row(scn, source, r, q, t_start):
    g = grade(q, r)
    return dict(request_id=r.rid, scenario=scn, source=source, start_time=t_start,
                first_token_ms=round(r.first_token_ms, 1) if r.first_token_ms is not None else "", total_ms=round(r.total_ms, 1),
                status=r.status, retries=r.retries, tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                retrieval_setting=r.setting, answer_grade=g, degraded=int(r.degraded))


async def run_scenario(name, spec, backend, setting_key, n=None, q_offset=0):
    n = n or spec["n"]
    if backend == "standin":
        async with httpx.AsyncClient() as h:
            await h.post(STANDIN + "/admin/reset")
            await h.post(STANDIN + "/admin/config", json={**BASE, **spec["server"]})
        s = Settings(backend="standin", model="stand-in", max_inflight=8, **BASE)
        s.tpd = BASE["tpd"]
    else:
        s = Settings(backend="groq")
    client = LLMClient(s); gw = Gateway(s, client, spec["policy"]); gw.start()
    pipe = Pipeline(s, gw, SETTINGS_GRID[setting_key])
    rng = random.Random(42)
    t_zero = time.monotonic(); rows, details = [], []
    offsets = [0.0] * n if spec["rate"] is None else [i * 60.0 / spec["rate"] + rng.uniform(0, 0.3) for i in range(n)]

    async def fire(i, off):
        await asyncio.sleep(max(0, off - (time.monotonic() - t_zero)))
        q = QUESTIONS[(i + q_offset) % len(QUESTIONS)]
        ts = datetime.now(timezone.utc).isoformat()
        r = await pipe.run(f"{name}-{i:03d}", q["question"])
        rows.append(row(name, "real" if backend == "groq" else "stand-in", r, q, ts))
        tr = r.trace
        details.append(dict(request_id=r.rid, scenario=name, level=r.level, degraded=r.degraded, error=r.error, status=r.status,
                            total_ms=round(r.total_ms), first_token_ms=round(r.first_token_ms or 0), wait_ms=round(tr.wait * 1000), work_ms=round(tr.work * 1000),
                            retry_ms=round(tr.retry * 1000), model_ttft_ms=round(tr.model_ttft or 0), events=tr.events, q=q["id"]))

    async def events():
        for t, cfg in spec["events"]:
            await asyncio.sleep(max(0, t - (time.monotonic() - t_zero)))
            async with httpx.AsyncClient() as h: await h.post(STANDIN + "/admin/config", json=cfg)

    await asyncio.gather(events(), *[fire(i, o) for i, o in enumerate(offsets)])
    dur = time.monotonic() - t_zero
    meta = dict(scenario=name, n=n, duration_s=round(dur, 1), offered_per_min=round(n / max(1e-9, (max(offsets) or 1)) * 60, 1) if spec["rate"] else "burst",
                peak_queue=gw.peak_queue, n429=gw.n429, calls=gw.ncalls)
    await gw.stop(); await client.close()
    return rows, details, meta


def append(rows, details):
    new = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, COLS + []); 
        if new: w.writeheader()
        w.writerows([{k: r[k] for k in COLS} for r in rows])
    with DETAIL.open("a") as f:
        for d in details: f.write(json.dumps(d) + "\n")


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="all"); ap.add_argument("--backend", default="standin")
    ap.add_argument("--fresh", action="store_true"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.fresh:
        for f in (CSV, DETAIL, RES / "meta.jsonl"): f.unlink(missing_ok=True)
    srv = start_server() if a.backend == "standin" else None
    try:
        metas = []
        if a.backend == "groq":      # protect the real daily budget: small, slow runs only
            SCEN.clear(); SCEN.update({"steady": dict(n=10, rate=6, server={}, events=[], policy="managed"),
                                       "burst": dict(n=8, rate=None, server={}, events=[], policy="managed")})
            SETTINGS_GRID_USE = {os.getenv("RELAY_SETTING", "R3"): SETTINGS_GRID[os.getenv("RELAY_SETTING", "R3")]}
        else: SETTINGS_GRID_USE = SETTINGS_GRID
        if a.phase in ("eval", "all"):
            for key in SETTINGS_GRID_USE:
                spec = dict(n=20, rate=20, server={}, events=[], policy="managed")
                rows, det, meta = await run_scenario(f"eval_{key}", spec, a.backend, key); meta["setting"] = key
                append(rows, det); metas.append(meta); print(meta, flush=True)
        if a.phase in ("load", "all"):
            key = os.getenv("RELAY_SETTING", "R3")
            for name, spec in SCEN.items():
                if a.only and name != a.only: continue
                rows, det, meta = await run_scenario(name, spec, a.backend, key); meta["setting"] = key
                append(rows, det); metas.append(meta); print(meta, flush=True)
        with (RES / "meta.jsonl").open("a") as f:
            for m in metas: f.write(json.dumps(m) + "\n")
    finally:
        if srv: srv.terminate()

if __name__ == "__main__":
    asyncio.run(main())
