"""Stand-in model service (OpenAI/Groq-compatible /v1/chat/completions, SSE streaming).

It can: be slow (lognormal TTFT + per-prompt-token cost), slow down under load
(inflight beyond a knee), fail (5xx), hang, succeed slowly (tail multiplier), refuse
with 429 at RPM/TPM/RPD/TPD limits (with Retry-After and x-ratelimit-* headers),
and stream in pieces. Defaults are PRIORS; calibration.json (written by
experiments/calibrate.py from real Groq measurements) overrides them.

Its "answers" are extractive (best-matching sentences from the supplied passages,
with [S#] tags) so the pipeline's citation/verification logic is exercised for real.
It is NOT a language model: quality numbers measured against it say little about Groq.
"""
import asyncio, json, math, os, random, re, time, uuid
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CFG = dict(rpm=300, tpm=80000, rpd=10000, tpd=2_000_000, max_inflight=8, count_rejected=True,
           ttft_median_ms=350.0, ttft_sigma=0.45, ms_per_prompt_token=0.12,
           tps=250.0, tps_sigma=0.25, load_knee=4, load_alpha=0.8,
           p_slow=0.03, slow_mult=6.0, p_error=0.01, p_hang=0.005, hang_s=60.0,
           day_retry_after=1800, seed=1)
CAL = Path(__file__).resolve().parents[1] / "results" / "calibration.json"
if CAL.exists():
    CFG.update({k: v for k, v in json.loads(CAL.read_text()).items() if k in CFG})
RNG = random.Random(CFG["seed"])

app = FastAPI(title="Relay stand-in")
S = dict(req=deque(), tok=deque(), day_req=0, day_tok=0, inflight=0, peak=0, n=0, rej=0, err=0)
SEM = None

STOP = set("a an the of to in on for and or is are was were be it its this that with as by at from how what why which "
           "who when where does do did can could would will use used using about into than then their they".split())


def words(t):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\.\-]*", t.lower()) if w not in STOP and len(w) > 1]


def est_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.8))


# ----------------------------------------------------------------------------- extractive "model"
def respond(messages):
    sysm = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = messages[-1]["content"]
    task = (re.search(r"TASK:\s*(\w+)", sysm) or [None, "draft"])[1]
    if task == "understand":
        q = re.search(r"QUESTION:\s*(.*)", user, re.S)
        return json.dumps({"query": " ".join(words(q.group(1) if q else user))})
    if task == "verify":
        return "SUPPORTED"
    q = (re.search(r"QUESTION:\s*(.*?)\n", user + "\n") or [None, ""])[1]
    ps = re.findall(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\nQUESTION:|\Z)", user, re.S)
    qw = set(words(q))
    cands = []
    for sid, text in ps:
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
            sw = set(words(sent))
            if len(sent.split()) < 6 or not qw:
                continue
            ov = len(qw & sw) / len(qw)
            cands.append((ov, sid, " ".join(sent.split()[:45])))
    cands.sort(reverse=True)
    if not cands or cands[0][0] < 0.34:
        return "NOT_IN_PAPERS"
    out, seen = [], set()
    for ov, sid, sent in cands:
        if ov >= max(0.34, cands[0][0] * 0.8) and sent not in seen and len(out) < 2:
            out.append(f"{sent} [{sid}]"); seen.add(sent)
    return " ".join(out)


# ----------------------------------------------------------------------------- limits
def limit_check(now, need):
    for w in (S["req"],):
        while w and now - w[0] > 60: w.popleft()
    while S["tok"] and now - S["tok"][0][0] > 60: S["tok"].popleft()
    if S["day_req"] >= CFG["rpd"]:
        return "requests per day (RPD)", CFG["rpd"], S["day_req"], CFG["day_retry_after"]
    if S["day_tok"] + need > CFG["tpd"]:
        return "tokens per day (TPD)", CFG["tpd"], S["day_tok"], CFG["day_retry_after"]
    if len(S["req"]) >= CFG["rpm"]:
        return "requests per minute (RPM)", CFG["rpm"], len(S["req"]), max(0.2, S["req"][0] + 60 - now)
    used = sum(t for _, t in S["tok"])
    if used + need > CFG["tpm"]:
        acc, wait = used + need, 0.2
        for ts, t in S["tok"]:
            acc -= t; wait = ts + 60 - now
            if acc <= CFG["tpm"]: break
        return "tokens per minute (TPM)", CFG["tpm"], used, max(0.2, wait)
    return None


def rl_headers(now):
    used = sum(t for _, t in S["tok"])
    return {"x-ratelimit-limit-requests": str(CFG["rpd"]), "x-ratelimit-remaining-requests": str(max(0, CFG["rpd"] - S["day_req"])),
            "x-ratelimit-limit-tokens": str(CFG["tpm"]), "x-ratelimit-remaining-tokens": str(max(0, CFG["tpm"] - used)),
            "x-ratelimit-reset-tokens": "1.0s", "x-ratelimit-reset-requests": "60s"}


@app.get("/health")
async def health():
    return {"status": "ok", "cfg": CFG}


@app.post("/admin/config")
async def admin_config(req: Request):
    CFG.update({k: v for k, v in (await req.json()).items() if k in CFG}); RNG.seed(CFG["seed"])
    return CFG


@app.post("/admin/reset")
async def admin_reset():
    global SEM
    for k in ("req", "tok"): S[k].clear()
    S.update(day_req=0, day_tok=0, inflight=0, peak=0, n=0, rej=0, err=0)
    SEM = asyncio.Semaphore(CFG["max_inflight"])
    return {"ok": True}


@app.get("/admin/stats")
async def admin_stats():
    return {k: v for k, v in S.items() if k not in ("req", "tok")}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    global SEM
    if SEM is None: SEM = asyncio.Semaphore(CFG["max_inflight"])
    body = await req.json()
    now = time.time()
    msgs = body["messages"]; max_tokens = int(body.get("max_tokens") or 256)
    ptok = est_tokens(json.dumps(msgs)); need = ptok + max_tokens
    S["n"] += 1
    rej = limit_check(now, need)
    if rej:
        kind, lim, used, retry = rej
        S["rej"] += 1
        if CFG["count_rejected"] and "minute" in kind:
            S["req"].append(now)
        h = rl_headers(now); h["retry-after"] = str(int(math.ceil(retry)))
        msg = f"Rate limit reached for model on {kind}: Limit {lim}, Used {used}, Requested {need}. Please try again in {retry:.1f}s."
        return JSONResponse({"error": {"message": msg, "type": "tokens" if "token" in kind else "requests", "code": "rate_limit_exceeded"}},
                            status_code=429, headers=h)
    S["req"].append(now); S["tok"].append((now, need)); S["day_req"] += 1; S["day_tok"] += need
    if RNG.random() < CFG["p_error"]:
        await asyncio.sleep(0.05); S["err"] += 1
        return JSONResponse({"error": {"message": "internal error", "type": "server_error"}}, status_code=500, headers=rl_headers(now))
    hang = RNG.random() < CFG["p_hang"]
    slow = RNG.random() < CFG["p_slow"]
    text = respond(msgs)
    otok = est_tokens(text)
    headers = rl_headers(now)

    async def gen():
        async with SEM:
            S["inflight"] += 1; S["peak"] = max(S["peak"], S["inflight"])
            try:
                load = 1 + CFG["load_alpha"] * max(0, S["inflight"] - CFG["load_knee"]) / max(1, CFG["load_knee"])
                ttft = RNG.lognormvariate(math.log(CFG["ttft_median_ms"]), CFG["ttft_sigma"]) + ptok * CFG["ms_per_prompt_token"]
                ttft *= load * (CFG["slow_mult"] if slow else 1)
                await asyncio.sleep(CFG["hang_s"] if hang else ttft / 1000)
                words_ = text.split(" ")
                tps = max(20, RNG.lognormvariate(math.log(CFG["tps"]), CFG["tps_sigma"]) / load)
                step = max(1, len(words_) // 12)          # <=12 pieces per reply
                for i in range(0, len(words_), step):
                    piece = " ".join(words_[i:i + step]) + (" " if i + step < len(words_) else "")
                    yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]}) + "\n\n"
                    await asyncio.sleep(est_tokens(piece) / tps)
                S["day_tok"] += otok - max_tokens
                yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": otok, "total_tokens": ptok + otok}}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                S["inflight"] -= 1
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
