"""Fit the stand-in from real measurements (results/real_probe.json + real rows in results.csv).
Writes results/calibration.json, which simulator/standin.py loads at start-up."""
import json, math, statistics as st
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
P = json.loads((RES / "real_probe.json").read_text())
ok = [(f, t, i, o) for f, t, i, o in P["latency_samples"] if f not in ("", None)]
n_all = len(P["latency_samples"])
ttft = sorted(f for f, *_ in ok); med = st.median(ttft)
logs = [math.log(x) for x in ttft]; sigma = st.pstdev(logs)
tps = [o / ((t - f) / 1000) for f, t, i, o in ok if t - f > 30 and o > 5]
cal = dict(ttft_median_ms=round(med, 1), ttft_sigma=round(sigma, 3), tps=round(st.median(tps), 1) if tps else 250.0,
           p_slow=round(sum(1 for x in ttft if x > 3 * med) / len(ttft), 3), p_error=round((n_all - len(ok)) / n_all, 3))
# per-prompt-token cost: only if the probe varied prompt size; otherwise keep prior
lim = P.get("limits", {}); b = lim.get("burst_first_429")
hdr = P.get("headers_ok") or {}
if hdr.get("x-ratelimit-limit-tokens"): cal["tpm"] = int(hdr["x-ratelimit-limit-tokens"])
if hdr.get("x-ratelimit-limit-requests"): cal["rpd"] = int(hdr["x-ratelimit-limit-requests"])
if b: cal["rpm"] = max(1, b["after_requests"])
(RES / "calibration.json").write_text(json.dumps(cal, indent=1)); print(cal)
