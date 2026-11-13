"""Every number in the report is computed here from results/results.csv (+ detail/meta for queue length)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from app.config import SETTINGS as S

RES = Path(__file__).resolve().parents[1] / "results"


def load():
    df = pd.read_csv(RES / "results.csv")
    det = [json.loads(l) for l in (RES / "results_detail.jsonl").read_text().splitlines()] if (RES / "results_detail.jsonl").exists() else []
    meta = {}
    if (RES / "meta.jsonl").exists():
        for l in (RES / "meta.jsonl").read_text().splitlines():
            m = json.loads(l); meta[m["scenario"]] = m
    return df, pd.DataFrame(det), meta


def pct(x, qs=(50, 95, 99)):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return [float(np.percentile(x, q)) for q in qs] if len(x) else [float("nan")] * len(qs)


def f0(v): return "n/a" if v != v else f"{v:,.0f}"


def table2(df, meta, scenarios):
    a, b = [], []
    for sc in scenarios:
        d = df[df.scenario == sc]
        if d.empty: a.append([sc, "not measured"] + ["not measured"] * 6); b.append([sc] + ["not measured"] * 7); continue
        m = meta.get(sc, {}); n = len(d)
        src = ",".join(sorted(d.source.unique()))
        ft, tt = pct(d.first_token_ms), pct(d.total_ms)
        a.append([sc, src, n, m.get("duration_s", "n/a"), m.get("offered_per_min", "n/a"),
                  " / ".join(f0(x) for x in ft), " / ".join(f0(x) for x in tt)])
        tok_ans = (d.tokens_in + d.tokens_out).mean()
        cost = (d.tokens_in.mean() * S.price_in + d.tokens_out.mean() * S.price_out) / 1e6 * 1000
        b.append([sc, f"{100*(d.status=='ok').mean():.1f}", f"{100*(d.status=='rate_limited').mean():.1f}",
                  f"{100*(d.status.isin(['timeout','error'])).mean():.1f}", f"{d.retries.mean():.2f}", m.get("peak_queue", "n/a"),
                  f"{tok_ans:,.0f}", f"${cost:.3f}"])
    return a, b


def table3(df, keys):
    rows = []
    for k in keys:
        d = df[df.scenario == f"eval_{k}"]
        if d.empty: rows.append([k] + ["not measured"] * 7); continue
        g = d.answer_grade.value_counts(); n = len(d)
        ans = d[d.answer_grade.isin(["correct", "wrong", "refused_wrongly"])]
        ft, tt = pct(d.first_token_ms, (50, 95)), pct(d.total_ms, (50, 95))
        rows.append([d.retrieval_setting.iloc[0], n, f"{100*g.get('correct',0)/n:.1f}", f"{100*g.get('wrong',0)/n:.1f}",
                     f"{100*g.get('refused_correctly',0)/n:.1f}", f"{(d.tokens_in+d.tokens_out).mean():,.0f}",
                     " / ".join(f0(x) for x in ft), " / ".join(f0(x) for x in tt)])
    return rows
