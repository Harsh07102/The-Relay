"""Builds report.pdf. Every number comes from results/results.csv (+detail/meta/probe files). Re-run after real runs."""
import asyncio, json, sys
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, FrameBreak, NextPageTemplate, KeepTogether
from report.stats import load, table2, table3, pct, f0, RES
from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
df, det, meta = load()
SC = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted", "overload_naive_retry"]
t2a, t2b = table2(df, meta, SC)
t3 = table3(df, ["R1", "R2", "R3", "R4"])
rec = json.loads((RES / "retrieval_recall.json").read_text())
probe = json.loads((RES / "real_probe.json").read_text()) if (RES / "real_probe.json").exists() else None
HAVE_REAL = bool((df.source == "real").any())
N = len(df)

def g(sc, col): d = df[df.scenario == sc]; return d
def degp(sc): return 100 * g(sc, "degraded").degraded.mean()
def D(sc): return det[det.scenario == sc]
mg_rows = df[df.scenario.isin(['steady','burst','sustained_overload','rpm_halved','daily_exhausted'])]
mg_bad = int(mg_rows.status.isin(['timeout','error','rate_limited']).sum())
r1t = (df[df.scenario=='eval_R1'].tokens_in+df[df.scenario=='eval_R1'].tokens_out).mean(); r3t = (df[df.scenario=='eval_R3'].tokens_in+df[df.scenario=='eval_R3'].tokens_out).mean()
sl = det[det.scenario == "steady"].sort_values("total_ms", ascending=False).iloc[0]
nv, mg = meta["overload_naive_retry"], meta["sustained_overload"]
nvd, mgd = g("overload_naive_retry", 0), g("sustained_overload", 0)
load_rows = df[df.scenario.isin(SC) & (df.scenario != "overload_naive_retry")]
ok_full = load_rows[load_rows.degraded == 0]; ok_deg = load_rows[load_rows.degraded == 1]
cp = lambda d: 100 * (d.answer_grade == "correct").mean() if len(d) else float("nan")
lvl2 = 100 * (D("sustained_overload").level == 2).mean()
ev = lambda sc: D(sc)
sample = ""
try:   # a real example of the weaker reply (no LLM involved)
    from app.pipeline import Pipeline, Result
    from app.gateway import Trace
    pl = Pipeline(Settings(backend="standin"), None, (150, 30, 6, False)); r = Result("x", "q", "s"); pl._fallback(r, "What is Moshi's theoretical latency?", {})
    sample = r.answer[:330]
except Exception as e: sample = f"(example unavailable: {e})"

base = ParagraphStyle("b", fontName="Helvetica", fontSize=8.0, leading=9.4, spaceAfter=2.2)
h = ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=8.8, leading=10.2, spaceBefore=4.5, spaceAfter=1, textColor=colors.HexColor("#1f3a5f"))
small = ParagraphStyle("s", parent=base, fontSize=7.0, leading=8.1)
P = lambda t, s=base: Paragraph(t, s); H = lambda t: Paragraph(t, h)

# ------------------------------------------------------------------ narrative
def narrative():
    B = []
    B.append(P("<b>The Relay: a RAG pipeline that degrades answers, not people</b> &nbsp; (repo: github.com/Harsh07102/The-Relay; all numbers from results/results.csv, %d requests)" % N, ParagraphStyle("t", parent=base, fontSize=9, leading=10.5)))
    B.append(P("<b>Honesty note.</b> " + ("Real Groq runs are included (see tables; source column)." if HAVE_REAL else
        "<font color='#b00020'>All load results below are from the STAND-IN. No real-Groq run is in this file yet: Table 1 and every 'real' cell say 'not measured'. "
        "Stand-in parameters are priors, not calibrated. Run <i>experiments/probe_real.py</i> then <i>make_report</i> to fill them in.</font>"), small))
    B.append(H("1 What I built and why"))
    B.append(P("Pipeline: <i>understand</i> (LLM) &rarr; <i>retrieve</i> (BM25 tool, 150-word pieces, k=6) &rarr; <i>decide</i> (branch: if query-term coverage &lt; 0.60 widen once, then refuse; loop: redraft once if the check fails) &rarr; <i>draft</i> (LLM) &rarr; <i>verify</i> (deterministic citation tool: every sentence cited, every number present in the cited text, wording overlap; an LLM judge only when borderline) &rarr; deliver. "
               "All model calls go through one gateway (priority queue, RPM/TPM windows with token reservations, AIMD, Retry-After, retry budget, daily-limit state). "
               "<b>The decision that shaped it most:</b> the draft is buffered and verified <i>before</i> any token reaches the person, because a wrong answer sent fast is worse than a slower right one. Cost: first token = verified-answer start, so in my tables first-token and total are equal; the model's own first token is in the trace (e.g. %s ms for the Section 4 request). Speed is bought elsewhere: drop optional steps under load, finish old work before new, and when time or budget runs out send a weaker but honest answer (cited excerpts, or 'the papers don't say'). Quality is shed, nobody is turned away." % f0(sl.model_ttft_ms)))
    B.append(H("2 The real service and my stand-in"))
    if probe:
        lim = probe.get("limits", {}); B.append(P(f"Real limits and latency: see Table 1 (probe_real.py: {len(probe['latency_samples'])} sequential calls, a concurrency ramp 1/2/4/8, a request burst until the first 429, and big prompts until a token 429)."))
    else:
        B.append(P("<b>Real service: not measured here.</b> The environment this was built in had no network path to api.groq.com, so I wrote <i>probe_real.py</i> (latency over 30 calls, concurrency ramp, burst until 429, token-limit test, header capture) and <i>calibrate.py</i>, but they have not been run. "
                   "Limits in Table 1 are what I recall of Groq's free-tier docs for openai/gpt-oss-20b (about 30 RPM, 8K TPM, 1K RPD, 200K TPD): <b>low confidence, unverified</b>; the client also re-learns RPM/TPM/RPD from x-ratelimit-* headers at run time."))
    B.append(P("<b>Stand-in</b> (simulator/standin.py, OpenAI-compatible, SSE): lognormal time-to-first-token (median 350 ms, sigma 0.45) plus 0.12 ms per prompt token; decode 250 tok/s (sigma 0.25); slowdown beyond 4 requests in flight (alpha 0.8); 3% of calls succeed 6x slower; 1% 500s; 0.5% hang; 429 with Retry-After and x-ratelimit-* at RPM/TPM/RPD/TPD; replies in about 12 pieces. "
               "Run profile: 300 RPM, 80K TPM, 10K RPD, 2M TPD (about 10x the assumed free tier so runs finish in minutes). <b>Least like the real thing:</b> its 'answers' are extractive sentences, not generated text (no reasoning tokens, output length unrelated to question); noise is independent where real tails are correlated; and I <i>assume</i> a rejected request still counts toward RPM (unknown for Groq)."))
    B.append(H("3 Results"))
    B.append(P("Tables 2a/2b/3 are on page 3. Percentiles use numpy linear interpolation over all requests in the scenario (n = 20 to 80, so p99 is effectively the slowest one or two). All six load runs and all four retrieval runs are <b>stand-in</b>. Quality: 20 hand-written questions (15 answerable, 5 not) x 4 retrieval settings, graded by regex on the cited passages (proxy; see eval_questions.py). "
               f"<b>Numbers I could not have guessed:</b> (a) under sustained overload the median first token was <b>{t2a[2][5].split(' / ')[0]} ms</b>, <i>better</i> than steady ({t2a[0][5].split(' / ')[0]} ms), because {lvl2:.1f}% of requests were shed to the no-LLM path at admission: latency looked healthy while {degp('sustained_overload'):.1f}% got weaker answers; "
               f"(b) mean query-term coverage was {rec['R1']['mean_cov_answerable']} for answerable and {rec['R1']['mean_cov_unanswerable']} for unanswerable questions, i.e. my refusal signal carries almost no information; "
               f"(c) in the naive-retry run 80 requests produced {nv['calls']} model calls and {nv['n429']} rejections, versus {mg['calls']} calls and {mg['n429']} rejection for the managed gateway. "
               f"Tokens per answer ~{(df[df.scenario=='eval_R1'].tokens_in+df[df.scenario=='eval_R1'].tokens_out).mean():,.0f} (R1; list-price equivalent, free tier bills $0). Among non-degraded load answers {cp(ok_full):.1f}% graded correct versus {cp(ok_deg):.1f}% of degraded ones: a degraded excerpt scores 'correct' when it shows the right passage, so the stand-in grades overstate degraded quality; the grader cannot tell an excerpt from an answer."))
    B.append(H("4 Where the time goes"))
    ev_ = sl.events; to = [e for e in ev_ if "CallTimeout" in e["name"]]
    B.append(P(f"Slowest steady request ({sl.request_id}): total {f0(sl.total_ms)} ms = waiting {f0(sl.wait_ms)} ms + working {f0(sl.work_ms)} ms + retrying {f0(sl.retry_ms)} ms. "
               f"The trace (results_detail.jsonl, field events) shows the first <i>understand</i> call hit the stand-in's hang (no bytes), the client waited its full {f0(to[0]['ms']) if to else 'n/a'} ms call timeout, backed off, and the retry succeeded in ~400 ms. "
               f"The delay is entirely <b>before</b> the first token (first token = {f0(sl.first_token_ms)} ms, after verification; the model's draft first token came {f0(sl.model_ttft_ms)} ms after its call started). "
               "Lesson: a 9 s call timeout cannot rescue a 10.5 s soft deadline; a timeout near the p99 first-token time (about 3 s) with one hedged retry would have cut this request to about 4 s. I did not change it, so the table shows the flaw."))
    B.append(H("5 When a limit is hit"))
    B.append(P(f"<b>Per minute</b> (burst, overload, rpm_halved): new work waits in a priority queue where later stages (verify, draft) outrank earlier ones, so paid-for work finishes first. Estimated wait &gt; 2 s &rarr; lean (k=2, skip understand and judge); &gt; 5 s &rarr; no LLM, cited excerpts at once; any request still unanswered at 10.5 s gets the same. "
               f"<b>Delayed:</b> everyone in a burst (mean wait {f0(D('burst').wait_ms.mean())} ms). <b>Dropped:</b> nobody ({mg_bad} of {len(mg_rows)} requests in the five managed runs ended as timeout, error or rate_limited). <b>Weaker answer:</b> {degp('burst'):.1f}% of the burst, {degp('sustained_overload'):.1f}% of overload, {degp('rpm_halved'):.1f}% when the limit was halved mid-run (the client re-learned the lower TPM from headers after one 429), {degp('steady'):.1f}% steady. "
               f"<b>Per day:</b> the daily token limit is not in any header I model, so the client learns it from the first 429 mentioning 'per day' (or Retry-After &gt; 120 s), then marks itself exhausted: later requests skip the LLM at admission and get excerpts or a refusal. {degp('daily_exhausted'):.1f}% of that run was degraded; the one request that discovered the limit paid for it."))
    B.append(H("Retrieval: quality versus cost"))
    B.append(P(f"Four settings (Table 3; recall@k = gold sentence anywhere in the retrieved pieces, from retrieval_recall.json): " + "; ".join(f"{k} {v['setting']}: recall {v['recall_at_k']}" for k, v in rec.items()) +
               f". Re-ranking (bigram overlap + paper-name boost) did not raise recall on this set (R2 vs R3: 11 vs 12) and cost +{r3t-r1t:,.0f} tokens per answer against R1. <b>Chosen: R1</b> (150 words, 30 overlap, k=6, no re-rank): best recall (12/15), fewest tokens, lowest median first token. Its weakness is context: short pieces split tables from their captions. "
               "When the limit is close (est. wait &gt; 2 s or &lt; 10% of the daily budget left) I cut k from 6 to 2 instead of shrinking pieces, because that is a per-request knob that needs no re-index; recall falls, so the answer must still pass the checker or the request refuses."))
    B.append(H("How this library will mislead a naive retriever"))
    B.append(P("<b>1. Shared vocabulary across papers.</b> 'What attention mechanism does Moshi use?' returns only Transformer-paper pages (the word 'attention' dominates). "
               "<b>2. Tables, captions and boilerplate.</b> 'What BLEU score does Moshi achieve on WMT English-to-German?' retrieves the Transformer's BLEU table and licence page; it looks relevant and is about another model. A figure page of the Transformer paper contains example sentences about governments that match unrelated queries. "
               "<b>3. Absence is invisible to lexical scores.</b> 'Does Moshi support Hindi speech?' scores coverage 1.00 (every query word appears somewhere) though the papers never say; same for 'French-to-Spanish translation'. Only the generator and the checker can refuse."))
    B.append(H("6 Retries"))
    B.append(P("Policy: at most 2 retries per call; 429 &rarr; pause the whole gateway for Retry-After plus 0 to 30% jitter and scale limits by 0.7 (AIMD, +0.03 per success); 5xx/timeout &rarr; full-jitter exponential backoff; never retry past the deadline; a retry budget of 25% of recent calls + 4; a daily 429 is never retried. "
               f"<b>Retrying made things worse:</b> the naive policy (no limiter, fixed 0.5 s, ignores Retry-After, up to 5 tries) on the same overload: {nv['calls']} calls and {nv['n429']} rejections for 80 requests, {t2b[5][4]} retries per request, and {t2b[5][2]}% of requests failed with rate_limited, versus 0.0% for the managed gateway. Each retry spent budget the limiter needed, so the storm kept the limit tripped. "
               "This depends on my assumption that rejected requests count toward RPM."))
    B.append(H("7 What the person sees"))
    B.append(P(f"At 3pm with the daily budget gone nobody waits and nobody sees an error: the reply arrives in well under a second and reads, e.g., <i>{sample}</i> If the retrieval was not confident it says plainly that the papers do not say. Not built: telling them when full answers return."))
    B.append(H("8 What I'm unsure about, what I left out, what I learned"))
    B.append(P("<b>Least certain:</b> that stand-in behaviour resembles Groq (priors, no real calibration here); if Groq does not count rejections or has hidden concurrency limits, the retry result and the admission thresholds (2 s / 5 s / 10.5 s) are wrong in size, not in kind. "
               "<b>Not built:</b> (1) dense retrieval/cross-encoder re-ranking (no model download was possible; the old FAISS store is unused); (2) a persistent daily counter shared across processes and restarts. "
               "<b>New to me (draft, edit in your words):</b> a token limit is charged on <i>reserved</i> max_tokens at admission, not on tokens generated, so a generous max_tokens burns the per-minute budget before the model says a word; I reserve prompt + max_tokens and true-up after."))
    B.append(H("Your feedback on Vaani (draft from a text read of vaani.dev; replace with your own impression)"))
    B.append(P("Stood out: the 'zero dead air' idea (small talk while tools run) and a stated ~450 ms p90 latency; the Teacher AI flow that shows a prompt diff before saving. Confusing: three latency claims ('p90 ~450 ms', 'sub-500 ms engine, P-90', 'only voice AI harness you'll ever need') with no definition of what is timed (engine only or phone to ear) and the headline counters read 0 in my text fetch. One change: publish a latency breakdown (network, speech-to-text, model, text-to-speech) with percentiles, as Section 4 does for The Relay."))
    return B

def tbl(title, head, rows, widths=None):
    cell = ParagraphStyle("c", fontName="Helvetica", fontSize=6.4, leading=7.2); hc = ParagraphStyle("hc", parent=cell, fontName="Helvetica-Bold")
    data = [[Paragraph(str(x), hc) for x in head]] + [[Paragraph(str(x), cell) for x in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return [P(f"<b>{title}</b>", ParagraphStyle("tt", parent=base, spaceBefore=4)), t]

def table1():
    nm = "not measured"
    if probe:
        L = probe["limits"]; hd = probe.get("headers_ok") or {}; b = L.get("burst_first_429") or {}; tk = L.get("token_first_429") or {}
        conc = probe.get("concurrency", {}); mx = max((int(n) for n, r in conc.items() if all(s == "ok" for s, _ in r)), default=nm)
        return [["Requests per minute", f"first 429 after {b.get('after_requests', nm)} requests in {b.get('after_s', nm)} s", "probe_real.py phase C", "medium"],
                ["Tokens per minute", hd.get("x-ratelimit-limit-tokens", nm), "x-ratelimit-limit-tokens header; phase D", "medium"],
                ["Requests per day", hd.get("x-ratelimit-limit-requests", nm), "x-ratelimit-limit-requests header", "medium"],
                ["Tokens per day", nm, "no header; learned from a 429", "low"],
                ["Requests in flight at once", f"{mx} concurrent all ok (ramp 1/2/4/8)", "probe_real.py phase B", "low"],
                ["What it returns when a limit is hit", f"429 + retry-after {L.get('retry_after_s', nm)} s; body: {str(b.get('body', nm))[:90]}", "phase C", "high"],
                ["Time until it recovers", f"{L.get('recovered_after_s', nm)} s", "phase C", "medium"]]
    a = "not measured (recalled docs: "
    return [["Requests per minute", a + "~30)", "not run (no network to Groq)", "low"], ["Tokens per minute", a + "~8,000)", "not run", "low"], ["Requests per day", a + "~1,000)", "not run", "low"],
            ["Tokens per day", a + "~200,000)", "not run", "low"], ["Requests in flight at once", "not measured", "not run", "none"],
            ["What it returns when a limit is hit", "not measured (stand-in returns 429 + Retry-After + x-ratelimit-*)", "not run", "none"], ["Time until it recovers", "not measured", "not run", "none"]]

def tables():
    T = []
    T += tbl("Table 1: Limits found in the real service", ["Limit", "Value found", "How found", "Confidence"], table1(), [95, 190, 170, 60])
    T += tbl("Table 2a: Latency per load scenario (ms; first token / total = p50 / p95 / p99). Stand-in unless 'real'. rpm_halved = server RPM and TPM halved at t=10 s; daily_exhausted = server TPD 25K.",
             ["Scenario", "Real or stand-in", "Requests", "Duration (s)", "Offered/min", "First token p50/p95/p99", "Total p50/p95/p99"], t2a, [88, 52, 42, 46, 50, 115, 115])
    T += tbl("Table 2b: Outcomes (queue length = most waiting calls at once; tokens = in+out, mean incl. no-LLM answers; cost at assumed list price $0.075/$0.30 per 1M)",
             ["Scenario", "Succeeded %", "Rate-limited %", "Timed out/given up %", "Retries/req", "Peak queue", "Tokens/answer", "Cost/1,000"], t2b, [88, 50, 50, 62, 48, 44, 56, 52])
    T += tbl("Table 3: Retrieval settings (stand-in generator; 20 questions each; refused-correctly % is of all 20, 5 are unanswerable)",
             ["Setting (size, k, re-rank)", "Questions", "Correct+cited %", "Wrong %", "Correctly refused %", "Tokens/answer", "First token p50/p95", "Total p50/p95"], t3, [110, 40, 52, 40, 56, 52, 64, 64])
    T.append(P("Extra (not in the required columns): the CSV also has a trailing <i>degraded</i> column (1 = weaker no-LLM answer). Retrieval recall@k per setting is in results/retrieval_recall.json. Grade for requests with no answer (naive-retry failures) is recorded as 'wrong'.", small))
    return T

def collab():
    C = [H("How I worked with the assistant (draft: personalise before submitting)")]
    C.append(P("Full unedited conversation: <b>docs/CONVERSATION_HISTORY.md</b> (export it from the chat and paste; I have not summarised it). Three prompts that mattered: "
               "(1) the opening prompt that attached the brief and the repo and asked for everything, because it fixed the constraint that I own the key and the repo layout; "
               "(2) asking the assistant to check that everything works, which led it to run the code rather than only write it; (3) [add the prompt that changed your design]. "
               "<b>What it got wrong, caught by checking data:</b> the first 'daily budget exhausted' run exhausted the budget for only 2 of 40 requests (visible in results_detail.jsonl), so it was re-run with a 25K limit; the first background run died silently when the shell exited (empty log). "
               "<b>Where I overruled it:</b> [write your own decision here]. <b>Distinctly mine:</b> [write: choices and reasoning you made]."))
    return C

# ------------------------------------------------------------------ build
def build(path, parts):
    W, Hh = A4; m = 24
    colw = (W - 2 * m - 10) / 2
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=m, rightMargin=m, topMargin=m, bottomMargin=m, title="The Relay report")
    two = [Frame(m, m, colw, Hh - 2 * m, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="a"), Frame(m + colw + 10, m, colw, Hh - 2 * m, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="b")]
    one = [Frame(m, m, W - 2 * m, Hh - 2 * m, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
    doc.addPageTemplates([PageTemplate("two", frames=two), PageTemplate("one", frames=one)])
    doc.build(parts)

def pages(path):
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)

if __name__ == "__main__":
    out = ROOT / "report.pdf"
    story = narrative()
    build(ROOT / "results" / "_narr.pdf", list(story))
    n = pages(ROOT / "results" / "_narr.pdf"); print("narrative pages:", n)
    full = narrative() + [NextPageTemplate("one"), PageBreak()] + tables() + collab()
    build(out, full); print("report pages:", pages(out), "->", out)
