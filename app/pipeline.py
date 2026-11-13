"""The Relay pipeline: understand -> retrieve(tool) -> decide(branch/loop) -> draft -> verify -> deliver.

Design decision that shaped everything: the draft is BUFFERED and verified before any
answer token reaches the person (never send a wrong/uncited answer just to be fast).
So "first token" = first token of the VERIFIED answer; the model's own TTFT is
recorded separately (trace.model_ttft). Speed is bought elsewhere: fewer calls
(optional steps dropped under load), a priority queue, and a weaker-but-honest
extractive answer when time or budget runs out (nobody is turned away).
"""
import asyncio, json, re, time
from dataclasses import dataclass, field
from app.config import Settings
from app.gateway import Gateway, Trace, DailyExhausted, DeadlineExceeded
from app.llm import RateLimited, ServerError, CallTimeout
from app.retrieval.search import get_retriever, short_title, toks

COVERAGE_MIN = 0.60       # branch: below this the papers probably don't cover the question
SUPPORT_PASS, SUPPORT_BORDER = 0.50, 0.75

SYS = {
 "understand": "TASK: understand\nRewrite the user's question as a short keyword search query for a library of three ML papers. "
               'Reply with JSON only: {"query": "..."}',
 "draft": "TASK: draft\nAnswer ONLY from the numbered passages. Be concise (max 3 sentences). End EVERY sentence with the "
          "passage tags that support it, like [S1] or [S1][S3]. Copy numbers exactly as written. If the passages do not "
          "contain the answer, reply exactly: NOT_IN_PAPERS",
 "draft_strict": "TASK: draft\nYour previous answer was not supported. Use only wording and numbers that appear in the passages, "
                 "one or two sentences, each ending with [S#] tags. If unsure reply exactly: NOT_IN_PAPERS",
 "verify": "TASK: verify\nDoes the PASSAGE support the CLAIM? Reply exactly SUPPORTED or UNSUPPORTED.",
}


@dataclass
class Result:
    rid: str; question: str; setting: str
    answer: str = ""; refused: bool = False; degraded: bool = False; level: int = 0
    cited: list = field(default_factory=list)
    status: str = "ok"; first_token_ms: float | None = None; total_ms: float = 0.0
    retries: int = 0; tokens_in: int = 0; tokens_out: int = 0
    trace: Trace | None = None; error: str = ""; steps: list = field(default_factory=list)


def _fmt(passages):
    return "\n".join(f"[S{i+1}] ({short_title(p['title'])}, p.{p['page_number']})\n{p['text']}\n" for i, p in enumerate(passages))


def _numbers(t): return set(re.findall(r"\d+(?:\.\d+)?", t))


def check_citations(answer, passages):
    """Deterministic checker (the 'tool'): every sentence cited, ids valid, numbers and wording present in cited text.
    Returns (ok, min_support, cited_ids)."""
    sents = [s for s in re.split(r"(?<=[\.\!\?\]])\s+(?=[A-Z0-9])", answer.strip()) if s.strip()]
    if not sents: return False, 0.0, []
    worst, cited = 1.0, []
    for s in sents:
        ids = [int(i) for i in re.findall(r"\[S(\d+)\]", s)]
        if not ids or any(i < 1 or i > len(passages) for i in ids): return False, 0.0, []
        cited += ids
        body = re.sub(r"\[S\d+\]", "", s)
        ptext = " ".join(passages[i - 1]["text"] for i in ids)
        if not _numbers(body) <= _numbers(ptext): return False, 0.0, []
        w = set(toks(body))
        if w: worst = min(worst, len(w & set(toks(ptext))) / len(w))
    return True, worst, sorted(set(cited))


class Pipeline:
    def __init__(self, s: Settings, gw: Gateway, setting=(350, 60, 4, False)):
        self.s, self.gw = s, gw
        self.setting = setting

    @property
    def sname(self): return get_retriever(*self.setting).name

    # -------------------------------------------------------------- public
    async def run(self, rid, question) -> Result:
        t0 = time.monotonic(); tr = Trace()
        r = Result(rid, question, self.sname, trace=tr)
        ctx = {}
        r.level = self.gw.level()
        managed = self.gw.policy == "managed"
        deadline = t0 + self.s.patience_s
        try:
            if r.level >= 2 and managed:
                raise DailyExhausted(0) if time.monotonic() < self.gw.day_blocked_until else DeadlineExceeded("shed at admission")
            await asyncio.wait_for(self._flow(r, question, ctx, tr, t0, deadline, r.level), self.s.soft_deadline_s if managed else self.s.patience_s)
        except (asyncio.TimeoutError, DailyExhausted, DeadlineExceeded, RateLimited, ServerError, CallTimeout) as e:
            r.error = type(e).__name__
            if managed:
                self._fallback(r, question, ctx)
            else:
                r.status = "rate_limited" if isinstance(e, RateLimited) else ("timeout" if isinstance(e, (asyncio.TimeoutError, DeadlineExceeded, CallTimeout)) else "error")
        t1 = time.monotonic()
        r.total_ms = (t1 - t0) * 1000
        if r.status == "ok" and r.first_token_ms is None: r.first_token_ms = r.total_ms
        r.retries, r.tokens_in, r.tokens_out = tr.retries, tr.tokens_in, tr.tokens_out
        if r.total_ms > self.s.patience_s * 1000 and r.status == "ok": r.status = "timeout"   # the person left
        return r

    # -------------------------------------------------------------- steps
    async def _flow(self, r, question, ctx, tr, t0, deadline, level):
        cs, ov, k, rr = self.setting
        if level >= 1: k = min(k, 2)                       # lean: fewer passages -> fewer tokens
        retr = get_retriever(cs, ov, k, rr)
        query = question
        # 1. understand (optional: skipped when lean)
        if level == 0:
            try:
                res = await self.gw.call([{"role": "system", "content": SYS["understand"]}, {"role": "user", "content": f"QUESTION: {question}"}],
                                         self.s.understand_max, "understand", deadline, tr, t0)
                m = re.search(r"\{.*\}", res.text, re.S)
                q2 = json.loads(m.group(0)).get("query") if m else None
                if q2 and len(q2) > 3: query = f"{question} {q2}"
            except (ValueError, KeyError): pass            # bad JSON -> raw question, no harm
        # 2. retrieve (tool call with its own timeout)
        t = time.monotonic()
        passages, cov = await asyncio.wait_for(asyncio.to_thread(retr.search, query, k), 2.0)
        tr.ev(t0, "work", "retrieve", time.monotonic() - t); ctx["passages"], ctx["cov"] = passages, cov
        # 3. decide: branch + loop
        if cov < COVERAGE_MIN:
            passages, cov = await asyncio.wait_for(asyncio.to_thread(retr.search, question + " " + query, k + 2), 2.0)   # loop: widen once
            ctx["passages"], ctx["cov"] = passages, cov
            if cov < COVERAGE_MIN:
                return self._refuse(r, tr, t0, "low_coverage")
        # 4. draft (buffered) + 5. verify (loop once)
        for attempt in range(2):
            sysk = "draft" if attempt == 0 else "draft_strict"
            res = await self.gw.call([{"role": "system", "content": SYS[sysk]}, {"role": "user", "content": f"{_fmt(passages)}\nQUESTION: {question}\n"}],
                                     self.s.draft_max, "draft", deadline, tr, t0)
            if res.text.strip().startswith("NOT_IN_PAPERS"):
                return self._refuse(r, tr, t0, "model_refused")
            ok, support, ids = check_citations(res.text, passages)
            if ok and support >= SUPPORT_BORDER: return self._deliver(r, res.text, passages, ids, tr, t0)
            if ok and support >= SUPPORT_PASS:
                if level == 0:                              # borderline -> LLM judge (optional step)
                    claim = re.sub(r"\[S\d+\]", "", res.text)
                    ptxt = " ".join(passages[i - 1]["text"] for i in ids)
                    v = await self.gw.call([{"role": "system", "content": SYS["verify"]}, {"role": "user", "content": f"PASSAGE: {ptxt[:1800]}\nCLAIM: {claim}"}],
                                           self.s.verify_max, "verify", deadline, tr, t0)
                    if "UNSUPPORTED" not in v.text.upper(): return self._deliver(r, res.text, passages, ids, tr, t0)
                else:
                    return self._deliver(r, res.text, passages, ids, tr, t0)
        return self._refuse(r, tr, t0, "unverified")

    def _deliver(self, r, text, passages, ids, tr, t0):
        r.first_token_ms = (time.monotonic() - t0) * 1000
        ans = re.sub(r"\[S(\d+)\]", lambda m: f"[{short_title(passages[int(m.group(1))-1]['title'])}, p.{passages[int(m.group(1))-1]['page_number']}]", text)
        r.answer = ans; r.cited = [passages[i - 1] for i in ids]

    def _refuse(self, r, tr, t0, why):
        r.first_token_ms = (time.monotonic() - t0) * 1000
        r.refused = True; r.answer = self.s.refuse_text; r.error = why

    def _fallback(self, r, question, ctx):
        """Weaker answer: quote the best passage(s) verbatim with citations; refuse if retrieval is not confident."""
        r.degraded = True
        if "passages" not in ctx:
            retr = get_retriever(*self.setting)
            ctx["passages"], ctx["cov"] = retr.search(question, 2)
        if ctx["cov"] < COVERAGE_MIN or not ctx["passages"]:
            r.refused = True; r.answer = self.s.refuse_text; return
        top = ctx["passages"][:2]
        qw = set(toks(question)); best = []
        for p in top:
            sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", p["text"]))
            sc = max(sents, key=lambda s: len(qw & set(toks(s))) if len(s.split()) > 5 else -1)
            best.append((p, " ".join(sc.split()[:50])))
        r.answer = ("The model is busy, so here are the most relevant excerpts instead of a written answer: " +
                    " ".join(f"\u201c{t}\u201d [{short_title(p['title'])}, p.{p['page_number']}]" for p, t in best))
        r.cited = [p for p, _ in best]
