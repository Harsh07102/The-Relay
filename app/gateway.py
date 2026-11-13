"""The gateway: the one place that knows about limits, queues, retries and shedding.

Managed policy (default):
  * priority queue: later pipeline stages outrank earlier ones, so work already paid
    for finishes before new work starts (no wasted tokens on abandoned requests);
  * per-minute RPM/TPM windows with token *reservations* (prompt + max_tokens) that are
    trued-up to actual usage; effective limits scale by an AIMD factor that drops on
    429 and recovers on success; limits are re-learned from x-ratelimit-* headers;
  * 429 -> global pause for Retry-After (+jitter): every caller waits, none hammer;
  * daily limit -> DailyExhausted immediately (waiting an hour is not a retry);
  * retry budget (<=25% of recent calls + 4) so retries cannot become a storm.
Naive policy (for the "retry made it worse" experiment): no limiter, fixed 0.5 s retry,
ignores Retry-After, up to 5 attempts.
"""
import asyncio, heapq, random, time
from collections import deque
from app.config import Settings
from app.llm import LLMClient, RateLimited, ServerError, CallTimeout, LLMResult

STAGE_PRIO = {"verify": 0, "draft": 1, "understand": 2}


class DailyExhausted(Exception):
    def __init__(self, retry_after): super().__init__(f"daily budget exhausted ({retry_after:.0f}s)"); self.retry_after = retry_after
class DeadlineExceeded(Exception): pass


class Trace:
    """Latency accounting for one request: wait (queued), work (model running), retry (failed attempts + backoff)."""
    def __init__(self): self.wait = self.work = self.retry = 0.0; self.events = []; self.retries = 0; self.tokens_in = self.tokens_out = 0; self.model_ttft = None
    def ev(self, t0, kind, name, dur, **kw): self.events.append(dict(t=round((time.monotonic() - t0) * 1000), kind=kind, name=name, ms=round(dur * 1000), **kw))


class Gateway:
    def __init__(self, s: Settings, client: LLMClient, policy="managed"):
        self.s, self.client, self.policy = s, client, policy
        self.tpm_base, self.rpm_base, self.rpd, self.tpd = s.tpm, s.rpm, s.rpd, s.tpd
        self.factor = 1.0
        self.req_win, self.tok_win = deque(), deque()
        self.day_req = self.day_tok = 0
        self.day_blocked_until = 0.0; self.pause_until = 0.0
        self.inflight = 0
        self.heap = []; self.seq = 0
        self.wake = asyncio.Event(); self.peak_queue = 0
        self.recent_calls, self.recent_retries = deque(), deque()
        self.n429 = self.ncalls = 0
        self._task = None

    def start(self):
        if self._task is None: self._task = asyncio.create_task(self._pump())

    async def stop(self):
        if self._task: self._task.cancel()

    # ------------------------------------------------------------------ capacity maths
    @property
    def rpm(self): return max(1, int(self.rpm_base * self.factor))
    @property
    def tpm(self): return max(200, int(self.tpm_base * self.factor))

    def _prune(self, now):
        while self.req_win and now - self.req_win[0] > 60: self.req_win.popleft()
        while self.tok_win and now - self.tok_win[0][0] > 60: self.tok_win.popleft()

    def _delay_for(self, est, now):
        if now < self.pause_until: return self.pause_until - now
        if self.inflight >= self.s.max_inflight: return 0.05
        self._prune(now)
        if len(self.req_win) >= self.rpm * self.s.safety: return max(0.05, self.req_win[0] + 60 - now)
        used = sum(e[1] for e in self.tok_win); cap = self.tpm * self.s.safety
        if used + est > cap:
            acc = used + est
            for e in self.tok_win:
                acc -= e[1]
                if acc <= cap: return max(0.05, e[0] + 60 - now)
            return 1.0
        return 0.0

    def est_wait(self):
        """Seconds a new request's first call would probably queue (drives admission level)."""
        now = time.monotonic(); self._prune(now)
        queued = sum(w[2].est for w in self.heap if not w[2].fut.done())
        used = sum(e[1] for e in self.tok_win)
        over = max(0.0, used + queued - self.tpm * self.s.safety)
        w = over / (self.tpm / 60.0)
        w = max(w, max(0, len(self.heap) + len(self.req_win) - self.rpm * self.s.safety) / (self.rpm / 60.0))
        return max(w, self.pause_until - now, 0.0)

    def daily_left_frac(self): return max(0.0, 1 - self.day_tok / max(1, self.tpd))
    def level(self):
        """0 normal, 1 lean (drop optional steps / fewer passages), 2 extractive only (no LLM)."""
        if time.monotonic() < self.day_blocked_until: return 2
        w = self.est_wait()
        if w > self.s.shed_wait_s: return 2
        if w > self.s.lean_wait_s or self.daily_left_frac() < 0.10: return 1
        return 0

    # ------------------------------------------------------------------ scheduler
    async def _pump(self):
        while True:
            if not self.heap:
                self.wake.clear(); await self.wake.wait(); continue
            prio, seq, w = self.heap[0]
            if w.fut.done(): heapq.heappop(self.heap); continue
            now = time.monotonic(); d = self._delay_for(w.est, now)
            if d <= 0:
                heapq.heappop(self.heap)
                self.inflight += 1; entry = [now, w.est]
                self.req_win.append(now); self.tok_win.append(entry)
                w.fut.set_result(entry); continue
            self.wake.clear()
            try: await asyncio.wait_for(self.wake.wait(), timeout=min(d, 0.25))
            except asyncio.TimeoutError: pass

    async def acquire(self, est, prio, deadline):
        now = time.monotonic()
        if now < self.day_blocked_until: raise DailyExhausted(self.day_blocked_until - now)
        class W: pass
        w = W(); w.est = est; w.fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self.heap, (prio, self.seq, w)); self.seq += 1
        self.peak_queue = max(self.peak_queue, len(self.heap)); self.wake.set()
        try: return await asyncio.wait_for(w.fut, timeout=max(0.01, deadline - now))
        except asyncio.TimeoutError: raise DeadlineExceeded("queued past deadline")

    def release(self, entry, actual):
        self.inflight -= 1
        if entry is not None and actual: entry[1] = actual
        self.day_req += 1; self.day_tok += actual or 0
        self.wake.set()

    def observe(self, h):
        try:
            if "x-ratelimit-limit-tokens" in h: self.tpm_base = int(h["x-ratelimit-limit-tokens"])
            if "x-ratelimit-limit-requests" in h: self.rpd = int(h["x-ratelimit-limit-requests"])
        except ValueError: pass

    # ------------------------------------------------------------------ the call
    async def call(self, messages, max_tokens, stage, deadline, trace: Trace, t0, json_chars=None):
        est = int(len(str(messages)) / 3.8) + max_tokens + (self.s.reasoning_headroom if "gpt-oss" in self.s.model and self.s.backend == "groq" else 0)
        prio = STAGE_PRIO[stage]; attempt = 0; managed = self.policy == "managed"
        while True:
            now = time.monotonic()
            entry = None
            if managed:
                tq = time.monotonic()
                try: entry = await self.acquire(est, prio, deadline)
                finally: trace.wait += time.monotonic() - tq; trace.ev(t0, "wait", stage, time.monotonic() - tq)
            tc = time.monotonic(); self.ncalls += 1; self.recent_calls.append(tc)
            try:
                res: LLMResult = await self.client.chat(messages, max_tokens, max(1.0, min(self.s.call_timeout_s, deadline - tc)))
                dt = time.monotonic() - tc
                if managed: self.release(entry, res.tokens_in + res.tokens_out); self.factor = min(1.0, self.factor + 0.03); self.observe(res.headers)
                trace.work += dt; trace.ev(t0, "work", stage, dt, ttft=round(res.ttft_ms))
                trace.tokens_in += res.tokens_in; trace.tokens_out += res.tokens_out
                if stage == "draft": trace.model_ttft = res.ttft_ms
                return res
            except (RateLimited, ServerError, CallTimeout) as e:
                dt = time.monotonic() - tc
                if managed: self.release(entry, est if isinstance(e, RateLimited) else 0)
                trace.retry += dt; trace.ev(t0, "retry", f"{stage}:{type(e).__name__}", dt)
                if isinstance(e, RateLimited):
                    self.n429 += 1; self.observe(e.headers)
                    if managed:
                        if e.daily:
                            self.day_blocked_until = time.monotonic() + e.retry_after; raise DailyExhausted(e.retry_after)
                        self.factor = max(0.3, self.factor * 0.7)
                        self.pause_until = max(self.pause_until, time.monotonic() + e.retry_after)
                attempt += 1; now = time.monotonic()
                if managed:
                    self._trim(now)
                    budget_ok = len(self.recent_retries) <= 0.25 * len(self.recent_calls) + 4
                    wait = (max(e.retry_after, 0.2) * (1 + 0.3 * random.random())) if isinstance(e, RateLimited) else random.uniform(0, 0.4 * 2 ** attempt)
                    if attempt > 2 or not budget_ok or now + wait + 1.5 > deadline: raise
                else:
                    wait = 0.5                                   # naive: fixed, ignores Retry-After
                    if attempt > 5 or now + wait > deadline: raise
                self.recent_retries.append(now); trace.retries += 1
                trace.retry += wait; trace.ev(t0, "retry", "backoff", wait)
                await asyncio.sleep(wait)

    def _trim(self, now):
        for q in (self.recent_calls, self.recent_retries):
            while q and now - q[0] > 30: q.popleft()
