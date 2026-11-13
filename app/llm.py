"""Streaming chat client for Groq (or the stand-in; both speak the OpenAI wire format)."""
import asyncio, json, re, time
from dataclasses import dataclass, field
import httpx
from app.config import Settings


class RateLimited(Exception):
    def __init__(self, retry_after, kind, headers, body=""):
        super().__init__(f"429 {kind} retry_after={retry_after}")
        self.retry_after, self.kind, self.headers, self.body = retry_after, kind, headers, body

    @property
    def daily(self):
        return "per day" in self.body.lower() or "(rpd)" in self.body.lower() or "(tpd)" in self.body.lower() or self.retry_after > 120


class ServerError(Exception): pass
class CallTimeout(Exception): pass


@dataclass
class LLMResult:
    text: str
    ttft_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    headers: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, s: Settings):
        self.s = s
        self.url = (s.standin_base if s.backend == "standin" else s.groq_base) + "/chat/completions"
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5),
                                      limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))

    async def close(self): await self.http.aclose()

    async def chat(self, messages, max_tokens, timeout_s) -> LLMResult:
        s = self.s
        body = {"model": s.model, "messages": messages, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        if s.backend == "groq" and s.model.startswith("openai/gpt-oss"):
            body["max_completion_tokens"] = max_tokens + s.reasoning_headroom
            body["reasoning_effort"] = "low"
        else:
            body["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {s.groq_key}"} if s.backend == "groq" else {}
        t0 = time.perf_counter(); ttft = None; text = ""; usage = {}
        try:
            async with asyncio.timeout(timeout_s):
                async with self.http.stream("POST", self.url, json=body, headers=headers) as r:
                    h = {k.lower(): v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
                    if r.status_code == 429:
                        b = (await r.aread()).decode("utf-8", "replace")
                        ra = float(h.get("retry-after") or (re.search(r"try again in ([\d\.]+)s", b) or [0, 2])[1])
                        raise RateLimited(ra, "429", h, b)
                    if r.status_code >= 500:
                        await r.aread(); raise ServerError(f"HTTP {r.status_code}")
                    if r.status_code != 200:
                        b = (await r.aread()).decode("utf-8", "replace"); raise ServerError(f"HTTP {r.status_code}: {b[:120]}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "): continue
                        raw = line[6:]
                        if raw == "[DONE]": break
                        ev = json.loads(raw)
                        if ev.get("usage"): usage = ev["usage"]
                        elif ev.get("x_groq", {}).get("usage"): usage = ev["x_groq"]["usage"]
                        for ch in ev.get("choices", []):
                            c = (ch.get("delta") or {}).get("content")
                            if c:
                                if ttft is None: ttft = (time.perf_counter() - t0) * 1000
                                text += c
        except (TimeoutError, httpx.TimeoutException):
            raise CallTimeout(f"no complete reply within {timeout_s:.1f}s")
        except httpx.TransportError as e:
            raise ServerError(f"transport: {e}")
        tot = (time.perf_counter() - t0) * 1000
        return LLMResult(text.strip(), ttft if ttft is not None else tot, tot,
                         usage.get("prompt_tokens", len(json.dumps(messages)) // 4),
                         usage.get("completion_tokens", len(text) // 4), h)
