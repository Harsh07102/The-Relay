"""Central settings. Secrets come from .env (never committed)."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return type(default)(os.getenv(name, default))


@dataclass
class Settings:
    backend: str = os.getenv("RELAY_BACKEND", "groq")          # "groq" | "standin"
    groq_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base: str = "https://api.groq.com/openai/v1"
    standin_base: str = os.getenv("STANDIN_URL", "http://127.0.0.1:8099/v1")

    # What the client ASSUMES about the provider until headers / 429s correct it.
    # Groq free-tier numbers below are from memory of the docs -> LOW confidence;
    # experiments/probe_real.py measures them and the gateway re-learns from headers.
    rpm: int = _f("RELAY_RPM", 30)
    tpm: int = _f("RELAY_TPM", 8000)
    rpd: int = _f("RELAY_RPD", 1000)
    tpd: int = _f("RELAY_TPD", 200000)
    max_inflight: int = _f("RELAY_MAX_INFLIGHT", 4)
    safety: float = 0.92            # use at most 92% of any per-minute limit

    # Latency policy (seconds). "Fast" = verified answer starts within 4 s (p50)
    # and the person is still there at 15 s.
    patience_s: float = _f("RELAY_PATIENCE", 15.0)
    soft_deadline_s: float = _f("RELAY_SOFT", 10.5)   # after this, send a weaker answer
    call_timeout_s: float = 9.0
    shed_wait_s: float = 5.0        # estimated queue wait above which we skip the LLM
    lean_wait_s: float = 2.0        # above this we drop optional steps

    # Pipeline budgets (max output tokens per step)
    understand_max: int = 60
    draft_max: int = 220
    verify_max: int = 8
    reasoning_headroom: int = 200   # gpt-oss spends hidden tokens on reasoning

    # List-price equivalent, USD per 1M tokens (assumption; free tier bills $0)
    price_in: float = 0.075
    price_out: float = 0.30

    refuse_text: str = ("The papers in this library don't say. I won't guess; "
                        "try rephrasing, or ask about a specific paper.")


SETTINGS = Settings()
