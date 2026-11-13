"""Lexical retriever (BM25) with optional cheap re-ranking.

Why BM25 and not the FAISS/MiniLM store already in this repo: it needs no model
download, runs in <5 ms (so retrieval never competes with the LLM for time), and
its confidence signal (query-term coverage) is directly interpretable, which the
pipeline uses to decide whether to refuse. The dense store is kept for comparison.
"""
import json, math, re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.retrieval.chunking import create_chunks

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "data" / "processed" / "pages.json"

STOP = set("""a an the of to in on for and or is are was were be been being it its this that these those
with as by at from how what why which who whom when where does do did can could should would will
use used using paper papers about into than then their there they them his her our we you your i not no
also such more most some any each both between over under per via""".split())

PAPER_ALIASES = {  # question mentions -> title fragment (used by the re-ranker)
    "attention is all you need": ["transformer", "vaswani", "multi-head", "self-attention", "bleu", "wmt"],
    "moshi": ["moshi", "mimi", "helium", "inner monologue", "full-duplex", "kyutai"],
    "dynamic chunking": ["h-net", "hnet", "dynamic chunking", "dechunk", "routing module", "smoothing module"],
}


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def toks(text: str):
    return [stem(w) for w in re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower())
            if w not in STOP]


def short_title(title: str) -> str:
    t = title.lower()
    if "attention" in t: return "Attention Is All You Need"
    if "moshi" in t: return "Moshi"
    return "Dynamic Chunking (H-Net)"


class Retriever:
    def __init__(self, chunk_size=350, overlap=60, k=4, rerank=False):
        self.chunk_size, self.overlap, self.k, self.rerank = chunk_size, overlap, k, rerank
        papers = json.loads(PAGES.read_text())
        self.chunks = create_chunks(papers, chunk_size, overlap)
        self.tok = [Counter(toks(c["text"])) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tok]
        self.avg = sum(self.len) / len(self.len)
        df = Counter()
        for t in self.tok:
            df.update(t.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}
        self.n = n

    @property
    def name(self):
        return f"{self.chunk_size}w/{self.overlap}ov, k={self.k}, rerank={'yes' if self.rerank else 'no'}"

    def _bm25(self, q, i, k1=1.4, b=0.75):
        tf, L, s = self.tok[i], self.len[i], 0.0
        for w in q:
            f = tf.get(w, 0)
            if f:
                s += self.idf.get(w, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / self.avg))
        return s

    def search(self, query: str, k: int | None = None):
        k = k or self.k
        q = list(dict.fromkeys(toks(query)))
        if not q:
            return [], 0.0
        mass = sum(self.idf.get(w, 0) for w in q) or 1.0
        scored = [(self._bm25(q, i), i) for i in range(self.n)]
        scored.sort(reverse=True)
        pool = scored[: (k * 5 if self.rerank else k)]
        ql = query.lower()
        out = []
        for s, i in pool:
            c = self.chunks[i]
            if self.rerank:
                text = c["text"].lower()
                bigr = {" ".join(p) for p in zip(q, q[1:])}
                bhit = sum(1 for g in bigr if g in " ".join(toks(c["text"]))) / max(1, len(bigr))
                paper_boost = 0.0
                for frag, keys in PAPER_ALIASES.items():
                    if any(key in ql for key in keys) and frag in c["title"].lower():
                        paper_boost = 0.35
                s = s / (pool[0][0] or 1) + 0.5 * bhit + paper_boost
            cov = sum(self.idf.get(w, 0) for w in q if w in self.tok[i]) / mass
            out.append({**c, "score": s, "coverage": cov})
        out.sort(key=lambda r: r["score"], reverse=True)
        out = out[:k]
        best_cov = max((r["coverage"] for r in out), default=0.0)
        return out, best_cov


@lru_cache(maxsize=8)
def get_retriever(chunk_size=350, overlap=60, k=4, rerank=False) -> Retriever:
    if not PAGES.exists():
        from app.retrieval.ingest import ingest_all_papers
        ingest_all_papers()
    return Retriever(chunk_size, overlap, k, rerank)
