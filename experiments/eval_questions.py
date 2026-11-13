"""20 hand-written questions (15 answerable from the papers, 5 not). Grading is an automatic proxy:
 answerable   -> correct if answered AND the cited passages contain the gold pattern (so the claim is grounded in the
                 right place); refused_wrongly if refused; wrong otherwise.
 unanswerable -> refused_correctly if refused; wrong if it produced an answer.
Spot-check a sample by hand; a regex cannot tell a fluent wrong answer that cites the right page."""
import re

A, M, H = "attention", "moshi", "dynamic"
Q = [  # (paper, question, gold regex or None)
 (A, "How many attention heads does the base Transformer use?", r"h\s*=\s*8"),
 (A, "What BLEU score did the big Transformer get on WMT 2014 English-to-German?", r"28\.4"),
 (A, "What kind of positional encoding does the Transformer use?", r"sinusoid"),
 (A, "How many warmup steps were used in the learning rate schedule?", r"warmup_steps\s*=\s*4000"),
 (A, "How long did training the big model take and on what hardware?", r"3\.5 days on 8 P100"),
 (M, "What is Moshi's theoretical latency?", r"160\s*ms"),
 (M, "What frame rate does the Mimi codec run at?", r"12\.5\s*Hz"),
 (M, "What is Helium in the Moshi paper?", r"7B-parameter text LLM"),
 (M, "What is Inner Monologue in Moshi?", r"aligned text and speech"),
 (M, "How many quantizers does Mimi use and what bitrate does that give?", r"Q = 8 quantizers"),
 (H, "How does the routing module predict chunk boundaries?", r"cosine similarity between adjacent"),
 (H, "Why does dynamic chunking need a ratio loss?", r"trivial solutions"),
 (H, "What does the smoothing module do in H-Net?", r"smoothing module which interpolates"),
 (H, "Does H-Net work directly on raw bytes?", r"byte"),
 (H, "What does dynamic chunking aim to replace in the modeling pipeline?", r"tokeniz"),
 (None, "What BLEU score does Moshi achieve on WMT English-to-German?", None),          # lexical-collision trap
 (None, "What is the context window of GPT-4?", None),
 (None, "Does Moshi support Hindi speech?", None),
 (None, "What did the Transformer paper report for French-to-Spanish translation?", None),
 (None, "How much did it cost in dollars to train the H-Net models?", None),
]
QUESTIONS = [dict(id=f"q{i+1:02d}", paper=p, question=q, gold=g) for i, (p, q, g) in enumerate(Q)]


def grade(q, r):
    if q["gold"] is None:
        return "refused_correctly" if r.refused else "wrong"
    if r.refused: return "refused_wrongly"
    cited = " ".join(c["text"] for c in r.cited)
    return "correct" if r.cited and re.search(q["gold"], re.sub(r"\s+", " ", cited), re.I) else "wrong"
