"""Model-independent retrieval check: does the gold passage appear in the top-k? (answerable questions only)
Also records query-term coverage for answerable vs unanswerable questions -> shows the refusal signal is weak."""
import json, re, statistics as st
from pathlib import Path
from app.retrieval.search import get_retriever
from experiments.eval_questions import QUESTIONS
from experiments.run_all import SETTINGS_GRID, RES
out = {}
for key, cfg in SETTINGS_GRID.items():
    r = get_retriever(*cfg); hit = 0; ca, cu = [], []
    for q in QUESTIONS:
        res, c = r.search(q["question"])
        if q["gold"]:
            hit += any(re.search(q["gold"], re.sub(r"\s+", " ", x["text"]), re.I) for x in res); ca.append(c)
        else: cu.append(c)
    out[key] = dict(setting=r.name, recall_at_k=f"{hit}/15", mean_cov_answerable=round(st.mean(ca), 2), mean_cov_unanswerable=round(st.mean(cu), 2))
(RES / "retrieval_recall.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
