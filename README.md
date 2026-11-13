# The Relay

Multi-step RAG assistant over three arXiv papers, built to survive rate limits, bursts and slow calls.

```
question -> understand (LLM) -> retrieve (BM25 tool) -> decide (branch/loop) -> draft (LLM, buffered)
         -> verify (deterministic citation tool + LLM judge when borderline) -> deliver / refuse / weaker answer
```
* `app/gateway.py` – limits, priority queue, AIMD, Retry-After handling, retry budget, daily-limit handling
* `app/pipeline.py` – the steps, branches and loops; `app/retrieval/` – ingest, chunking, BM25 (+ original FAISS store)
* `simulator/standin.py` – stand-in model service (slow, failing, limited, load-sensitive, streaming)
* `experiments/` – `probe_real.py` (real Groq), `calibrate.py`, `run_all.py` (load shapes + retrieval settings), `eval_questions.py`
* `report/make_report.py` – builds every table from `results/results.csv` and renders `report.pdf`

## Run
```
pip install -r requirements.txt           
cp .env.example .env                        # put GROQ_API_KEY and GROQ_MODEL in it
python -m app.retrieval.ingest              # PDFs -> data/processed/pages.json
python main.py "What is Moshi's theoretical latency?"          # real Groq
# real measurements (uses ~200 of your daily requests):
python -m experiments.probe_real && python -m experiments.calibrate
RELAY_SETTING=R1 python -m experiments.run_all --backend groq --phase all
# stand-in loads (calibrated from the above if results/calibration.json exists):
python -m experiments.run_all --phase all --fresh   # then: RELAY_SETTING=R1 ... --phase load
python -m report.make_report                        # regenerates report.pdf from results.csv
```
Never commit `.env`. Papers pinned: 1706.03762v7, 2410.00037v1, 2507.07955v2 (version stamps verified on page 1 of the PDFs in `data/papers`).
