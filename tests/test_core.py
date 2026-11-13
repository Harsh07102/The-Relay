import asyncio, pytest
from app.pipeline import check_citations
from app.retrieval.search import get_retriever
from app.config import Settings
from app.gateway import Gateway, Trace, DailyExhausted
from app.llm import LLMClient

P = [{"text": "We employ h = 8 parallel attention layers, or heads.", "title": "Attention Is All You Need", "page_number": 5}]

def test_checker_accepts_supported_cited_sentence():
    ok, sup, ids = check_citations("The base model employs h = 8 parallel attention heads [S1].", P); assert ok and ids == [1] and sup > .5

def test_checker_rejects_uncited_or_invented_number():
    assert not check_citations("The model uses 8 heads.", P)[0]
    assert not check_citations("It uses h = 12 heads [S1].", P)[0]

def test_retrieval_finds_gold():
    res, cov = get_retriever(150, 30, 6, False).search("How many attention heads does the base Transformer use?")
    assert any("h = 8" in r["text"] for r in res)

def test_gateway_daily_block_fails_fast():
    async def go():
        s = Settings(backend="standin"); gw = Gateway(s, LLMClient(s)); gw.start(); gw.day_blocked_until = 1e18
        with pytest.raises(DailyExhausted): await gw.acquire(100, 1, 1e18)
    asyncio.run(go())
