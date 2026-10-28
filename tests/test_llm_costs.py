from siteproof import llm


def test_cost_unknown_when_price_missing(monkeypatch):
    monkeypatch.setattr(llm, "load_prices", lambda: {"m": {"input": None, "output": None}})
    assert llm.cost_usd("m", 1000, 1000) is None


def test_cost_and_counterfactual(monkeypatch):
    prices = {"small": {"input": 0.1, "output": 0.4}, "big": {"input": 1.0, "output": 4.0}}
    monkeypatch.setattr(llm, "load_prices", lambda: prices)
    rows = [
        {"model": "small", "tokens_in": 1_000_000, "tokens_out": 500_000},
        {"model": "big", "tokens_in": 100_000, "tokens_out": 10_000},
    ]
    s = llm.summarize_costs(rows, judge_model="big")
    assert abs(s["per_model"]["small"]["cost_usd"] - 0.3) < 1e-9
    assert abs(s["per_model"]["big"]["cost_usd"] - 0.14) < 1e-9
    assert abs(s["total_usd"] - 0.44) < 1e-9
    # all 1.1M in / 0.51M out at big prices
    assert abs(s["all_judge_usd"] - (1.1 + 2.04)) < 1e-9


def test_cache_key_stable():
    a = llm._cache_key("m", [{"role": "user", "content": "x"}], {"temperature": 0})
    b = llm._cache_key("m", [{"role": "user", "content": "x"}], {"temperature": 0})
    c = llm._cache_key("m", [{"role": "user", "content": "y"}], {"temperature": 0})
    assert a == b != c
