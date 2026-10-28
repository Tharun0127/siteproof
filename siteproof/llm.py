"""Token Factory client wrapper: disk cache, cost ledger, latency."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "llm"
LEDGER = ROOT / ".cache" / "ledger.jsonl"
PRICES = ROOT / "config" / "prices.yaml"

load_dotenv(ROOT / ".env")


def make_client() -> OpenAI:
    key = os.getenv("TOKEN_FACTORY_API_KEY")
    if not key:
        raise RuntimeError("TOKEN_FACTORY_API_KEY is not set (copy .env.example to .env).")
    base = os.getenv("TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    return OpenAI(api_key=key, base_url=base, timeout=600)


def load_prices() -> dict[str, dict[str, float | None]]:
    data = yaml.safe_load(PRICES.read_text(encoding="utf-8")) or {}
    return data.get("models") or {}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    p = load_prices().get(model) or {}
    if p.get("input") is None or p.get("output") is None:
        return None
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class CallResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cached: bool
    cost_usd: float | None
    raw: dict[str, Any]


class LLM:
    """Every model call goes through here so it is cached and costed."""

    def __init__(self, client: OpenAI | None = None, run_id: str | None = None):
        self._client = client
        self.run_id = run_id

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_client()
        return self._client

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        stage: str = "",
        use_cache: bool = True,
        **params: Any,
    ) -> CallResult:
        key = _cache_key(model, messages, params)
        path = CACHE_DIR / f"{key}.json"
        if use_cache and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            res = self._result(model, raw, raw.get("_latency_s", 0.0), cached=True)
        else:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(model=model, messages=messages, **params)
            latency = time.perf_counter() - t0
            raw = resp.model_dump()
            raw["_latency_s"] = latency
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw), encoding="utf-8")
            res = self._result(model, raw, latency, cached=False)
        self._log(res, stage)
        return res

    @staticmethod
    def _result(model: str, raw: dict, latency: float, cached: bool) -> CallResult:
        usage = raw.get("usage") or {}
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        text = (raw["choices"][0]["message"].get("content") or "") if raw.get("choices") else ""
        return CallResult(
            text=text,
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=latency,
            cached=cached,
            cost_usd=cost_usd(model, tin, tout),
            raw=raw,
        )

    def _log(self, res: CallResult, stage: str) -> None:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "run_id": self.run_id,
            "stage": stage,
            "model": res.model,
            "tokens_in": res.tokens_in,
            "tokens_out": res.tokens_out,
            "latency_s": round(res.latency_s, 3),
            "cached": res.cached,
            "cost_usd": res.cost_usd,
        }
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def summarize_costs(rows: list[dict], judge_model: str) -> dict[str, Any]:
    """Per-model totals plus an 'every call used the judge model' counterfactual."""
    per: dict[str, dict[str, float]] = {}
    for r in rows:
        m = per.setdefault(r["model"], {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        m["calls"] += 1
        m["tokens_in"] += r["tokens_in"]
        m["tokens_out"] += r["tokens_out"]
    total = 0.0
    known = True
    for model, m in per.items():
        c = cost_usd(model, int(m["tokens_in"]), int(m["tokens_out"]))
        m["cost_usd"] = c
        if c is None:
            known = False
        else:
            total += c
    tin = sum(int(m["tokens_in"]) for m in per.values())
    tout = sum(int(m["tokens_out"]) for m in per.values())
    return {
        "per_model": per,
        "total_usd": total if known else None,
        "all_judge_usd": cost_usd(judge_model, tin, tout),
    }
