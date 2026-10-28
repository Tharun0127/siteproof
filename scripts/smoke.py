"""Phase 0 smoke test: list models, one tiny call per role, Tavily ping."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from siteproof.llm import LLM  # noqa: E402
from siteproof.models import all_models, check_models  # noqa: E402


def main() -> int:
    llm = LLM(run_id="smoke")
    check = check_models(llm.client, strict=False)
    if check.missing:
        print("\nMISSING configured IDs:")
        for role, mid in check.missing.items():
            print(f"  {role}: {mid}")

    print("\nrole        model                                       ok  latency  in/out   cost$")
    failures = 0
    for role, model in all_models().items():
        if role in check.missing:
            print(f"{role:<11} {model:<43} --  skipped (not served)")
            continue
        try:
            r = llm.chat(
                model,
                [{"role": "user", "content": "Reply with exactly: OK"}],
                stage="smoke",
                max_tokens=256,
                temperature=0,
            )
            cost = "n/a" if r.cost_usd is None else f"{r.cost_usd:.6f}"
            print(
                f"{role:<11} {model:<43} OK  {r.latency_s:6.2f}s  "
                f"{r.tokens_in}/{r.tokens_out}  {cost}  {'(cached)' if r.cached else ''}"
            )
        except Exception as e:  # report and keep going
            failures += 1
            print(f"{role:<11} {model:<43} ERR {type(e).__name__}: {str(e)[:200]}")

    if os.getenv("TAVILY_API_KEY"):
        from tavily import TavilyClient

        try:
            res = TavilyClient(os.environ["TAVILY_API_KEY"]).search(
                "Chennai rainfall October 2026", max_results=2
            )
            print(f"\ntavily      OK  {len(res.get('results', []))} results")
        except Exception as e:
            failures += 1
            print(f"\ntavily      ERR {type(e).__name__}: {str(e)[:200]}")
    else:
        print("\ntavily      skipped (TAVILY_API_KEY not set)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
