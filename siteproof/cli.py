"""`siteproof` command line."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="siteproof")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list served models and verify configured IDs")
    args = ap.parse_args(argv)

    if args.cmd == "models":
        from siteproof.llm import make_client
        from siteproof.models import check_models

        check_models(make_client())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
