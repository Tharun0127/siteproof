"""Single source of truth for model IDs. Override any role via env."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULTS: dict[str, str] = {
    "perception": "nvidia/Nemotron-3-Nano-Omni",
    "fast": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "reader": "nvidia/nemotron-3-super-120b-a12b",
    "reader_vl": "nvidia/Nemotron-3.5-Super-VL",
    "judge": "nvidia/Nemotron-3-Ultra-550b-a55b",
}

# Roles the pipeline cannot run without. reader_vl is optional (scanned bills only).
REQUIRED_ROLES = ("perception", "fast", "reader", "judge")


def model_for(role: str) -> str:
    return os.getenv(f"SITEPROOF_MODEL_{role.upper()}", DEFAULTS[role])


def all_models() -> dict[str, str]:
    return {role: model_for(role) for role in DEFAULTS}


@dataclass
class ModelCheck:
    available: list[str]
    missing: dict[str, str]  # role -> configured id not served


class MissingModelError(RuntimeError):
    pass


def check_models(client, strict: bool = True) -> ModelCheck:
    """Call GET /models, print what is served, fail loudly if a required ID is missing."""
    available = sorted(m.id for m in client.models.list().data)
    print(f"Token Factory serves {len(available)} models:")
    for mid in available:
        print(f"  {mid}")
    lower = {m.lower(): m for m in available}
    missing: dict[str, str] = {}
    for role, mid in all_models().items():
        if mid in available:
            continue
        # Case-only mismatch is common in docs; report the real ID.
        if mid.lower() in lower:
            missing[role] = f"{mid} (served as {lower[mid.lower()]})"
        else:
            missing[role] = mid
    hard = {r: m for r, m in missing.items() if r in REQUIRED_ROLES}
    if hard and strict:
        lines = "\n".join(f"  {r}: {m}" for r, m in hard.items())
        raise MissingModelError(
            "Configured model IDs not served by Token Factory:\n"
            f"{lines}\nSet SITEPROOF_MODEL_<ROLE> in .env to a served ID."
        )
    return ModelCheck(available=available, missing=missing)
