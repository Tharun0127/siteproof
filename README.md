# SiteProof

**The bill says done. The video says otherwise.**

SiteProof checks a contractor's monthly running account (RA) bill against a 2-minute phone video of the site and tells the developer exactly what to pay, what to hold, and shows the video evidence for every hold.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- Demo: {{DEMO_URL}}
- Video: {{VIDEO_URL}}

## The problem

_To be written._

## How it works

_Architecture diagram and pipeline description to come._

## Models

| Role | Model | Why |
|---|---|---|
| Perception | `nvidia/Nemotron-3-Nano-Omni` | Joint audio and video reasoning: narration says where, video shows what |
| Fast mapping | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | High-volume short classifications |
| Bill reader | `nvidia/nemotron-3-super-120b-a12b` | Long-context structured extraction |
| Scanned pages | `nvidia/Nemotron-3.5-Super-VL` | Tables in scanned bill images |
| Judge | `nvidia/Nemotron-3-Ultra-550b-a55b` | Final certification memo |
| Python | n/a | Every rupee and quantity is computed deterministically |

## Where Token Factory accelerated our workflow

_To be written._

## Results

_Filled from `eval_results.json`._

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # Windows; use .venv/bin on macOS/Linux
cp .env.example .env                    # add TOKEN_FACTORY_API_KEY and TAVILY_API_KEY
python scripts/smoke.py                 # list models, one call per role
```

## Glossary

_To be written._

## Data provenance

_To be written._

## Limitations

_To be written._

## Built during the submission period

All code in this repository is new and was written during the hackathon submission period, starting 28 Oct 2026.

## License

MIT, see [LICENSE](LICENSE).
