"""Probe Nemotron Omni input formats on real clips.

For each clip (first N seconds), try every rung of the fallback ladder:
  A  native video (video_url data URI, audio track inside)
  B1 frames (1 fps, <=64) as image_url + input_audio (wav)
  B2 frames + audio_url (data URI)
  C  frames + separate audio-only narration call, transcript passed as text
Report: worked?, latency, tokens, cost, sample observations. Writes runs/omni_probe/.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from siteproof.llm import LLM, ROOT  # noqa: E402
from siteproof.models import model_for  # noqa: E402

OUT = ROOT / "runs" / "omni_probe"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

SYSTEM = (
    "You are a senior site inspection engineer verifying construction progress from a "
    'walkthrough video. The narrator states locations ("Floor 3, bay 2"). Use the narration '
    "to know where you are and the video to see what is built. For each element you can see "
    "clearly, report its stage. If an element is not clearly visible, report stage "
    '"not_visible". Never guess. Report timestamps in seconds from the start of this clip. '
    "Return ONLY valid JSON. No prose."
)
TASK = (
    "Elements: columns, beams, slab, blockwork, internal_plaster, external_plaster, "
    "electrical_conduits, plumbing_rough_in, door_frames, window_frames, "
    "toilet_waterproofing, flooring.\n"
    "Stages for columns/beams/slab: not_started, rebar_or_formwork, cast, "
    "deshuttered_complete. Others: not_started, in_progress, complete. Any: not_visible.\n"
    'Return {"narration": "<what the narrator said>", "observations": [{"t_start": s, '
    '"t_end": s, "narration_zone_hint": str, "element": str, "stage": str, '
    '"extent_note": str, "confidence": 0-1, "evidence_description": str}]}'
)


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def prep(clip: Path, seconds: int, tmp: Path) -> dict:
    vid = tmp / "probe.mp4"
    ff(
        "-i",
        str(clip),
        "-t",
        str(seconds),
        "-vf",
        "scale=-2:480",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(vid),
    )
    wav = tmp / "probe.wav"
    ff("-i", str(vid), "-vn", "-ac", "1", "-ar", "16000", str(wav))
    frames_dir = tmp / "frames"
    frames_dir.mkdir()
    ff(
        "-i",
        str(vid),
        "-vf",
        "fps=1,scale=-2:480",
        "-frames:v",
        "64",
        "-q:v",
        "5",
        str(frames_dir / "f_%03d.jpg"),
    )
    frames = sorted(frames_dir.glob("*.jpg"))
    return {"video": vid, "wav": wav, "frames": frames}


def frame_parts(frames: list[Path]) -> list[dict]:
    parts: list[dict] = []
    for i, f in enumerate(frames):
        parts.append({"type": "text", "text": f"t={i}s"})
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(f)}"}}
        )
    return parts


def build(fmt: str, a: dict, transcript: str | None = None) -> list[dict]:
    text = {"type": "text", "text": TASK}
    if fmt == "A":
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64(a['video'])}"}},
            text,
        ]
    elif fmt == "B1":
        content = [
            *frame_parts(a["frames"]),
            {"type": "input_audio", "input_audio": {"data": b64(a["wav"]), "format": "wav"}},
            text,
        ]
    elif fmt == "B2":
        content = [
            *frame_parts(a["frames"]),
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64(a['wav'])}"}},
            text,
        ]
    elif fmt == "C":
        content = [
            *frame_parts(a["frames"]),
            {"type": "text", "text": f"Narration transcript: {transcript}"},
            text,
        ]
    else:
        raise ValueError(fmt)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse_json(text: str) -> dict | None:
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json").strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i : j + 1])
    except json.JSONDecodeError:
        return None


def transcribe(llm: LLM, model: str, a: dict) -> tuple[str | None, str | None]:
    """Audio-only call for rung C. Tries input_audio then audio_url."""
    for part in (
        {"type": "input_audio", "input_audio": {"data": b64(a["wav"]), "format": "wav"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64(a['wav'])}"}},
    ):
        try:
            r = llm.chat(
                model,
                [
                    {
                        "role": "user",
                        "content": [
                            part,
                            {"type": "text", "text": "Transcribe the speech verbatim. Text only."},
                        ],
                    }
                ],
                stage="probe_transcribe",
                max_tokens=512,
                temperature=0,
            )
            return r.text.strip(), part["type"]
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:300]}"
    return None, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="clip paths (default: first 3 in data/footage/raw)")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--formats", default="A,B1,B2,C")
    args = ap.parse_args()

    clips = (
        [Path(c) for c in args.clips]
        if args.clips
        else sorted(
            p for p in (ROOT / "data/footage/raw").iterdir() if p.suffix.lower() in VIDEO_EXT
        )[:3]
    )
    if not clips:
        print("No clips found in data/footage/raw/. Add at least one walkthrough clip.")
        return 2

    model = model_for("perception")
    llm = LLM(run_id="omni_probe")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in clips:
        with tempfile.TemporaryDirectory() as td:
            a = prep(clip, args.seconds, Path(td))
            transcript = None
            for fmt in args.formats.split(","):
                row = {"clip": clip.name, "format": fmt, "ok": False}
                try:
                    if fmt == "C":
                        transcript, how = transcribe(llm, model, a)
                        row["transcribe_via"] = how
                        if transcript is None:
                            raise RuntimeError(f"audio-only call failed: {how}")
                    t0 = time.perf_counter()
                    r = llm.chat(
                        model,
                        build(fmt, a, transcript),
                        stage=f"probe_{fmt}",
                        max_tokens=4096,
                        temperature=0,
                    )
                    parsed = parse_json(r.text)
                    obs = (parsed or {}).get("observations") or []
                    row.update(
                        ok=True,
                        json_ok=parsed is not None,
                        latency_s=round(
                            r.latency_s if not r.cached else time.perf_counter() - t0, 2
                        ),
                        tokens_in=r.tokens_in,
                        tokens_out=r.tokens_out,
                        cost_usd=r.cost_usd,
                        n_obs=len(obs),
                        narration=(parsed or {}).get("narration"),
                        sample=obs[:4],
                        raw_head=None if parsed else r.text[:400],
                    )
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                results.append(row)
                status = "OK " if row["ok"] else "ERR"
                print(
                    f"[{status}] {clip.name} {fmt}: "
                    + (
                        f"{row.get('n_obs')} obs, {row.get('latency_s')}s, "
                        f"json={row.get('json_ok')}"
                        if row["ok"]
                        else row["error"][:160]
                    )
                )

    (OUT / "probe_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Omni probe report",
        "",
        f"Model: `{model}`, first {args.seconds}s per clip",
        "",
        "| clip | format | ok | json | obs | latency s | tokens in/out | cost $ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['clip']} | {r['format']} | {r['ok']} | {r.get('json_ok', '')} | "
            f"{r.get('n_obs', '')} | {r.get('latency_s', '')} | "
            f"{r.get('tokens_in', '')}/{r.get('tokens_out', '')} | {r.get('cost_usd')} |"
        )
    lines += ["", "## Samples and errors", ""]
    for r in results:
        lines.append(f"### {r['clip']} / {r['format']}")
        lines.append(
            "```json\n"
            + json.dumps(
                {k: r.get(k) for k in ("error", "narration", "sample", "raw_head") if r.get(k)},
                indent=2,
            )[:3000]
            + "\n```"
        )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
