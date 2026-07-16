#!/usr/bin/env python3
"""Generate a simple Seven Ears visual card image from an audio file.

This is deliberately boring: ffmpeg-only plots, no public dashboard, no web server.
It produces a consistent PNG attachment for chat review.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffmpeg_plot(audio: Path, output: Path, title: str, kind: str, size: str) -> None:
    if kind == "wave":
        lavfi = f"showwavespic=s={size}:colors=#1f77b4,format=rgba,drawtext=text='{title}':x=(w-text_w)/2:y=12:fontsize=24:fontcolor=black:box=1:boxcolor=white@0.75"
    elif kind == "spectrum":
        lavfi = f"showspectrumpic=s={size}:mode=combined:scale=log:legend=0:color=viridis,format=rgba,drawtext=text='{title}':x=(w-text_w)/2:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45"
    else:
        raise ValueError(kind)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio),
        "-lavfi", lavfi,
        "-frames:v", "1",
        str(output),
    ])


def combine(top: Path, bottom: Path, output: Path) -> None:
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(top), "-i", str(bottom),
        "-filter_complex", "[0:v][1:v]vstack=inputs=2,format=rgba",
        "-frames:v", "1",
        str(output),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="Seven Ears Visual Card")
    args = ap.parse_args()

    audio = args.audio.expanduser().resolve()
    if not audio.exists():
        raise SystemExit(f"audio not found: {audio}")

    out = args.out or audio.with_suffix(".seven-ears.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="seven-ears-visual-") as td:
        td_path = Path(td)
        wave = td_path / "wave.png"
        spec = td_path / "spectrum.png"
        ffmpeg_plot(audio, wave, "Waveform", "wave", "1200x260")
        ffmpeg_plot(audio, spec, "Spectrogram", "spectrum", "1200x520")
        combine(wave, spec, out)

    print(out)


if __name__ == "__main__":
    main()
