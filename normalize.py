#!/usr/bin/env python3
"""
normalize.py — the relative "ear" layer for Seven Ears.

Seven Ears' acoustic core reports ABSOLUTE features against fixed thresholds —
the meter. This adds the EAR: a per-speaker rolling baseline, so a clip can be
described RELATIVE to that speaker's own recent norm — "lower/warmer than usual
for them," "softer than usual." It's the auditory-contrast layer: human hearing
judges a voice against the recent context, not against absolute Hz.

Feature choices (each learned the hard way against a listener's ear):
  - "lower/warmer vs higher" -> PITCH MEDIAN (the note the voice sounds at), NOT
                           spectral centroid. A low, breathy line can read
                           centroid-bright while a listener clearly hears it low;
                           pitch median tracks what the ear tracks.
  - "flatter vs more melodic" -> PITCH CONTOUR (contour_swing: slow-arc magnitude
                           over the phrase), NOT plain range/travel, which both
                           miss a smooth rise-and-fall bloom.
  - "softer vs more forward" -> loudness (loud_dbfs).

Centering: relative() uses MEDIAN + scaled MAD, not mean + stdev, so a handful of
loud/excited clips can't drag "usual" — a calm clip should not read "low" just
because the recent window happened to hold some shouted ones.

Known seam (intentionally not solved here): "melodic" (a sing-song contour shape)
is NOT the same axis as "varied" (the amount of pitch movement). A clip can sit
parked at one pitch yet carry a sing-song shape; contour_swing still conflates the
two. Splitting them is a perceptual-model problem left for future work.

Boundary (matches upstream ethos): a relative delta is EVIDENCE ("relative to
their recent notes"), never a claim about how the speaker feels.

Usage: python3 normalize.py [--reset] [--speaker NAME] clip1.mp3 clip2.wav ...
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor" / "AI_Ears"))
from acoustic_core import analyze_acoustic, read_wav_mono  # noqa: E402

FEATURES = ["pitch_median_hz", "pitch_contour_hz", "loud_dbfs"]
WINDOW = 20      # rolling "recent context" length
Z = 0.7          # how far from usual before we name it


def to_wav(src: Path) -> Path:
    if src.suffix.lower() == ".wav":
        return src
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "44100",
         "-sample_fmt", "s16", str(out)],
        check=True, capture_output=True,
    )
    return out


def estimate_pitch(x: np.ndarray, sr: int) -> dict:
    """Autocorrelation F0 over 70-400 Hz. Same method as seven_ears_card.estimate_pitch,
    plus octave-error correction so a doubled/halved frame can't poison the baseline."""
    frame, hop = int(sr * 0.04), int(sr * 0.02)
    fmin, fmax = 70.0, 400.0
    min_lag, max_lag = int(sr / fmax), int(sr / fmin)
    pitches = []
    for start in range(0, max(1, len(x) - frame), hop):
        seg = x[start:start + frame]
        if len(seg) < frame:
            continue
        seg = seg - np.mean(seg)
        if float(np.sqrt(np.mean(seg * seg))) < 0.003:
            continue
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        if len(corr) <= max_lag or corr[0] <= 0:
            continue
        lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
        if corr[lag] / corr[0] < 0.25:
            continue
        hz = sr / lag
        if fmin <= hz <= fmax:
            pitches.append(float(hz))
    if not pitches:
        return {"available": False}
    # octave-error correction: snap each frame to the octave of {p/4..p*4} nearest its
    # LOCAL median — kills doublings (a low tail read at 2x) without flattening real range.
    corrected: list[float] = []
    for p in pitches:
        ref = float(np.median(corrected[-8:])) if corrected else float(np.median(pitches))
        cands = [c for c in (p / 4, p / 2, p, p * 2, p * 4) if fmin <= c <= fmax]
        corrected.append(min(cands, key=lambda c: abs(np.log2(c / ref))) if cands else p)
    arr = np.array(corrected)
    return {
        "available": True,
        "median_hz": round(float(np.median(arr)), 1),
        "p10_hz": round(float(np.percentile(arr, 10)), 1),
        "p90_hz": round(float(np.percentile(arr, 90)), 1),
        # travel = mean frame-to-frame |Δpitch|: catches wiggle/inflection (an up-down-up
        # "bloom") that plain range (p90-p10) misses — a smooth slope and a bloom can share
        # a range but not a travel.
        "travel_hz": round(float(np.mean(np.abs(np.diff(arr)))), 1) if len(arr) >= 2 else 0.0,
        "track": [round(float(v), 1) for v in arr],  # per-frame, time-ordered (for contour)
    }


def contour_swing(track: list[float]) -> float:
    """Slow-arc magnitude: de-octave, smooth, then the range of time-slice medians.
    Catches a slow melodic shape (rise/fall over a phrase) that frame-to-frame travel
    misses, and is robust to octave-error spikes that inflate plain range. A phrase
    whose pitch arcs (e.g. 134->163->...->93) is flattened by both range and travel but
    shows up here."""
    if len(track) < 4:
        return 0.0
    a = np.array(track, dtype=float)
    med = float(np.median(a))
    a = np.where(a > 1.7 * med, a / 2.0, a)  # crude octave-error fix (doublings are ~2x)
    if len(a) >= 5:                          # median-smooth, window 5
        a = np.array([float(np.median(a[max(0, i - 2):i + 3])) for i in range(len(a))])
    meds = [float(np.median(s)) for s in np.array_split(a, 8) if len(s)]
    return round(max(meds) - min(meds), 1)


def features_of(path: Path) -> dict:
    wav = to_wav(path)
    a = analyze_acoustic(str(wav))
    x, sr = read_wav_mono(str(wav))
    p = estimate_pitch(x, sr)
    if wav != path:
        wav.unlink(missing_ok=True)
    if p.get("available"):
        a["pitch_median_hz"] = p["median_hz"]
        a["pitch_range_hz"] = round(p["p90_hz"] - p["p10_hz"], 1)
        a["pitch_travel_hz"] = p["travel_hz"]
        a["pitch_contour_hz"] = contour_swing(p["track"])
    else:
        a["pitch_median_hz"] = a["pitch_range_hz"] = a["pitch_travel_hz"] = a["pitch_contour_hz"] = None
    return a


class SpeakerBaseline:
    def __init__(self, store: Path, speaker: str):
        self.store, self.speaker = store, speaker
        self.data = json.loads(store.read_text()) if store.exists() else {}
        self.win = self.data.get(speaker, {f: [] for f in FEATURES})

    def relative(self, feat: str, val):
        # Robust centering: MEDIAN + scaled MAD, not mean + stdev. A few loud/excited
        # clips must not drag "usual" — otherwise a calm clip reads "low" only because
        # the recent window held some shouted ones. Median sits on the typical register;
        # MAD*1.4826 ~ stdev for normal data, so Z keeps its meaning. delta = distance
        # from typical, the honest "N Hz from usual."
        vals = self.win.get(feat, [])
        if val is None or len(vals) < 3:
            return None
        med = statistics.median(vals)
        spread = statistics.median([abs(v - med) for v in vals]) * 1.4826
        if spread < 1e-6:                       # degenerate window (MAD collapses) -> stdev
            spread = statistics.pstdev(vals) or 1e-6
        return (val - med) / spread, val - med

    def update(self, a: dict):
        for f in FEATURES:
            if a.get(f) is not None:
                self.win.setdefault(f, []).append(round(float(a[f]), 2))
                self.win[f] = self.win[f][-WINDOW:]
        self.data[self.speaker] = self.win
        self.store.write_text(json.dumps(self.data, indent=2))

    def n(self) -> int:
        return len(self.win.get("pitch_median_hz", []))


def phrase(feat: str, z: float, delta: float) -> str | None:
    warm, hot = z < -Z, z > Z
    if feat == "pitch_median_hz":
        if warm: return f"lower/warmer than usual (~{abs(round(delta))} Hz down in pitch)"
        if hot:  return f"higher-pitched than usual (~{round(delta)} Hz up)"
    elif feat == "pitch_contour_hz":
        if warm: return "flatter melody than usual (less rise-and-fall)"
        if hot:  return "more melodic / bigger pitch arc than usual"
    elif feat == "loud_dbfs":
        if warm: return "softer than usual"
        if hot:  return "louder / more forward than usual"
    return None


def read_clip(path: Path, base: SpeakerBaseline) -> str:
    a = features_of(path)
    meter = (f'pitch {a["pitch_median_hz"]} Hz (arc {a["pitch_contour_hz"]}, '
             f'swing {a["pitch_range_hz"]}, move {a["pitch_travel_hz"]}) · {a["loud_dbfs"]} dBFS')
    rels = []
    for f in FEATURES:
        r = base.relative(f, a.get(f))
        if r and (p := phrase(f, *r)):
            rels.append(p)
    n_before = base.n()
    base.update(a)
    if rels:
        ear = f"for {base.speaker}: " + "; ".join(rels)
    elif n_before < 3:
        ear = f"(building baseline — {n_before + 1}/3 clips)"
    else:
        ear = f"for {base.speaker}: right around usual"
    return f"{path.name}\n  meter: {meter}\n  ear:   {ear}"


def ear_reads(path, speaker: str = "speaker", store=None) -> str | None:
    """The relative 'ear' line for one clip, given a persisted per-speaker baseline.
    Returns e.g. 'lower/warmer than usual; softer than usual', or None while the
    baseline is still building (<3 clips). This is the card-facing entry point."""
    store = Path(store) if store else HERE / "speaker_baselines.json"
    base = SpeakerBaseline(store, speaker)
    a = features_of(Path(path))
    rels = []
    for f in FEATURES:
        r = base.relative(f, a.get(f))
        if r and (p := phrase(f, *r)):
            rels.append(p)
    n_before = base.n()
    base.update(a)
    if rels:
        return "; ".join(rels)
    return None if n_before < 3 else "right around usual"


def main() -> None:
    args = sys.argv[1:]
    store = HERE / "speaker_baselines.json"
    if "--reset" in args:
        args.remove("--reset")
        store.unlink(missing_ok=True)
    speaker = "speaker"
    if "--speaker" in args:
        i = args.index("--speaker")
        speaker = args[i + 1]
        del args[i:i + 2]
    base = SpeakerBaseline(store, speaker)
    for p in args:
        print(read_clip(Path(p), base), "\n")


if __name__ == "__main__":
    main()
