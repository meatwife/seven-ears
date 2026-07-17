#!/usr/bin/env python3
"""Seven Ears: Music Edition — experimental first slice.

Sibling of seven_ears_card.py (Spoken Voice Edition). Music gets its own
listening organ instead of being squeezed through the speech profile.

This slice only reports what the signal gives evidence for:
- time-body: energy envelope, sections, silence map (Cameron-style)
- color: spectral brightness/band balance/tonality (Lux-style)
- pulse: onsets and tempo, only when the pulse is actually stable
- dynamics: loudness range and crest factor

It deliberately does NOT claim: emotion, genre, musical key, melody,
lyrics, or vocal presence. See MUSIC_EDITION_NOTES.md for why.

No network. numpy + stdlib only; ffmpeg only needed for non-WAV input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ANALYSIS_SR = 22050

# What this slice measures vs. refuses to guess. Kept in the output so the
# card can never silently outgrow its evidence.
UNSUPPORTED = {
    'vocal_presence': 'needs source separation or a trained model; band-energy heuristics confabulate on instruments',
    'emotion': 'not measurable from the signal; interpretation belongs to the listener',
    'genre': 'cultural category, not an acoustic quantity',
    'key_and_melody': 'deferred to a later slice (chroma analysis); not pretended in the meantime',
    'lyrics': 'out of scope; Music Edition does not transcribe',
}


# ---------------------------------------------------------------------------
# Loading

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f'expected 16-bit wav, got sample width {width}')
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr


def load_audio(src: Path, sr: int = ANALYSIS_SR) -> tuple[np.ndarray, int]:
    """Load any audio file to mono float. WAV reads directly; everything else
    goes through ffmpeg so tests and synthetic fixtures never need it."""
    if src.suffix.lower() == '.wav':
        x, native_sr = read_wav_mono(src)
        if native_sr == sr:
            return x, sr
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH and input is not analysis-rate WAV')
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / 'music.wav'
        run(['ffmpeg', '-y', '-v', 'error', '-i', str(src), '-ar', str(sr), '-ac', '1', '-c:a', 'pcm_s16le', str(dst)])
        return read_wav_mono(dst)


# ---------------------------------------------------------------------------
# Shared frame math

def stft_mag(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram: (frames, bins), bin freqs, frame times."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    window = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    mags = np.empty((n_frames, n_fft // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop:i * hop + n_fft] * window
        mags[i] = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = np.arange(n_frames) * hop / sr
    return mags, freqs, times


def energy_envelope(x: np.ndarray, sr: int, win_s: float = 0.5, hop_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """RMS energy in dBFS over coarse windows — the piece's time-body."""
    win = max(1, int(sr * win_s))
    hop = max(1, int(sr * hop_s))
    times, vals = [], []
    for start in range(0, max(1, len(x) - win + 1), hop):
        seg = x[start:start + win]
        vals.append(20 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9))
        times.append(start / sr)
    return np.array(times), np.array(vals)


# ---------------------------------------------------------------------------
# Time-body: dynamics, silence, sections

def dynamics_profile(env_db: np.ndarray, x: np.ndarray) -> dict:
    """Loudness spread of the non-silent body plus crest factor."""
    active = env_db[env_db > env_db.max() - 40]
    if len(active) == 0:
        return {'available': False}
    spread = float(np.percentile(active, 95) - np.percentile(active, 10))
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    crest_db = float(20 * np.log10((np.max(np.abs(x)) + 1e-9) / rms))
    if spread < 6:
        label = 'steady / compressed'
    elif spread < 15:
        label = 'moderate movement'
    else:
        label = 'wide dynamics'
    return {
        'available': True,
        'loudness_spread_db': round(spread, 1),
        'crest_factor_db': round(crest_db, 1),
        'peak_dbfs': round(float(env_db.max()), 1),
        'label': label,
    }


def silence_map(env_times: np.ndarray, env_db: np.ndarray, rel_floor_db: float = 40.0, min_hold_s: float = 1.0) -> dict:
    """Leading/trailing quiet and held internal drops, relative to the peak."""
    if len(env_db) == 0:
        return {'available': False}
    quiet = env_db < env_db.max() - rel_floor_db
    hop = float(env_times[1] - env_times[0]) if len(env_times) > 1 else 0.25
    total = float(env_times[-1] + hop)

    runs: list[tuple[float, float]] = []
    start = None
    for t, q in zip(env_times, quiet):
        if q and start is None:
            start = float(t)
        if not q and start is not None:
            runs.append((start, float(t)))
            start = None
    if start is not None:
        runs.append((start, total))

    leading = trailing = 0.0
    internal = []
    for s, e in runs:
        if s == 0.0:
            leading = e
        elif abs(e - total) < hop:
            trailing = e - s
        elif e - s >= min_hold_s:
            internal.append({'start_s': round(s, 2), 'dur_s': round(e - s, 2)})
    return {
        'available': True,
        'leading_quiet_s': round(leading, 2),
        'trailing_quiet_s': round(trailing, 2),
        'internal_drops': internal,
    }


def section_features(mags: np.ndarray, freqs: np.ndarray, times: np.ndarray, win_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-second feature vectors: log energy, band balance, centroid."""
    if len(times) < 2:
        return np.empty((0, 5)), np.array([])
    hop = times[1] - times[0]
    per_win = max(1, int(round(win_s / hop)))
    power = mags ** 2
    low = freqs < 250
    mid = (freqs >= 250) & (freqs < 2000)
    high = freqs >= 4000
    rows, row_times = [], []
    for i in range(0, len(times) - per_win + 1, per_win):
        p = power[i:i + per_win].mean(axis=0)
        total = p.sum() + 1e-12
        centroid = float((freqs * p).sum() / total)
        rows.append([
            float(np.log10(total)),
            float(p[low].sum() / total),
            float(p[mid].sum() / total),
            float(p[high].sum() / total),
            centroid / (freqs[-1] + 1e-9),
        ])
        row_times.append(float(times[i]))
    return np.array(rows), np.array(row_times)


def detect_sections(feats: np.ndarray, feat_times: np.ndarray, duration_s: float) -> list[dict]:
    """Novelty-based section boundaries: where does the piece change body?

    Compares the mean feature vector before vs after each point. Returns the
    section list (always at least one) with start times.
    """
    boundaries: list[float] = []
    n = len(feats)
    context = max(3, min(8, n // 5))
    min_gap_s = max(5.0, duration_s / 8)
    if n >= 2 * context + 1:
        z = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
        novelty = np.zeros(n)
        for i in range(context, n - context):
            novelty[i] = float(np.linalg.norm(z[i:i + context].mean(axis=0) - z[i - context:i].mean(axis=0)))
        body = novelty[context:n - context]
        # Absolute floor: z-scoring a near-constant feature amplifies numerical
        # jitter to O(1); a real section change moves several features by
        # multiple standard deviations, so demand clearly more than jitter.
        threshold = max(body.mean() + body.std(), 2.0)
        # The first and last computable points sit next to zero-padded novelty,
        # so they pass the local-maximum test almost for free; every song was
        # getting a phantom boundary at t == context. Only accept peaks whose
        # neighbors were both actually computed.
        for i in range(context + 1, n - context - 1):
            if novelty[i] <= threshold:
                continue
            if novelty[i] < novelty[i - 1] or novelty[i] < novelty[i + 1]:
                continue
            t = float(feat_times[i])
            if boundaries and t - boundaries[-1] < min_gap_s:
                continue
            boundaries.append(t)

    starts = [0.0] + boundaries
    ends = boundaries + [duration_s]
    return [
        {'start_s': round(s, 2), 'end_s': round(e, 2), 'dur_s': round(e - s, 2)}
        for s, e in zip(starts, ends)
    ]


# ---------------------------------------------------------------------------
# Color: brightness, band balance, tonality

def spectral_color(mags: np.ndarray, freqs: np.ndarray, env_db: np.ndarray | None = None) -> dict:
    power = mags ** 2
    frame_energy = power.sum(axis=1)
    keep = frame_energy > frame_energy.max() * 1e-4
    if not keep.any():
        return {'available': False}
    power = power[keep]
    frame_energy = frame_energy[keep]

    centroids = (power * freqs).sum(axis=1) / (frame_energy + 1e-12)
    cum = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)
    rolloffs = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)

    total = power.sum(axis=0)
    grand = total.sum() + 1e-12
    bands = {
        'low_pct': round(float(total[freqs < 250].sum() / grand * 100), 1),
        'mid_pct': round(float(total[(freqs >= 250) & (freqs < 2000)].sum() / grand * 100), 1),
        'upper_mid_pct': round(float(total[(freqs >= 2000) & (freqs < 4000)].sum() / grand * 100), 1),
        'high_pct': round(float(total[freqs >= 4000].sum() / grand * 100), 1),
    }
    centroid_med = float(np.median(centroids))
    if centroid_med < 1200:
        brightness = 'dark / warm'
    elif centroid_med < 2500:
        brightness = 'mid-balanced'
    else:
        brightness = 'bright'
    flat_med = float(np.median(flatness))
    if flat_med < 0.05:
        tonality = 'strongly tonal (pitched material dominates)'
    elif flat_med < 0.3:
        tonality = 'mixed tonal and noisy'
    else:
        tonality = 'noise-leaning (percussive or textural)'
    return {
        'available': True,
        'centroid_median_hz': round(centroid_med),
        'centroid_p10_hz': round(float(np.percentile(centroids, 10))),
        'centroid_p90_hz': round(float(np.percentile(centroids, 90))),
        'rolloff85_median_hz': round(float(np.median(rolloffs))),
        'flatness_median': round(flat_med, 4),
        'band_balance': bands,
        'brightness_label': brightness,
        'tonality_label': tonality,
    }


# ---------------------------------------------------------------------------
# Pulse: onsets and tempo

def onset_envelope(mags: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux, lightly compressed."""
    logm = np.log1p(mags)
    flux = np.diff(logm, axis=0)
    flux[flux < 0] = 0
    return flux.sum(axis=1)


def detect_onsets(flux: np.ndarray, times: np.ndarray, min_sep_s: float = 0.1) -> list[float]:
    if len(flux) < 3:
        return []
    threshold = flux.mean() + 1.5 * flux.std()
    onsets: list[float] = []
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            t = float(times[i + 1])
            if onsets and t - onsets[-1] < min_sep_s:
                continue
            onsets.append(round(t, 3))
    return onsets


def estimate_tempo(flux: np.ndarray, frame_rate: float, bpm_min: float = 60.0, bpm_max: float = 200.0) -> dict:
    """Autocorrelation tempo estimate that refuses to guess.

    Returns bpm only when the pulse is periodic enough to support it, so a
    drone or field recording gets honesty instead of a confabulated BPM.
    Octave ambiguity (reporting half/double tempo) is a known limitation.
    """
    result: dict = {'supported': False, 'reason': None, 'bpm': None, 'confidence': None}
    if len(flux) < int(frame_rate * 4):
        result['reason'] = 'clip too short for a tempo estimate (need ~4s of frames)'
        return result
    # Absolute activity gate. Callers amplitude-normalize the spectrogram, so
    # a steady tone's numerical jitter sits far below 1.0 while real onsets in
    # audible material sit far above it.
    if flux.max() < 1.0:
        result['reason'] = 'no onset activity; nothing pulse-like to measure'
        return result
    env = flux - flux.mean()
    ac = np.correlate(env, env, mode='full')[len(env) - 1:]
    if ac[0] <= 0:
        result['reason'] = 'degenerate onset envelope'
        return result
    ac = ac / ac[0]
    min_lag = max(1, int(frame_rate * 60.0 / bpm_max))
    max_lag = min(len(ac) - 1, int(frame_rate * 60.0 / bpm_min))
    if max_lag <= min_lag:
        result['reason'] = 'clip too short for the tempo search range'
        return result
    window = ac[min_lag:max_lag + 1]
    best = int(np.argmax(window)) + min_lag
    # Octave correction: a beat period that falls between frames smears its
    # autocorrelation peak, letting the 2x lag win. If half (or a third of)
    # the lag is also a strong peak, prefer the faster tempo.
    for div in (2, 3):
        lo = int(np.floor(best / div)) - 1
        hi = int(np.ceil(best / div)) + 1
        if lo < min_lag:
            continue
        neighborhood = ac[lo:hi + 1]
        if neighborhood.max() >= 0.6 * ac[best]:
            best = int(np.argmax(neighborhood)) + lo
            break
    confidence = float(ac[best] - np.median(window))
    result['confidence'] = round(confidence, 3)
    if ac[best] < 0.15 or confidence < 0.1:
        result['reason'] = 'no stable pulse: onset pattern is not periodic enough to name a BPM'
        return result
    result['supported'] = True
    result['bpm'] = round(60.0 * frame_rate / best, 1)
    result['note'] = 'autocorrelation estimate; half/double-tempo ambiguity possible'
    return result


# ---------------------------------------------------------------------------
# Full analysis

def analyze_signal(x: np.ndarray, sr: int, source_name: str = 'signal') -> dict:
    duration_s = len(x) / sr
    env_times, env_db = energy_envelope(x, sr)
    mags, freqs, frame_times = stft_mag(x, sr)
    # Amplitude-normalize the onset path so estimate_tempo's absolute activity
    # gate means the same thing for quiet and loud recordings.
    flux = onset_envelope(mags / (float(np.abs(x).max()) + 1e-9))
    frame_rate = sr / 512
    # Below the activity gate, "onsets" are just numerical jitter peaks.
    onsets = detect_onsets(flux, frame_times) if flux.max() >= 1.0 else []
    feats, feat_times = section_features(mags, freqs, frame_times)
    return {
        'edition': 'music',
        'file': source_name,
        'duration_s': round(duration_s, 2),
        'analysis_sr': sr,
        'sections': detect_sections(feats, feat_times, duration_s),
        'dynamics': dynamics_profile(env_db, x),
        'silence': silence_map(env_times, env_db),
        'color': spectral_color(mags, freqs),
        'pulse': {
            'onset_count': len(onsets),
            'onset_density_per_s': round(len(onsets) / duration_s, 2) if duration_s > 0 else 0.0,
            'tempo': estimate_tempo(flux, frame_rate),
        },
        'not_measured': UNSUPPORTED,
    }


def analyze_file(path: Path) -> dict:
    x, sr = load_audio(path)
    return analyze_signal(x, sr, source_name=path.name)


# ---------------------------------------------------------------------------
# Card

def format_music_card(result: dict) -> str:
    lines = [f"Seven Ears, music card — {result['file']}"]
    lines.append(f"Length {result['duration_s']:.1f}s, analyzed at {result['analysis_sr']} Hz mono.")

    sections = result['sections']
    if len(sections) > 1:
        marks = ', '.join(f"{s['start_s']:.0f}s" for s in sections[1:])
        lines.append(f"Body: {len(sections)} sections; the sound changes shape around {marks}.")
    else:
        lines.append('Body: one continuous section; no strong shape-change detected.')

    dyn = result['dynamics']
    if dyn.get('available'):
        lines.append(
            f"Dynamics: {dyn['label']} — loudness spread {dyn['loudness_spread_db']} dB, "
            f"crest factor {dyn['crest_factor_db']} dB."
        )

    sil = result['silence']
    if sil.get('available'):
        bits = []
        if sil['leading_quiet_s'] >= 0.5:
            bits.append(f"{sil['leading_quiet_s']:.1f}s of quiet before it starts")
        if sil['trailing_quiet_s'] >= 0.5:
            bits.append(f"{sil['trailing_quiet_s']:.1f}s after it ends")
        for d in sil['internal_drops']:
            bits.append(f"a {d['dur_s']:.1f}s drop at {d['start_s']:.0f}s")
        if bits:
            lines.append('Silence: ' + '; '.join(bits) + '.')

    color = result['color']
    if color.get('available'):
        b = color['band_balance']
        lines.append(
            f"Color: {color['brightness_label']} (centroid ~{color['centroid_median_hz']} Hz); "
            f"{color['tonality_label']}."
        )
        lines.append(
            f"Band balance: {b['low_pct']}% low, {b['mid_pct']}% mid, "
            f"{b['upper_mid_pct']}% upper-mid, {b['high_pct']}% high."
        )

    pulse = result['pulse']
    tempo = pulse['tempo']
    if tempo['supported']:
        lines.append(
            f"Pulse: ~{tempo['bpm']} BPM (confidence {tempo['confidence']}), "
            f"{pulse['onset_count']} onsets ({pulse['onset_density_per_s']}/s). "
            f"{tempo['note']}."
        )
    else:
        lines.append(f"Pulse: {pulse['onset_count']} onsets; {tempo['reason']}.")

    lines.append(
        'No mind-reading: this card measures the sound, not the feeling. '
        'Emotion, genre, vocals, and melody are yours to hear, not mine to fake.'
    )
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Seven Ears: Music Edition (experimental slice)')
    parser.add_argument('audio', type=Path, help='path to a local audio file')
    parser.add_argument('--json', action='store_true', help='print raw JSON instead of the card')
    args = parser.parse_args()
    if not args.audio.exists():
        print(f'error: {args.audio} not found', file=sys.stderr)
        return 1
    result = analyze_file(args.audio)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_music_card(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
