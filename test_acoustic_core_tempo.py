"""The vendored acoustic core must refuse a BPM when there is no beat.

Before the salience/activity guards, `analyze_acoustic` named a tempo for any
input: 55.0 BPM for white noise and 184.6 BPM for a steady 440 Hz sine. Both
numbers are plausible tempos, which is what made the failure hard to see -- a
constant would have looked like an artifact.

These cases are written so they fail if either guard is removed:
  * the steady sine is caught ONLY by the absolute activity gate (its flux is
    numerical jitter, and normalising the autocorrelation makes that jitter
    look perfectly periodic),
  * the white noise is caught ONLY by the peak/salience test (it has plenty of
    onsets, just no periodic ones).
The click track guards the other direction: a real beat must still be found, so
the fix cannot be "refuse everything".
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "AI_Ears"))
from acoustic_core import analyze_acoustic  # noqa: E402

SR = 44100
DURATION_S = 10.0


def _write(tmp_path, name, x):
    p = tmp_path / name
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(p)


def _t():
    return np.arange(int(SR * DURATION_S)) / SR


def test_white_noise_refuses_bpm(tmp_path):
    """Onsets everywhere, none periodic -- the salience test must reject."""
    rng = np.random.default_rng(20260812)
    path = _write(tmp_path, "noise.wav", rng.normal(0, 0.2, len(_t())))
    assert analyze_acoustic(path)["tempo_bpm"] is None


def test_steady_tone_refuses_bpm(tmp_path):
    """No onsets at all -- the absolute activity gate must reject.

    This one previously survived a peak/salience guard, because dividing the
    autocorrelation by ac[0] rescales numerical jitter into a clean peak.
    """
    path = _write(tmp_path, "sine.wav", 0.3 * np.sin(2 * np.pi * 440 * _t()))
    assert analyze_acoustic(path)["tempo_bpm"] is None


def test_real_click_track_still_reports_bpm(tmp_path):
    """The guards must not simply refuse everything."""
    t = _t()
    click = np.zeros(len(t))
    burst = np.exp(-np.arange(2000) / 200.0) * np.sin(2 * np.pi * 1200 * np.arange(2000) / SR)
    for beat in np.arange(0, DURATION_S, 0.5):          # 120 BPM
        i = int(beat * SR)
        click[i:i + 2000] += 0.8 * burst
    bpm = analyze_acoustic(_write(tmp_path, "click.wav", click))["tempo_bpm"]
    assert bpm is not None
    assert bpm == pytest.approx(120.0, abs=2.0)


def test_silence_refuses_bpm(tmp_path):
    """Degenerate input must not divide its way into a tempo."""
    path = _write(tmp_path, "silence.wav", np.zeros(len(_t())))
    assert analyze_acoustic(path)["tempo_bpm"] is None
