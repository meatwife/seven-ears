"""Unit tests for the ear layer (normalize.py).

Covers the deterministic logic without touching ffmpeg or audio files:
  - estimate_pitch on synthetic sines (fundamental picked, not an octave)
  - contour_swing (flat vs arc, octave-spike robustness, short-track guard)
  - SpeakerBaseline (the <3-sample gate, z/delta math, window cap, persistence)
  - phrase (both directions per feature + the deadband)

The audio path (features_of / ear_reads) is exercised end-to-end by a real
seven_ears_card run, so it's intentionally not re-mocked here.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from normalize import (
    SpeakerBaseline,
    WINDOW,
    Z,
    contour_swing,
    estimate_pitch,
    phrase,
)

SR = 44100


def sine(freq: float, dur_s: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * dur_s)) / SR
    return amp * np.sin(2 * math.pi * freq * t)


class EstimatePitchTests(unittest.TestCase):
    def test_clean_sine_reads_its_fundamental(self):
        p = estimate_pitch(sine(120.0), SR)
        self.assertTrue(p["available"])
        self.assertAlmostEqual(p["median_hz"], 120.0, delta=3.0)

    def test_higher_sine_reads_higher(self):
        p = estimate_pitch(sine(200.0), SR)
        self.assertTrue(p["available"])
        self.assertAlmostEqual(p["median_hz"], 200.0, delta=4.0)

    def test_fundamental_not_the_octave(self):
        # A 110 Hz tone must read ~110, never ~220 — the octave-error regression guard.
        p = estimate_pitch(sine(110.0), SR)
        self.assertLess(p["median_hz"], 160.0)

    def test_silence_is_unavailable(self):
        p = estimate_pitch(np.zeros(SR // 2), SR)
        self.assertFalse(p["available"])

    def test_track_is_time_ordered_and_full(self):
        p = estimate_pitch(sine(140.0), SR)
        self.assertGreater(len(p["track"]), 3)
        self.assertTrue(all(70.0 <= v <= 400.0 for v in p["track"]))


class ContourSwingTests(unittest.TestCase):
    def test_flat_track_is_near_zero(self):
        self.assertLessEqual(contour_swing([120.0] * 20), 1.0)

    def test_arc_track_has_real_swing(self):
        # rise then fall over the phrase — a smooth pitch "bloom"
        arc = [110, 120, 135, 150, 163, 150, 130, 110, 95, 93]
        self.assertGreater(contour_swing(arc), 30.0)

    def test_octave_spike_does_not_inflate_swing(self):
        flat = [120.0] * 20
        spiked = flat.copy()
        spiked[10] = 240.0  # one doubled frame
        # de-octave should keep the spiked track close to the flat one
        self.assertLess(contour_swing(spiked), 8.0)

    def test_short_track_guards_to_zero(self):
        self.assertEqual(contour_swing([120.0, 130.0]), 0.0)


class SpeakerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        self.tmp.unlink(missing_ok=True)

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def _feed(self, base, feat, values):
        for v in values:
            base.update({feat: v})

    def test_relative_none_until_three_samples(self):
        base = SpeakerBaseline(self.tmp, "speaker")
        self.assertIsNone(base.relative("pitch_median_hz", 120.0))
        self._feed(base, "pitch_median_hz", [100.0, 110.0])
        self.assertIsNone(base.relative("pitch_median_hz", 120.0))
        base.update({"pitch_median_hz": 120.0})
        self.assertIsNotNone(base.relative("pitch_median_hz", 120.0))

    def test_z_and_delta_are_correct(self):
        # Robust centering: median 110, MAD median(|dev|)=10, spread=10*1.4826=14.826.
        base = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base, "pitch_median_hz", [100.0, 110.0, 120.0])
        z, delta = base.relative("pitch_median_hz", 124.826)
        self.assertAlmostEqual(delta, 14.826, delta=0.01)  # distance from typical (median)
        self.assertAlmostEqual(z, 1.0, delta=0.01)

    def test_median_center_resists_outliers(self):
        # One loud outlier must not drag the CENTER — the reason for median+MAD.
        # A clip AT the calm register reads ~0 delta; a mean pulled toward the 300 Hz
        # outlier would report this same clip as ~-32 Hz "lower than usual" (the bug).
        base = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base, "pitch_median_hz", [108.0, 112.0, 110.0, 114.0, 109.0, 300.0])
        z, delta = base.relative("pitch_median_hz", 110.0)
        self.assertLess(abs(delta), 5.0)

    def test_degenerate_window_falls_back(self):
        # MAD collapses to 0 on a flat window; spread must fall back, not divide by ~0.
        base = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base, "pitch_median_hz", [110.0, 110.0, 110.0])
        z, delta = base.relative("pitch_median_hz", 120.0)
        self.assertEqual(delta, 10.0)
        self.assertTrue(math.isfinite(z))

    def test_window_caps_at_WINDOW(self):
        base = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base, "pitch_median_hz", [float(i) for i in range(WINDOW + 15)])
        self.assertEqual(base.n(), WINDOW)
        # only the most recent WINDOW values survive
        self.assertEqual(base.win["pitch_median_hz"][0], float(15))

    def test_persists_and_reloads(self):
        base = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base, "pitch_median_hz", [100.0, 110.0, 120.0])
        reloaded = SpeakerBaseline(self.tmp, "speaker")
        self.assertEqual(reloaded.n(), 3)
        self.assertEqual(json.loads(self.tmp.read_text())["speaker"]["pitch_median_hz"],
                         [100.0, 110.0, 120.0])

    def test_speakers_are_isolated(self):
        base_a = SpeakerBaseline(self.tmp, "speaker")
        self._feed(base_a, "pitch_median_hz", [100.0, 110.0, 120.0])
        base_o = SpeakerBaseline(self.tmp, "other")
        self.assertEqual(base_o.n(), 0)


class PhraseTests(unittest.TestCase):
    def test_pitch_both_directions(self):
        self.assertIn("lower/warmer", phrase("pitch_median_hz", -1.0, -20.0))
        self.assertIn("higher-pitched", phrase("pitch_median_hz", 1.0, 20.0))

    def test_contour_both_directions(self):
        self.assertIn("flatter", phrase("pitch_contour_hz", -1.0, -10.0))
        self.assertIn("more melodic", phrase("pitch_contour_hz", 1.0, 10.0))

    def test_loud_both_directions(self):
        self.assertIn("softer", phrase("loud_dbfs", -1.0, -3.0))
        self.assertIn("louder", phrase("loud_dbfs", 1.0, 3.0))

    def test_deadband_is_silent(self):
        # |z| below the threshold names nothing
        self.assertIsNone(phrase("pitch_median_hz", Z - 0.01, 1.0))
        self.assertIsNone(phrase("pitch_median_hz", -(Z - 0.01), -1.0))

    def test_unknown_feature_is_none(self):
        self.assertIsNone(phrase("brightness_hz", 5.0, 100.0))


if __name__ == "__main__":
    unittest.main()
