"""The vendored acoustic core must refuse a BPM when there is no beat.

Before the salience/activity guards, `analyze_acoustic` named a tempo for any
input: 55.0 BPM for white noise and 184.6 BPM for a steady 440 Hz sine. Both
numbers are plausible tempos, which is what made the failure hard to see.
"""
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "AI_Ears"))
from acoustic_core import analyze_acoustic  # noqa: E402

SR = 44100
DURATION_S = 10.0


class AcousticCoreTempoTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name, x):
        path = self.root / name
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SR)
            wav.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        return str(path)

    @staticmethod
    def _t():
        return np.arange(int(SR * DURATION_S)) / SR

    def test_white_noise_refuses_bpm(self):
        """Onsets everywhere, none periodic: salience must reject it."""
        rng = np.random.default_rng(20260812)
        path = self._write("noise.wav", rng.normal(0, 0.2, len(self._t())))
        self.assertIsNone(analyze_acoustic(path)["tempo_bpm"])

    def test_steady_tone_refuses_bpm(self):
        """No onsets: the absolute activity gate must reject it."""
        path = self._write("sine.wav", 0.3 * np.sin(2 * np.pi * 440 * self._t()))
        self.assertIsNone(analyze_acoustic(path)["tempo_bpm"])

    def test_real_click_track_still_reports_bpm(self):
        """The guards must not simply refuse everything."""
        t = self._t()
        click = np.zeros(len(t))
        samples = np.arange(2000)
        burst = np.exp(-samples / 200.0) * np.sin(2 * np.pi * 1200 * samples / SR)
        for beat in np.arange(0, DURATION_S, 0.5):
            i = int(beat * SR)
            click[i:i + 2000] += 0.8 * burst
        bpm = analyze_acoustic(self._write("click.wav", click))["tempo_bpm"]
        self.assertIsNotNone(bpm)
        self.assertAlmostEqual(bpm, 120.0, delta=2.0)

    def test_silence_refuses_bpm(self):
        """Degenerate input must not divide its way into a tempo."""
        path = self._write("silence.wav", np.zeros(len(self._t())))
        self.assertIsNone(analyze_acoustic(path)["tempo_bpm"])


if __name__ == "__main__":
    unittest.main()
