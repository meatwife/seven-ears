"""Tests for Seven Ears: Music Edition.

All audio here is synthesized with numpy — no private recordings ever enter
tests or commits. Signals are built at the analysis rate so ffmpeg is not
required to run the suite.
"""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from seven_ears_music import (
    ANALYSIS_SR,
    analyze_file,
    analyze_signal,
    format_music_card,
)

SR = ANALYSIS_SR


def sine(freq: float, dur_s: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * dur_s)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def noise(dur_s: float, amp: float = 0.5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(SR * dur_s))


def click_track(bpm: float, dur_s: float, amp: float = 0.8) -> np.ndarray:
    """Short noise bursts on the beat over a quiet bed."""
    x = noise(dur_s, amp=0.005, seed=1)
    period = int(SR * 60.0 / bpm)
    click_len = int(SR * 0.02)
    rng = np.random.default_rng(2)
    for start in range(0, len(x) - click_len, period):
        x[start:start + click_len] += amp * rng.standard_normal(click_len)
    return x


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


class TempoTests(unittest.TestCase):
    def test_click_track_tempo_near_120(self):
        result = analyze_signal(click_track(120, 12.0), SR)
        tempo = result['pulse']['tempo']
        self.assertTrue(tempo['supported'])
        self.assertAlmostEqual(tempo['bpm'], 120, delta=4)

    def test_click_track_tempo_near_90(self):
        result = analyze_signal(click_track(90, 12.0), SR)
        tempo = result['pulse']['tempo']
        self.assertTrue(tempo['supported'])
        self.assertAlmostEqual(tempo['bpm'], 90, delta=4)

    def test_steady_sine_refuses_bpm(self):
        result = analyze_signal(sine(220, 10.0), SR)
        tempo = result['pulse']['tempo']
        self.assertFalse(tempo['supported'])
        self.assertIsNone(tempo['bpm'])
        self.assertTrue(tempo['reason'])

    def test_unpatterned_noise_refuses_bpm(self):
        result = analyze_signal(noise(10.0), SR)
        tempo = result['pulse']['tempo']
        self.assertFalse(tempo['supported'])

    def test_short_clip_refuses_bpm(self):
        result = analyze_signal(click_track(120, 2.0), SR)
        tempo = result['pulse']['tempo']
        self.assertFalse(tempo['supported'])
        self.assertIn('short', tempo['reason'])


class SectionTests(unittest.TestCase):
    def test_two_body_signal_finds_boundary_near_middle(self):
        # 12s quiet warm sine, then 12s loud bright noise: one clear shape-change.
        x = np.concatenate([sine(200, 12.0, amp=0.1), noise(12.0, amp=0.6)])
        result = analyze_signal(x, SR)
        sections = result['sections']
        self.assertGreaterEqual(len(sections), 2)
        boundary = sections[1]['start_s']
        self.assertAlmostEqual(boundary, 12.0, delta=2.5)

    def test_uniform_signal_stays_one_section(self):
        result = analyze_signal(sine(300, 15.0), SR)
        self.assertEqual(len(result['sections']), 1)

    def test_fade_in_does_not_create_edge_boundary(self):
        # A fade-in next to the zero-padded start of the novelty curve used to
        # pass the local-maximum test almost for free, stamping a phantom
        # boundary at t == context on nearly every real song.
        x = noise(40.0, amp=0.4)
        n_fade = int(SR * 3.0)
        x[:n_fade] *= np.linspace(0, 1, n_fade)
        result = analyze_signal(x, SR)
        self.assertEqual(len(result['sections']), 1)


class ColorTests(unittest.TestCase):
    def test_noise_is_brighter_than_low_sine(self):
        bright = analyze_signal(noise(5.0), SR)['color']
        dark = analyze_signal(sine(200, 5.0), SR)['color']
        self.assertGreater(bright['centroid_median_hz'], dark['centroid_median_hz'])
        self.assertEqual(dark['brightness_label'], 'dark / warm')

    def test_sine_reads_tonal_and_noise_reads_noisy(self):
        tonal = analyze_signal(sine(440, 5.0), SR)['color']
        noisy = analyze_signal(noise(5.0), SR)['color']
        self.assertLess(tonal['flatness_median'], noisy['flatness_median'])
        self.assertIn('tonal', tonal['tonality_label'])

    def test_band_balance_sums_to_roughly_100(self):
        b = analyze_signal(noise(5.0), SR)['color']['band_balance']
        self.assertAlmostEqual(sum(b.values()), 100.0, delta=1.0)


class DynamicsAndSilenceTests(unittest.TestCase):
    def test_constant_sine_is_steady(self):
        dyn = analyze_signal(sine(220, 8.0), SR)['dynamics']
        self.assertLess(dyn['loudness_spread_db'], 6)
        self.assertEqual(dyn['label'], 'steady / compressed')

    def test_loud_quiet_alternation_reads_dynamic(self):
        parts = []
        for i in range(4):
            parts.append(sine(220, 2.0, amp=0.6))
            parts.append(sine(220, 2.0, amp=0.01))
        dyn = analyze_signal(np.concatenate(parts), SR)['dynamics']
        self.assertGreater(dyn['loudness_spread_db'], 15)

    def test_leading_silence_is_measured(self):
        x = np.concatenate([np.zeros(int(SR * 2.0)), sine(220, 6.0)])
        sil = analyze_signal(x, SR)['silence']
        self.assertAlmostEqual(sil['leading_quiet_s'], 2.0, delta=0.5)

    def test_internal_drop_is_found(self):
        x = np.concatenate([sine(220, 4.0), np.zeros(int(SR * 2.0)), sine(220, 4.0)])
        sil = analyze_signal(x, SR)['silence']
        self.assertEqual(len(sil['internal_drops']), 1)
        self.assertAlmostEqual(sil['internal_drops'][0]['dur_s'], 2.0, delta=0.6)


class HonestyAndCardTests(unittest.TestCase):
    def test_unsupported_claims_are_declared(self):
        result = analyze_signal(sine(220, 5.0), SR)
        for key in ('vocal_presence', 'emotion', 'genre', 'key_and_melody', 'lyrics'):
            self.assertIn(key, result['not_measured'])

    def test_card_has_no_emotion_language(self):
        card = format_music_card(analyze_signal(click_track(120, 12.0), SR))
        self.assertIn('Seven Ears, music card', card)
        self.assertIn('No mind-reading', card)
        for banned in ('sad', 'happy', 'angry', 'melanch', 'euphori'):
            self.assertNotIn(banned, card.lower())

    def test_card_reports_refusal_when_no_pulse(self):
        card = format_music_card(analyze_signal(sine(220, 10.0), SR))
        self.assertNotIn('BPM', card)
        self.assertIn('nothing pulse-like', card.lower())

    def test_steady_sine_reports_no_onsets(self):
        pulse = analyze_signal(sine(220, 10.0), SR)['pulse']
        self.assertEqual(pulse['onset_count'], 0)


class FileRoundTripTests(unittest.TestCase):
    def test_analyze_file_reads_synthetic_wav(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'click.wav'
            write_wav(path, click_track(120, 12.0))
            result = analyze_file(path)
            self.assertEqual(result['file'], 'click.wav')
            self.assertTrue(result['pulse']['tempo']['supported'])


if __name__ == '__main__':
    unittest.main()
