#!/usr/bin/env python3
"""Local acoustic analysis extracted from menelly/AI_Ears.

Original project: https://github.com/menelly/AI_Ears
Copyright (c) 2026 Ace, MIT License. See LICENSE in this directory.

This vendored extract contains only the local NumPy acoustic half used by
Seven Ears. AI_Ears' cloud/STT backends are intentionally not included.
"""
from __future__ import annotations

import wave

import numpy as np

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return samples, sample_rate


def analyze_acoustic(wav_path: str) -> dict:
    samples, sample_rate = read_wav_mono(wav_path)
    duration = len(samples) / sample_rate
    if len(samples) == 0:
        return {"error": "empty audio"}

    window_size, hop = 2048, 512
    frame_count = max(1, 1 + (len(samples) - window_size) // hop) if len(samples) >= window_size else 1
    frames = np.zeros((frame_count, window_size))
    for index in range(frame_count):
        segment = samples[index * hop:index * hop + window_size]
        frames[index, :len(segment)] = segment

    window = np.hanning(window_size)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)

    magnitude = spectrum + 1e-9
    mean_magnitude = magnitude.mean(axis=0)
    centroid = float(np.sum(frequencies * mean_magnitude) / np.sum(mean_magnitude))
    brightness_label = (
        "very bright" if centroid > 4000 else
        "bright" if centroid > 2500 else
        "warm" if centroid > 1200 else
        "dark"
    )

    peak = float(np.max(np.abs(samples)))
    peak_dbfs = 20 * np.log10(peak + 1e-9)
    rms_overall = float(np.sqrt(np.mean(samples ** 2)))
    crest = 20 * np.log10((peak + 1e-9) / (rms_overall + 1e-9))
    voiced = rms_db[rms_db > (rms_db.max() - 60)]
    if voiced.size < 4:
        voiced = rms_db
    loud = float(np.percentile(voiced, 95))
    quiet = float(np.percentile(voiced, 5))
    dynamic_range = loud - quiet
    dynamics_label = (
        "very dynamic" if dynamic_range > 25 else
        "dynamic" if dynamic_range > 14 else
        "even" if dynamic_range > 7 else
        "flat/compressed"
    )

    chroma = np.zeros(12)
    for frequency, strength in zip(frequencies, mean_magnitude):
        if frequency < 55 or frequency > 5000:
            continue
        midi = 69 + 12 * np.log2(frequency / 440.0)
        chroma[int(round(midi)) % 12] += strength
    chroma = chroma / (chroma.sum() + 1e-9)
    best: tuple[float, str | None, str | None] = (-2.0, None, None)
    for shift in range(12):
        rotated = np.roll(chroma, -shift)
        major = float(np.corrcoef(rotated, KS_MAJOR)[0, 1])
        minor = float(np.corrcoef(rotated, KS_MINOR)[0, 1])
        if major > best[0]:
            best = (major, NOTES[shift], "major")
        if minor > best[0]:
            best = (minor, NOTES[shift], "minor")
    key_confidence, key_root, key_mode = best

    flux = np.maximum(0, np.diff(spectrum, axis=0)).sum(axis=1)
    tempo_bpm = None
    # Absolute activity gate, BEFORE any periodicity test. A steady tone has no
    # onsets at all, so its flux is pure numerical jitter -- and normalising the
    # autocorrelation by ac[0] rescales that jitter into a confident-looking
    # peak. The salience test alone cannot catch it, because after division the
    # jitter genuinely is periodic. (Measured on this extract: a 440 Hz sine
    # still returned 184.6 BPM with the peak/salience guard in place.)
    # Threshold: onset flux relative to mean frame energy separates cleanly --
    #   steady sine 0.0004 | white noise 0.2674 | 120 BPM click 15.39
    # a 668x gap whose geometric midpoint is 0.0103, so 0.01 sits ~25x above the
    # case it rejects and ~27x below the nearest case it must accept. Chosen from
    # that spread rather than tuned until the three test signals agreed with me.
    frame_energy = float(spectrum.sum(axis=1).mean())
    has_onsets = float(flux.max()) > 0.01 * (frame_energy + 1e-9)
    if len(flux) > 8 and has_onsets:
        flux = flux - flux.mean()
        # FFT autocorrelation is equivalent to np.correlate(..., mode="full")
        # for the non-negative lags used here, while scaling to longer clips.
        fft_size = 1 << (2 * len(flux) - 1).bit_length()
        transformed = np.fft.rfft(flux, n=fft_size)
        autocorrelation = np.fft.irfft(transformed * np.conj(transformed), n=fft_size)[:len(flux)]
        # Normalise so the salience thresholds below are scale-free, matching
        # upstream hear_core.py and seven_ears_music.estimate_tempo.
        if autocorrelation[0] > 0:
            autocorrelation = autocorrelation / autocorrelation[0]
        frames_per_second = sample_rate / hop
        low = int(frames_per_second * 60 / 240)
        high = min(int(frames_per_second * 60 / 50), len(autocorrelation) - 1)
        if high > low + 2:
            # A plain argmax over the band returns its largest value whether or
            # not that value is a peak, so a signal with no beat still gets a
            # confident BPM (measured on this extract: 55.0 for white noise,
            # 184.6 for a steady sine). Upstream calls this "the old ~246-BPM
            # ceiling artifact"; here the FFT autocorrelation puts it elsewhere,
            # which makes it harder to spot rather than easier.
            # Fix, as upstream: require a genuine LOCAL peak that stands clearly
            # above the band, and report nothing when none does.
            band = autocorrelation[low:high]
            peaks = np.where((band[1:-1] > band[:-2]) & (band[1:-1] >= band[2:]))[0] + 1
            if peaks.size:
                k = int(peaks[np.argmax(band[peaks])])
                if band[k] > np.median(band) + 0.10 and band[k] >= 0.15:
                    tempo_bpm = 60.0 * frames_per_second / (low + k)

    threshold = np.percentile(rms_db, 30)
    floor = max(threshold, quiet + 6)
    quiet_mask = rms_db < floor
    events = []
    index = 0
    while index < len(quiet_mask):
        if quiet_mask[index]:
            end = index
            while end < len(quiet_mask) and quiet_mask[end]:
                end += 1
            start_s, end_s = index * hop / sample_rate, end * hop / sample_rate
            if (end_s - start_s) >= 0.18 and start_s > 0.05 and end_s < duration - 0.05:
                events.append((round(start_s, 2), round(end_s, 2), round(end_s - start_s, 2)))
            index = end
        else:
            index += 1
    events.sort(key=lambda event: -event[2])

    return {
        "duration_s": round(duration, 2),
        "sample_rate": sample_rate,
        "brightness_hz": round(centroid),
        "brightness_label": brightness_label,
        "key": f"{key_root} {key_mode}" if key_root else "unclear",
        "key_confidence": round(float(key_confidence), 2),
        "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else None,
        "peak_dbfs": round(peak_dbfs, 1),
        "crest_db": round(float(crest), 1),
        "loud_dbfs": round(loud, 1),
        "quiet_dbfs": round(quiet, 1),
        "dynamic_range_db": round(dynamic_range, 1),
        "dynamics_label": dynamics_label,
        "pauses": events[:6],
    }
