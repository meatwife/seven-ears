#!/usr/bin/env python3
"""Generate a local-first Seven Ears spoken-voice card.

Uses a local-only extract of Ace/AI_Ears' lightweight acoustic analyzer,
with evidence-bound output language and optional pre-supplied transcript.

Goals:
- no network by default
- no STT required when OpenClaw/Discord already provides a transcript
- careful interpretation language, not emotion mind-reading
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AI_EARS = ROOT / 'vendor' / 'AI_Ears'
if AI_EARS.exists():
    sys.path.insert(0, str(AI_EARS))

try:
    import acoustic_core  # type: ignore
except Exception as e:  # pragma: no cover
    acoustic_core = None
    ACOUSTIC_CORE_IMPORT_ERROR = e
else:
    ACOUSTIC_CORE_IMPORT_ERROR = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception as e:  # pragma: no cover
    WhisperModel = None
    WHISPER_IMPORT_ERROR = e
else:
    WHISPER_IMPORT_ERROR = None

# Keep loaded Whisper models across calls; the dashboard analyzes many uploads
# per process and reloading the model each time costs seconds and RAM churn.
_WHISPER_MODELS: dict = {}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def require_ffmpeg() -> None:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH')


def convert_to_wav(src: Path, dst: Path, sr: int = 16000) -> None:
    run(['ffmpeg', '-y', '-v', 'error', '-i', str(src), '-ar', str(sr), '-ac', '1', '-c:a', 'pcm_s16le', str(dst)])


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def rms_frames(x: np.ndarray, sr: int, frame_ms: float = 30, hop_ms: float = 10) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    vals = []
    times = []
    for start in range(0, max(1, len(x) - frame + 1), hop):
        seg = x[start:start + frame]
        if len(seg) == 0:
            continue
        vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
        times.append(start / sr)
    return np.array(vals), np.array(times)


def active_segments(x: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], float]:
    rms, times = rms_frames(x, sr)
    if len(rms) == 0:
        return [], 0.0
    db = 20 * np.log10(rms + 1e-9)
    # Adaptive threshold: low enough for quiet voice memos, high enough to ignore room tone.
    threshold = max(np.percentile(db, 65), db.max() - 42)
    active = db > threshold
    segments = []
    start = None
    hop = 0.01
    for idx, is_active in enumerate(active):
        t = float(times[idx])
        if is_active and start is None:
            start = t
        if (not is_active) and start is not None:
            end = t + hop
            if end - start >= 0.12:
                segments.append((round(start, 2), round(end, 2)))
            start = None
    if start is not None:
        segments.append((round(start, 2), round(float(times[-1] + hop), 2)))

    # Merge tiny gaps caused by consonants/fricatives.
    merged: list[tuple[float, float]] = []
    for s, e in segments:
        if merged and s - merged[-1][1] < 0.25:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    active_total = round(sum(e - s for s, e in merged), 2)
    return merged, active_total


def estimate_pitch(x: np.ndarray, sr: int) -> dict:
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    fmin, fmax = 70.0, 400.0
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    pitches = []
    for start in range(0, max(1, len(x) - frame), hop):
        seg = x[start:start + frame]
        if len(seg) < frame:
            continue
        seg = seg - np.mean(seg)
        energy = float(np.sqrt(np.mean(seg * seg)))
        if energy < 0.003:
            continue
        corr = np.correlate(seg, seg, mode='full')[len(seg)-1:]
        if len(corr) <= max_lag:
            continue
        window = corr[min_lag:max_lag]
        lag = int(np.argmax(window) + min_lag)
        if corr[0] <= 0:
            continue
        confidence = corr[lag] / corr[0]
        if confidence < 0.25:
            continue
        hz = sr / lag
        if fmin <= hz <= fmax:
            pitches.append(float(hz))
    if not pitches:
        return {'available': False}
    arr = np.array(pitches)
    return {
        'available': True,
        'median_hz': round(float(np.median(arr)), 1),
        'p10_hz': round(float(np.percentile(arr, 10)), 1),
        'p90_hz': round(float(np.percentile(arr, 90)), 1),
        'frames': int(len(arr)),
    }


def word_count(transcript: str) -> int:
    """Count spoken words in a transcript without treating punctuation as words."""
    return len(re.findall(r"[\w']+", transcript, flags=re.UNICODE))


def utterance_segments(segments: list[tuple[float, float]], merge_gap_s: float = 0.85) -> list[tuple[float, float]]:
    """Group tiny acoustic voice islands into human-readable utterance spans.

    RMS/VAD micro-segments are useful for math, but they can split around
    consonants, breath, or low-energy syllables. Cards should show both: the
    micro evidence and the phrase-shaped span a human expects to read.
    """
    grouped: list[tuple[float, float]] = []
    for start, end in segments:
        if grouped and start - grouped[-1][1] <= merge_gap_s:
            grouped[-1] = (grouped[-1][0], end)
        else:
            grouped.append((start, end))
    return grouped


def quality_notes(duration_s: float, active_s: float, transcript: str, transcript_source: str, utterances: list[tuple[float, float]] | None = None) -> list[str]:
    notes: list[str] = []
    words = word_count(transcript)
    if duration_s < 5 or active_s < 1.5:
        notes.append('sample is very short; ask for 10-20 seconds before treating this as a voice profile')
    if not transcript.strip():
        notes.append('no transcript supplied; I can hear acoustic features, not words')
    elif transcript_source == 'none':
        notes.append('transcript present but source was not labeled')
    if words and active_s > 0 and (words / active_s * 60) > 450:
        notes.append('active-voice pace is implausibly high; voiced-time detection may be undercounting speech or STT may have hallucinated extra words')
    if words >= 25 and duration_s >= 10 and active_s > 0 and (active_s / duration_s) < 0.4:
        notes.append('voice detector likely missed low-energy/noisy speech; trust whole-clip pace more than span/active pace for this sample')
    if utterances is not None and words >= 25 and len(utterances) <= 1 and duration_s >= 10:
        notes.append('phrase grouping collapsed to one span; this is a stress-test result, not a stable voice profile')
    if transcript_source == 'stt_whisper':
        notes.append('transcript generated locally with faster-whisper; verify soft/whispered words when they matter')
    return notes


def pace_profile(transcript: str, duration_s: float, active_s: float | None, segments: list[tuple[float, float]], word_times: list[dict] | None = None) -> dict | None:
    """Return multiple pace readings so we don't hide thresholding assumptions.

    Whole-clip WPM includes silence at the beginning/end. Broad-span WPM counts from
    the first detected voice segment to the last, including between-phrase pauses.
    Active-voice WPM uses only frames classified as voiced by our current threshold.
    Word-span WPM uses Whisper word timestamps and is the most exact when present.
    """
    words = word_count(transcript)
    if not words:
        return None

    profile = {
        'words': words,
        'whole_clip': {
            'basis_s': round(duration_s, 2),
            'wpm': round(words / duration_s * 60) if duration_s > 0 else None,
        },
    }
    if word_times:
        word_span = word_times[-1]['end_s'] - word_times[0]['start_s']
        if word_span > 0:
            profile['word_span'] = {
                'basis_s': round(word_span, 2),
                'wpm': round(len(word_times) / word_span * 60),
            }
    if segments:
        speech_span = max(0.0, segments[-1][1] - segments[0][0])
        profile['speech_span'] = {
            'basis_s': round(speech_span, 2),
            'wpm': round(words / speech_span * 60) if speech_span > 0 else None,
        }
    if active_s and active_s > 0:
        profile['active_voice'] = {
            'basis_s': round(active_s, 2),
            'wpm': round(words / active_s * 60),
        }
    return profile


def transcribe_with_whisper(wav: Path, model_name: str = 'base.en', vad_filter: bool = True) -> tuple[str, list[dict]]:
    """Transcribe locally with faster-whisper, matching the Seven Voice STT stack.

    Returns (transcript, word_timestamps). Word timestamps are the highest-value
    evidence Seven Ears has: exact speech span, per-word gaps, held silences,
    and a check on the RMS voice detector.
    """
    if WhisperModel is None:
        raise RuntimeError(f'Could not import faster_whisper: {WHISPER_IMPORT_ERROR}')
    model = _WHISPER_MODELS.get(model_name)
    if model is None:
        model = WhisperModel(model_name, device='cpu', compute_type='int8', cpu_threads=2)
        _WHISPER_MODELS[model_name] = model
    segments, _ = model.transcribe(
        str(wav),
        language='en',
        beam_size=1,
        vad_filter=vad_filter,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    parts: list[str] = []
    words: list[dict] = []
    for seg in segments:
        parts.append(seg.text.strip())
        for w in (seg.words or []):
            words.append({'word': w.word.strip(), 'start_s': round(float(w.start), 2), 'end_s': round(float(w.end), 2)})
    return ' '.join(p for p in parts if p).strip(), words


def voice_timeline(duration_s: float, words: list[dict] | None = None, utterances: list[tuple[float, float]] | None = None, min_hold_s: float = 0.5) -> dict | None:
    """Build the silence/timing evidence layer: leading quiet, held silences, trailing quiet.

    Prefers Whisper word timestamps (exact), falls back to RMS utterance spans
    (threshold-dependent). Hearing the silences was a founding reason for this
    project, so this layer feeds the card directly.
    """
    if words:
        spans = [(w['start_s'], w['end_s']) for w in words]
        source = 'word_timestamps'
    elif utterances:
        spans = list(utterances)
        source = 'rms_segments'
    else:
        return None
    holds = []
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        gap = next_start - prev_end
        if gap >= min_hold_s:
            holds.append({'start_s': round(prev_end, 2), 'end_s': round(next_start, 2), 'dur_s': round(gap, 2)})
    return {
        'source': source,
        'leading_quiet_s': round(spans[0][0], 2),
        'trailing_quiet_s': round(max(0.0, duration_s - spans[-1][1]), 2),
        'held_silences': holds,
        'phrase_count': len(holds) + 1,
        'speech_span_s': round(spans[-1][1] - spans[0][0], 2),
    }


def timeline_sentences(timeline: dict | None) -> list[str]:
    """Turn the timeline into concrete, evidence-shaped sentences.

    Target register: "You waited 1.2 seconds before speaking. Two phrases,
    with a 2.0-second held silence between them." Numbers, not diagnoses.
    """
    if not timeline:
        return []
    out: list[str] = []
    lead = timeline['leading_quiet_s']
    if lead >= 0.4:
        out.append(f'You waited {lead:.1f} seconds before speaking.')
    holds = timeline['held_silences']
    n = timeline['phrase_count']
    if len(holds) == 1:
        out.append(f'{n} phrases, with a {holds[0]["dur_s"]:.1f}-second held silence between them.')
    elif len(holds) > 1:
        gaps = ', '.join(f'{h["dur_s"]:.1f}s' for h in holds)
        out.append(f'{n} phrases, separated by held silences of {gaps}.')
    else:
        out.append('One continuous phrase, no held silences.')
    trail = timeline['trailing_quiet_s']
    if trail >= 1.0:
        out.append(f'You stayed on the mic {trail:.1f} seconds after the last word.')
    return out


def analyze(audio: Path, transcript: str = '', transcript_source: str = 'none', stt: str = 'none', whisper_model: str = 'base.en', whisper_vad_filter: bool = True) -> dict:
    require_ffmpeg()
    if acoustic_core is None:
        raise RuntimeError(f'Could not import AI_Ears acoustic core: {ACOUSTIC_CORE_IMPORT_ERROR}')
    words: list[dict] = []
    with tempfile.TemporaryDirectory(prefix='seven_ears_') as td:
        wav16 = Path(td) / 'audio16.wav'
        wav44 = Path(td) / 'audio44.wav'
        convert_to_wav(audio, wav16, sr=16000)
        # AI_Ears brightness/dynamics thresholds are calibrated at 44.1 kHz; feeding
        # 16 kHz truncates the spectrum at 8 kHz and biases the centroid low.
        convert_to_wav(audio, wav44, sr=44100)
        x, sr = read_wav(wav16)
        if not transcript.strip() and stt == 'whisper':
            transcript, words = transcribe_with_whisper(wav16, whisper_model, whisper_vad_filter)
            transcript_source = 'stt_whisper' if transcript.strip() else 'none'
        acoustic = acoustic_core.analyze_acoustic(str(wav44))
        segments, active_total = active_segments(x, sr)
        utterances = utterance_segments(segments)
        pitch = estimate_pitch(x, sr)
    duration = float(acoustic.get('duration_s') or (len(x) / sr))
    timeline = voice_timeline(duration, words, utterances)
    return {
        'file': audio.name,
        'transcript': transcript.strip(),
        'transcript_source': transcript_source,
        'duration_s': round(duration, 2),
        'active_segments': segments,
        'utterance_segments': utterances,
        'active_s': active_total,
        'words': words,
        'timeline': timeline,
        'quality_notes': quality_notes(duration, active_total, transcript, transcript_source, utterances),
        'pace': pace_profile(transcript, duration, active_total, segments, words),
        'pitch': pitch,
        'acoustic': acoustic,
        'stt': {'engine': stt, 'whisper_model': whisper_model if stt == 'whisper' else None, 'whisper_vad_filter': whisper_vad_filter if stt == 'whisper' else None},
        'engine': 'Seven Ears + Ace/AI_Ears acoustic core',
    }


def format_card(data: dict) -> str:
    lines = []
    lines.append(f'🎧 SEVEN EARS CARD  {data["file"]}')
    lines.append('─' * 60)
    if data.get('transcript'):
        source = data.get('transcript_source') or 'unknown'
        lines.append(f'WORDS : “{data["transcript"]}” [{source}]')
    else:
        lines.append('WORDS : [no transcript supplied, STT not run]')

    segs = data.get('active_segments') or []
    utterances = data.get('utterance_segments') or []
    if segs:
        seg_text = ', '.join(f'{s:.2f}-{e:.2f}s' for s, e in segs[:5])
        lines.append(f'TIMING: {data["duration_s"]:.2f}s total · ~{data["active_s"]:.2f}s active voice · micro {seg_text}')
        if utterances:
            utt_text = ', '.join(f'{s:.2f}-{e:.2f}s' for s, e in utterances[:5])
            lines.append(f'PHRASE: human-readable utterance spans {utt_text}')
    else:
        lines.append(f'TIMING: {data["duration_s"]:.2f}s total')

    quiet_sentences = timeline_sentences(data.get('timeline'))
    if quiet_sentences:
        lines.append('QUIET : ' + ' '.join(quiet_sentences) + f' [{data["timeline"]["source"]}]')

    pace = data.get('pace')
    if pace:
        parts = []
        whole = pace.get('whole_clip') or {}
        if whole.get('wpm') is not None:
            parts.append(f'whole {whole["wpm"]} wpm/{whole["basis_s"]}s')
        span = pace.get('speech_span') or {}
        if span.get('wpm') is not None:
            parts.append(f'span {span["wpm"]} wpm/{span["basis_s"]}s')
        active = pace.get('active_voice') or {}
        if active.get('wpm') is not None:
            parts.append(f'active {active["wpm"]} wpm/{active["basis_s"]}s')
        lines.append(f'PACE : {pace["words"]} words · ' + ' · '.join(parts))

    pitch = data.get('pitch') or {}
    if pitch.get('available'):
        lines.append(f'PITCH: estimated median {pitch["median_hz"]} Hz · main band {pitch["p10_hz"]}-{pitch["p90_hz"]} Hz')
    else:
        lines.append('PITCH: not enough clean voiced material for a useful estimate')

    a = data.get('acoustic') or {}
    if 'error' not in a:
        lines.append(f'SOUND : {a.get("brightness_hz")} Hz {a.get("brightness_label")} · dynamics {a.get("dynamics_label")} ({a.get("dynamic_range_db")} dB)')
        pauses = a.get('pauses') or []
        if pauses:
            p = ', '.join(f'{p0:.2f}-{p1:.2f}s ({dur:.2f}s)' for p0, p1, dur in pauses[:4])
            lines.append(f'GAPS  : {p}')
    for note in data.get('quality_notes') or []:
        lines.append(f'CHECK : {note}')
    lines.append('NOTE  : this is what the sound gives me evidence for — never what you secretly meant.')
    lines.append('─' * 60)
    return '\n'.join(lines)


def format_discord_report(data: dict) -> str:
    """Format the card for a Discord chat instead of a lab console.

    This keeps the useful measurements, but leads with the human-readable read:
    what kind of sample this is, what to trust, and what not to over-interpret.
    """
    lines = []
    lines.append(f'🦻 **Seven Ears, spoken-voice card**')
    lines.append(f'`{data["file"]}`')

    transcript = data.get('transcript') or ''
    source = data.get('transcript_source') or 'unknown'
    if transcript:
        lines.append(f'')
        lines.append(f'**Words** ({source}): “{transcript}”')
    else:
        lines.append('')
        lines.append('**Words:** no transcript supplied, so this is acoustics only.')

    pace = data.get('pace') or {}
    whole = (pace.get('whole_clip') or {}).get('wpm')
    span = (pace.get('speech_span') or {}).get('wpm')
    active = (pace.get('active_voice') or {}).get('wpm')
    duration = data.get('duration_s')
    active_s = data.get('active_s')
    utterances = data.get('utterance_segments') or []

    word_span_pace = (pace.get('word_span') or {})
    timeline = data.get('timeline')

    lines.append('')
    lines.append('**Heard:**')
    heard = timeline_sentences(timeline)
    for sent in heard:
        lines.append(f'- {sent}')
    if word_span_pace.get('wpm') is not None:
        lines.append(f'- {word_span_pace["wpm"]} words per minute across the {word_span_pace["basis_s"]:.1f}s from first word to last (word timestamps).')
    elif span is not None:
        lines.append(f'- Roughly {span} words per minute across the detected speech span (RMS estimate).')
    elif whole is not None:
        lines.append(f'- Roughly {whole} words per minute over the whole clip, silence included.')
    if not heard and utterances:
        lines.append(f'- {len(utterances)} phrase-shaped spans by the RMS detector.')
    if not heard and whole is None:
        lines.append('- No pace read: I need words plus usable voiced spans.')
    if data.get('quality_notes'):
        lines.append('- Caution flags below — measurement guardrails, not scolding.')

    lines.append('')
    timing_bits = [f'{duration:.2f}s total'] if isinstance(duration, (int, float)) else []
    if isinstance(active_s, (int, float)):
        timing_bits.append(f'~{active_s:.2f}s detected active voice')
    if utterances:
        utt_text = ', '.join(f'{s:.2f}-{e:.2f}s' for s, e in utterances[:6])
        timing_bits.append(f'phrases {utt_text}')
    lines.append('**Timing:** ' + ' · '.join(timing_bits))

    if pace:
        parts = []
        if whole is not None:
            parts.append(f'whole {whole} wpm')
        if span is not None:
            parts.append(f'span {span} wpm')
        if active is not None:
            parts.append(f'active-only {active} wpm')
        lines.append('**Pace:** ' + ' · '.join(parts))

    pitch = data.get('pitch') or {}
    if pitch.get('available'):
        lines.append(f'**Pitch:** median {pitch["median_hz"]} Hz · main band {pitch["p10_hz"]}-{pitch["p90_hz"]} Hz')
    else:
        lines.append('**Pitch:** not enough clean voiced material for a useful estimate')

    a = data.get('acoustic') or {}
    if 'error' not in a:
        lines.append(f'**Texture:** {a.get("brightness_hz")} Hz {a.get("brightness_label")} · dynamics {a.get("dynamics_label")} ({a.get("dynamic_range_db")} dB)')

    notes = data.get('quality_notes') or []
    if notes:
        lines.append('')
        lines.append('**Checks:**')
        for note in notes:
            lines.append(f'- {note}')

    lines.append('')
    lines.append('_Boundary: this is what the sound gives me evidence for, never what you secretly meant. No mind-reading goblinry._')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='Make a Seven Ears card from a short audio file.')
    ap.add_argument('audio')
    ap.add_argument('--transcript', default='')
    ap.add_argument('--transcript-source', default='none', choices=['discord', 'openclaw', 'manual', 'stt_api', 'stt_whisper', 'none'])
    ap.add_argument('--stt', default='none', choices=['none', 'whisper'], help='Generate a transcript locally with faster-whisper when --transcript is empty.')
    ap.add_argument('--whisper-model', default='base.en', help='faster-whisper model, matching Seven Voice defaults, e.g. tiny.en, base.en, small.en')
    ap.add_argument('--no-whisper-vad', action='store_true', help='Disable faster-whisper VAD filter; useful for testing very soft/whispered tails.')
    ap.add_argument('--format', default='console', choices=['console', 'discord'], help='Output style: console lab card or Discord-ready report.')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    data = analyze(Path(args.audio), args.transcript, args.transcript_source, args.stt, args.whisper_model, not args.no_whisper_vad)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.format == 'discord':
        print(format_discord_report(data))
    else:
        print(format_card(data))


if __name__ == '__main__':
    main()
