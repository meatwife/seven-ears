#!/usr/bin/env python3
"""Local, cooperative sequential listening. See docs/FIRST_LISTEN.md.

Conceptual inspiration: v3nommy/Music-for-Machine-Ears. Independent code;
measurements use Seven Ears' existing Music Edition, not MME code or data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
from seven_ears_music import analyze_signal, load_audio, spectral_color, stft_mag


class ListenError(ValueError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def measure(x, sr):
    # Existing analyzer needs >=2 STFT frames. Do not pad a tiny tail into
    # invented duration or feed empty flux to max().
    if len(x) < 2560 or float(np.max(np.abs(x))) < 1e-9:
        return {'duration_s': len(x) / sr, 'available': False,
                'reason': 'too short or digitally silent; no texture/pulse claim'}
    result = analyze_signal(x, sr, source_name='passage')
    result.pop('file')
    return result


def format_passage_card(packet):
    start, end = packet['start_s'], packet['end_s']
    motion, evidence = packet['motion'], packet['measurements']
    lines = [f'Passage {start:.2f}–{end:.2f}s. Measurements use only this passage.']
    for point in motion:
        brightness = (f"centroid ~{point['centroid_hz']} Hz" if point['centroid_hz'] is not None
                      else 'no measurable spectral color')
        lines.append(f"{point['start_s']:.2f}–{point['end_s']:.2f}s: "
                     f"RMS {point['rms_dbfs']} dBFS; {brightness}.")
    color = evidence.get('color', {})
    if color.get('available'):
        lines.append(f"Texture: {color['brightness_label']}; {color['tonality_label']}.")
        bands = color.get('band_balance', {})
        if bands:
            lines.append(
                'Band balance: '
                f"low {bands.get('low_pct')}%, mid {bands.get('mid_pct')}%, "
                f"upper-mid {bands.get('upper_mid_pct')}%, high {bands.get('high_pct')}%."
            )
    dynamics = evidence.get('dynamics', {})
    if dynamics.get('available'):
        lines.append(
            f"Dynamics: {dynamics['label']}; {dynamics['loudness_spread_db']} dB local spread; "
            f"peak {dynamics['peak_dbfs']} dBFS."
        )
    pulse = evidence.get('pulse', {})
    if pulse:
        lines.append(
            f"Activity: {pulse.get('onset_count', 0)} detected onsets "
            f"({pulse.get('onset_density_per_s', 0.0)}/s)."
        )
    tempo = pulse.get('tempo', {})
    if tempo.get('supported'):
        lines.append(f"Local pulse estimate ~{tempo['bpm']} BPM; half/double ambiguity possible.")
    else:
        lines.append('No supported local tempo estimate.')
    lines.append('Write your own impression, uncertainty or anticipation; no required emotion.')
    return '\n'.join(lines)


def passage(x, sr, start):
    evidence = measure(x, sr)
    motion = []
    # Absolute level; no normalization, ranking or thresholds from future audio.
    for offset in range(0, len(x), 5 * sr):
        part = x[offset:offset + 5 * sr]
        rms = float(np.sqrt(np.mean(part * part)))
        mags, freqs, _ = stft_mag(part, sr)
        color = spectral_color(mags, freqs)
        motion.append({'start_s': round(start + offset / sr, 4),
                       'end_s': round(start + (offset + len(part)) / sr, 4),
                       'rms_dbfs': round(20 * math.log10(max(rms, 1e-9)), 1),
                       'centroid_hz': color.get('centroid_median_hz')})
    end = start + len(x) / sr
    packet = {'start_s': round(start, 4), 'end_s': round(end, 4),
              'motion': motion, 'measurements': evidence}
    packet['card'] = format_passage_card(packet)
    return packet


def boundary_feature(x, sr):
    """Small causal summary for deciding whether an encountered block closes a passage."""
    rms = float(np.sqrt(np.mean(x * x)))
    measured = measure(x, sr)
    color = measured.get('color', {})
    return {
        'rms_dbfs': 20 * math.log10(max(rms, 1e-9)),
        'centroid_hz': color.get('centroid_median_hz'),
        'bands': color.get('band_balance'),
        'onset_density': measured.get('pulse', {}).get('onset_density_per_s', 0.0),
    }


def boundary_changed(before, after):
    level_delta = abs(after['rms_dbfs'] - before['rms_dbfs'])
    # A fade already below this floor has no useful new body to begin; keep the
    # tail with the passage instead of manufacturing a subsecond corpse packet.
    if after['rms_dbfs'] <= -50 and after['rms_dbfs'] < before['rms_dbfs']:
        return False
    if before['centroid_hz'] and after['centroid_hz']:
        octave_delta = abs(math.log2(after['centroid_hz'] / before['centroid_hz']))
    else:
        octave_delta = 0.0
    if before['bands'] and after['bands']:
        names = ('low_pct', 'mid_pct', 'upper_mid_pct', 'high_pct')
        band_delta = sum(abs(after['bands'][name] - before['bands'][name]) for name in names)
    else:
        band_delta = 0.0
    activity_delta = abs(after['onset_density'] - before['onset_density'])
    major = level_delta >= 7 or octave_delta >= 1.0 or band_delta >= 55 or activity_delta >= 1.7
    moderate = sum((level_delta >= 5, octave_delta >= 0.6,
                    band_delta >= 30, activity_delta >= 1.0))
    return major or moderate >= 2


def adaptive_ranges(x, sr, minimum_s=15.0, maximum_s=30.0, probe_s=5.0):
    """Choose passage ends after changes are encountered, never by looking ahead."""
    minimum = round(minimum_s * sr)
    maximum = round(maximum_s * sr)
    probe = round(probe_s * sr)
    ranges = []
    start = 0
    while start < len(x):
        hard_end = min(len(x), start + maximum)
        end = min(hard_end, start + minimum)
        while end < hard_end:
            candidate = min(hard_end, end + probe)
            before = boundary_feature(x[max(start, candidate - 2 * probe):candidate - probe], sr)
            after = boundary_feature(x[candidate - probe:candidate], sr)
            end = candidate
            if boundary_changed(before, after):
                break
        ranges.append((start, end))
        start = end
    return ranges


def prepare_signal(x, sr, destination, seconds=20.0, metadata=None, adaptive=False):
    if not isinstance(sr, int) or sr <= 0:
        raise ListenError('sample rate must be a positive integer')
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ListenError('audio must be nonempty, finite and mono')
    if not math.isfinite(seconds) or not 5 <= seconds <= 60:
        raise ListenError('passage length must be finite and between 5 and 60 seconds')
    destination = Path(destination)
    if destination.exists():
        raise ListenError('session already exists; use next to resume')
    if adaptive:
        ranges = adaptive_ranges(x, sr)
    else:
        step = round(seconds * sr)
        ranges = [(i, min(len(x), i + step)) for i in range(0, len(x), step)]
    packets = [passage(x[start:end], sr, start / sr) for start, end in ranges]
    whole = measure(x, sr)
    session_metadata = dict(metadata or {})
    session_metadata['passage_mode'] = 'adaptive-15-30-causal' if adaptive else f'fixed-{seconds:g}s'
    # Build completely off-path, then publish without clobbering an existing
    # session. A crash during preparation cannot expose a half-written session.
    fd, temp = tempfile.mkstemp(prefix='.first-listen-', dir=destination.parent)
    os.close(fd)
    try:
        with sqlite3.connect(temp) as db:
            db.executescript('''
                CREATE TABLE session (version INTEGER, whole TEXT, metadata TEXT);
                CREATE TABLE passages (id INTEGER PRIMARY KEY, token TEXT UNIQUE,
                    payload TEXT NOT NULL, shown INTEGER NOT NULL DEFAULT 0,
                    note TEXT, noted_at TEXT);
            ''')
            db.execute('INSERT INTO session VALUES (1, ?, ?)',
                       (encode(whole), encode(session_metadata)))
            for i, packet in enumerate(packets, 1):
                payload = encode(packet)
                token = hashlib.sha256((str(i) + payload).encode()).hexdigest()
                db.execute('INSERT INTO passages(id,token,payload) VALUES (?,?,?)',
                           (i, token, payload))
        # Context manager commits but does not close SQLite connections.
        db.close()
        os.link(temp, destination)
    finally:
        os.unlink(temp)
    return {'ready': True, 'instruction': 'Use next. No track context revealed until finish.'}


def connect(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ListenError('session missing; prepare it first')
    db = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    if db.execute('SELECT version FROM session').fetchone()[0] != 1:
        db.close()
        raise ListenError('unsupported session version')
    return db


def next_passage(path):
    db = connect(path)
    try:
        with db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM passages WHERE note IS NULL ORDER BY id LIMIT 1').fetchone()
            if row is None:
                return {'complete': True, 'instruction': 'All notes saved. Use finish.'}
            db.execute('UPDATE passages SET shown=1 WHERE id=?', (row['id'],))
            return {'passage_id': row['id'], 'token': row['token'],
                    **json.loads(row['payload'])}
    finally:
        db.close()


def save_note(path, token, note):
    if not note.strip() or len(note) > 16000:
        raise ListenError('note must contain 1–16000 characters')
    db = connect(path)
    try:
        with db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM passages WHERE token=?', (token,)).fetchone()
            if row is None or not row['shown']:
                raise ListenError('token is not an encountered passage')
            if row['note'] is not None:
                if row['note'] != note:
                    raise ListenError('original notes cannot be rewritten')
                return {'saved': True, 'passage_id': row['id']}
            db.execute("UPDATE passages SET note=?, noted_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                       (note, row['id']))
            return {'saved': True, 'passage_id': row['id']}
    finally:
        db.close()


def finish(path):
    db = connect(path)
    try:
        if db.execute('SELECT 1 FROM passages WHERE note IS NULL LIMIT 1').fetchone():
            raise ListenError('whole-song context is withheld until every passage has a note')
        row = db.execute('SELECT * FROM session').fetchone()
        return {'complete': True, 'whole_song': json.loads(row['whole']),
                'metadata': json.loads(row['metadata']), 'journal': journal_rows(db)}
    finally:
        db.close()


def journal_rows(db):
    rows = db.execute('SELECT id,payload,note,noted_at FROM passages WHERE note IS NOT NULL ORDER BY id')
    return [{'passage_id': r['id'], 'start_s': json.loads(r['payload'])['start_s'],
             'end_s': json.loads(r['payload'])['end_s'], 'note': r['note'],
             'noted_at': r['noted_at']} for r in rows]


def journal(path):
    db = connect(path)
    try:
        return journal_rows(db)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('audio', type=Path)
    prep.add_argument('session', type=Path)
    prep.add_argument('--seconds', type=float, default=20)
    prep.add_argument('--adaptive', action='store_true',
                      help='causal 15–30s passages; closes only after an encountered change')
    for command in ['next', 'note', 'finish', 'journal']:
        sub = commands.add_parser(command)
        sub.add_argument('session', type=Path)
        if command == 'note':
            sub.add_argument('--token', required=True)
            sub.add_argument('--note-file', type=Path, help='UTF-8 file; otherwise read stdin')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            x, sr = load_audio(args.audio)
            result = prepare_signal(x, sr, args.session, args.seconds,
                                    {'source_name': args.audio.name}, adaptive=args.adaptive)
        elif args.command == 'next':
            result = next_passage(args.session)
        elif args.command == 'note':
            note = args.note_file.read_text(encoding='utf-8') if args.note_file else sys.stdin.read()
            result = save_note(args.session, args.token, note)
        elif args.command == 'finish':
            result = finish(args.session)
        else:
            result = journal(args.session)
        print(encode(result))
        return 0
    except ListenError as exc:
        print(encode({'error': str(exc)}), file=sys.stderr)
    except Exception:
        # Decoder/SQLite errors can contain filenames or hidden payloads.
        print(encode({'error': 'operation failed; check input, session integrity and local dependencies'}), file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
