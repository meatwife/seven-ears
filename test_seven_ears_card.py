import unittest

from seven_ears_card import (
    format_discord_report,
    pace_profile,
    quality_notes,
    timeline_sentences,
    utterance_segments,
    voice_timeline,
    word_count,
)


class SevenEarsCardTests(unittest.TestCase):
    def test_utterance_segments_merge_close_micro_segments(self):
        self.assertEqual(
            utterance_segments([(1.0, 1.2), (1.5, 1.8), (3.0, 3.2)]),
            [(1.0, 1.8), (3.0, 3.2)],
        )

    def test_utterance_segments_keep_long_gap_separate(self):
        self.assertEqual(
            utterance_segments([(1.0, 1.2), (2.2, 2.6)]),
            [(1.0, 1.2), (2.2, 2.6)],
        )

    def test_quality_notes_short_and_no_transcript(self):
        notes = quality_notes(1.98, 0.45, '', 'none')
        self.assertTrue(any('very short' in n for n in notes))
        self.assertTrue(any('no transcript' in n for n in notes))

    def test_word_count_handles_punctuation(self):
        self.assertEqual(word_count("Testing, testing, testicles testable."), 4)

    def test_discord_report_includes_guardrail_and_phrase_layer(self):
        report = format_discord_report({
            'file': 'sample.ogg',
            'transcript': 'hello pause hello',
            'transcript_source': 'manual',
            'duration_s': 10.0,
            'active_s': 3.0,
            'utterance_segments': [(1.0, 2.0), (4.0, 5.0)],
            'quality_notes': ['sample note'],
            'pace': {
                'whole_clip': {'wpm': 90, 'basis_s': 10.0},
                'speech_span': {'wpm': 120, 'basis_s': 4.0},
                'active_voice': {'wpm': 300, 'basis_s': 3.0},
            },
            'pitch': {'available': True, 'median_hz': 132.0, 'p10_hz': 110.0, 'p90_hz': 220.0},
            'acoustic': {'brightness_hz': 2500, 'brightness_label': 'bright', 'dynamics_label': 'dynamic', 'dynamic_range_db': 40.0},
        })
        self.assertIn('Seven Ears, spoken-voice card', report)
        self.assertIn('2 phrase-shaped spans by the RMS detector', report)
        self.assertIn('No mind-reading', report)
        self.assertIn('what the sound gives me evidence for', report)

    def test_voice_timeline_from_word_timestamps(self):
        words = [
            {'word': 'hello', 'start_s': 1.2, 'end_s': 1.6},
            {'word': 'there', 'start_s': 1.7, 'end_s': 2.0},
            {'word': 'friend', 'start_s': 4.0, 'end_s': 4.5},
        ]
        tl = voice_timeline(6.5, words=words)
        self.assertEqual(tl['source'], 'word_timestamps')
        self.assertEqual(tl['leading_quiet_s'], 1.2)
        self.assertEqual(tl['trailing_quiet_s'], 2.0)
        self.assertEqual(tl['phrase_count'], 2)
        self.assertEqual(len(tl['held_silences']), 1)
        self.assertEqual(tl['held_silences'][0]['dur_s'], 2.0)

    def test_voice_timeline_falls_back_to_rms_segments(self):
        tl = voice_timeline(5.0, words=None, utterances=[(0.5, 1.5), (3.0, 4.0)])
        self.assertEqual(tl['source'], 'rms_segments')
        self.assertEqual(tl['phrase_count'], 2)

    def test_voice_timeline_none_without_evidence(self):
        self.assertIsNone(voice_timeline(5.0, words=None, utterances=None))

    def test_timeline_sentences_are_evidence_shaped(self):
        tl = voice_timeline(6.5, words=[
            {'word': 'hi', 'start_s': 1.2, 'end_s': 1.4},
            {'word': 'again', 'start_s': 3.4, 'end_s': 4.5},
        ])
        sentences = ' '.join(timeline_sentences(tl))
        self.assertIn('You waited 1.2 seconds before speaking.', sentences)
        self.assertIn('2 phrases, with a 2.0-second held silence between them.', sentences)
        self.assertIn('You stayed on the mic 2.0 seconds after the last word.', sentences)

    def test_timeline_sentences_single_phrase(self):
        tl = voice_timeline(3.0, words=[
            {'word': 'hi', 'start_s': 0.1, 'end_s': 0.3},
            {'word': 'there', 'start_s': 0.4, 'end_s': 0.8},
        ])
        sentences = ' '.join(timeline_sentences(tl))
        self.assertIn('One continuous phrase, no held silences.', sentences)
        self.assertNotIn('You waited', sentences)

    def test_pace_profile_includes_word_span_basis(self):
        words = [
            {'word': 'one', 'start_s': 1.0, 'end_s': 1.3},
            {'word': 'two', 'start_s': 1.4, 'end_s': 1.7},
            {'word': 'three', 'start_s': 1.8, 'end_s': 2.0},
        ]
        profile = pace_profile('one two three', 10.0, 1.0, [(1.0, 2.0)], words)
        self.assertEqual(profile['word_span']['basis_s'], 1.0)
        self.assertEqual(profile['word_span']['wpm'], 180)

    def test_discord_report_uses_timeline_evidence(self):
        words = [
            {'word': 'hi', 'start_s': 1.2, 'end_s': 1.4},
            {'word': 'again', 'start_s': 3.4, 'end_s': 4.5},
        ]
        report = format_discord_report({
            'file': 'sample.ogg',
            'transcript': 'hi again',
            'transcript_source': 'stt_whisper',
            'duration_s': 6.5,
            'active_s': 1.3,
            'utterance_segments': [(1.2, 1.4), (3.4, 4.5)],
            'timeline': voice_timeline(6.5, words=words),
            'quality_notes': [],
            'pace': pace_profile('hi again', 6.5, 1.3, [(1.2, 1.4), (3.4, 4.5)], words),
            'pitch': {'available': False},
            'acoustic': {'brightness_hz': 2500, 'brightness_label': 'bright', 'dynamics_label': 'dynamic', 'dynamic_range_db': 40.0},
        })
        self.assertIn('You waited 1.2 seconds before speaking.', report)
        self.assertIn('held silence', report)
        self.assertIn('word timestamps', report)
        self.assertNotIn('Calm', report)
        self.assertNotIn('stress', report)


    def test_discord_report_includes_ear_line_when_present(self):
        base = {
            'file': 'sample.ogg',
            'transcript': 'hey', 'transcript_source': 'manual',
            'duration_s': 3.0, 'active_s': 1.0, 'utterance_segments': [(0.5, 1.5)],
            'quality_notes': [],
            'pace': {'whole_clip': {'wpm': 60, 'basis_s': 3.0}},
            'pitch': {'available': False},
            'acoustic': {'brightness_hz': 2000, 'brightness_label': 'warm', 'dynamics_label': 'even', 'dynamic_range_db': 20.0},
        }
        with_ear = format_discord_report({**base, 'ear': 'for speaker: softer than usual'})
        self.assertIn('softer than usual', with_ear)
        self.assertIn('relative to recent notes', with_ear)
        without_ear = format_discord_report(base)
        self.assertNotIn('**Ear**', without_ear)


if __name__ == '__main__':
    unittest.main()
