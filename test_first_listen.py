import concurrent.futures
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import seven_ears_first_listen as fl

SR = 22050


def tone(seconds, hz=220, amp=0.1):
    return amp * np.sin(2 * np.pi * hz * np.arange(round(SR * seconds)) / SR)


class FirstListenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'session.sqlite'

    def prepare(self, x=None):
        fl.prepare_signal(tone(10.2) if x is None else x, SR, self.path,
                          seconds=5, metadata={'title': 'HIDDEN ENDING'})

    def test_different_futures_cannot_change_first_packet(self):
        prefix = tone(5)
        self.prepare(np.concatenate([prefix, tone(5, 330, 0.001)]))
        other = Path(self.tmp.name) / 'other.sqlite'
        fl.prepare_signal(np.concatenate([prefix, np.ones(SR * 12)]), SR, other, seconds=5)
        a, b = fl.next_passage(self.path), fl.next_passage(other)
        self.assertEqual(a, b)
        self.assertNotIn('HIDDEN', fl.encode(a))
        self.assertNotIn('metadata', a)
        self.assertNotIn('total', a)

    def test_card_exposes_existing_passage_detail(self):
        self.prepare(tone(5))
        card = fl.next_passage(self.path)['card']
        self.assertIn('Band balance: low ', card)
        self.assertIn('Dynamics:', card)
        self.assertIn('Activity:', card)

    def test_restart_replays_and_note_retry_does_not_skip(self):
        self.prepare()
        first = fl.next_passage(self.path)
        self.assertEqual(first, fl.next_passage(self.path))
        with self.assertRaises(fl.ListenError):
            fl.finish(self.path)
        with self.assertRaises(fl.ListenError):
            fl.save_note(self.path, first['token'], ' ')
        fl.save_note(self.path, first['token'], 'I expect a return.')
        second = fl.next_passage(self.path)
        fl.save_note(self.path, first['token'], 'I expect a return.')
        self.assertEqual(second, fl.next_passage(self.path))
        with self.assertRaises(fl.ListenError):
            fl.save_note(self.path, first['token'], 'Actually I knew the ending.')
        self.assertEqual(len(fl.journal(self.path)), 1)

    def test_finish_waits_for_final_note_and_preserves_order(self):
        self.prepare()
        for i in range(1, 4):
            packet = fl.next_passage(self.path)
            with self.assertRaises(fl.ListenError):
                fl.finish(self.path)
            fl.save_note(self.path, packet['token'], f'Impression {i}')
        self.assertTrue(fl.next_passage(self.path)['complete'])
        final = fl.finish(self.path)
        self.assertEqual(final['metadata']['title'], 'HIDDEN ENDING')
        self.assertEqual([r['note'] for r in final['journal']],
                         ['Impression 1', 'Impression 2', 'Impression 3'])
        self.assertAlmostEqual(final['journal'][-1]['end_s'], 10.2)

    def test_unseen_token_rejected(self):
        self.prepare()
        db = fl.connect(self.path)
        token = db.execute('SELECT token FROM passages WHERE id=2').fetchone()[0]
        db.close()
        with self.assertRaises(fl.ListenError):
            fl.save_note(self.path, token, 'skip ahead')
        with self.assertRaises(fl.ListenError):
            fl.save_note(self.path, 'unknown', 'skip ahead')

    def test_concurrent_retry_is_one_note(self):
        self.prepare()
        packet = fl.next_passage(self.path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: fl.save_note(self.path, packet['token'], 'same note'), range(4)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(len(fl.journal(self.path)), 1)
        self.assertEqual(fl.next_passage(self.path)['passage_id'], 2)

    def test_no_overwrite_and_private_permissions(self):
        self.prepare()
        before = self.path.read_bytes()
        with self.assertRaises(fl.ListenError):
            self.prepare()
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_failed_publish_leaves_no_session(self):
        with patch.object(fl.os, 'link', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.prepare(tone(0.1))
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_silence_short_tail_and_invalid_input(self):
        for samples in [np.zeros(1), tone(0.01), np.zeros(SR * 6)]:
            path = Path(self.tmp.name) / f'{len(samples)}.sqlite'
            fl.prepare_signal(samples, SR, path, seconds=5)
            while True:
                packet = fl.next_passage(path)
                if packet.get('complete'):
                    break
                self.assertNotIn('NaN', fl.encode(packet))
                fl.save_note(path, packet['token'], 'Little evidence.')
            self.assertTrue(fl.finish(path)['complete'])
        for x in [[], [float('nan')], [[1, 2]]]:
            with self.assertRaises(fl.ListenError):
                fl.prepare_signal(x, SR, self.path)
        for seconds in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(fl.ListenError):
                fl.prepare_signal(tone(1), SR, self.path, seconds=seconds)


if __name__ == '__main__':
    unittest.main()
