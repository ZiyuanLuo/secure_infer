"""Calibration samples from tensor-parallel timelines must exclude layer waits seen by any rank."""
import json
import os
import tempfile
import unittest

from secure_infer.reuse import entry_gate


def _row(rank, world, gates, request_id='req'):
    return dict(request_id=request_id, layers=2, rank=rank, world_size=world, timing_window='model_forward',
                steps=[dict(position=0, tokens=1024, start_ms=0., end_ms=100., stream_span_ms=100.),
                       dict(position=1024, tokens=2048, start_ms=100., end_ms=2183., stream_span_ms=2083.)],
                gates=[dict(layer=i, start_ms=g[0], end_ms=g[1], stream_span_ms=g[1]-g[0], host_callback_ms=0.)
                       for i, g in enumerate(gates)],
                loads=[dict(layer=i, blocks=8, start_ms=0., end_ms=50., stream_span_ms=50., host_submit_ms=50.)
                       for i in range(2)])


class TensorParallelCalibrationTests(unittest.TestCase):
    def ingest(self, rows):
        entry_gate._compute.clear()
        entry_gate._loads.clear()
        entry_gate._rank_timelines.clear()
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get('RUNTIME_RESULT_DIR')
            os.environ['RUNTIME_RESULT_DIR'] = directory
            entry_gate._timeline_offset = 0
            try:
                with open(os.path.join(directory, 'device_timeline.jsonl'), 'w') as stream:
                    for row in rows:
                        stream.write(json.dumps(row)+'\n')
                entry_gate.update()
                return [dict(s) for s in entry_gate._compute]
            finally:
                entry_gate._compute.clear()
                entry_gate._loads.clear()
                entry_gate._timeline_offset = 0
                if previous is None:
                    os.environ.pop('RUNTIME_RESULT_DIR')
                else:
                    os.environ['RUNTIME_RESULT_DIR'] = previous

    def test_wait_recorded_on_one_rank_is_removed(self):
        # Only rank 0 saw the 1411 ms of layer waits; every rank's step span contains them.
        waits = [(150., 900.), (1000., 1661.)]
        rows = [_row(0, 4, waits)] + [_row(r, 4, []) for r in (1, 2, 3)]
        samples = self.ingest(rows)
        self.assertEqual([s['tokens'] for s in samples], [1024, 2048])
        self.assertAlmostEqual(samples[0]['service_ms'], 100.)
        self.assertAlmostEqual(samples[1]['service_ms'], 2083.-1411.)

    def test_observing_rank_does_not_matter(self):
        waits = [(150., 900.), (1000., 1661.)]
        for observer in range(4):
            rows = [_row(r, 4, waits if r == observer else []) for r in range(4)]
            self.assertAlmostEqual(self.ingest(rows)[1]['service_ms'], 2083.-1411.)

    def test_longest_recorded_wait_is_used(self):
        rows = [_row(0, 4, [(150., 450.)]), _row(1, 4, [(150., 900.)]), _row(2, 4, []), _row(3, 4, [(150., 300.)])]
        self.assertAlmostEqual(self.ingest(rows)[1]['service_ms'], 2083.-750.)

    def test_single_rank_unchanged(self):
        samples = self.ingest([_row(0, 1, [(150., 900.)])])
        self.assertAlmostEqual(samples[1]['service_ms'], 2083.-750.)

    def test_incomplete_request_is_not_ingested(self):
        self.assertEqual(self.ingest([_row(0, 4, []), _row(1, 4, [])]), [])


if __name__ == '__main__':
    unittest.main()
