import pickle
import unittest
from types import SimpleNamespace
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput
from secure_infer.reuse.connector import RuntimeWorkerMetadata
from secure_infer.reuse.legacy_metadata import RuntimeControlStats


class LegacyMetadataTests(unittest.TestCase):
    def test_all_tp_rank_completions_survive_framework_aggregation(self):
        outputs = []
        for rank in range(8):
            metadata = RuntimeWorkerMetadata(
                secure_store_ready_counts={'request': 1},
                completed_native_loads={'loaded'},
                frontiers={'mixed': {'layer_ready': 80 - rank}})
            stats = pickle.loads(pickle.dumps(RuntimeControlStats({'metadata': metadata})))
            outputs.append(SimpleNamespace(kv_connector_output=KVConnectorOutput(kv_connector_stats=stats)))
        result = KVOutputAggregator(8).aggregate(outputs)
        metadata = result.kv_connector_output.kv_connector_stats.data['metadata']
        self.assertEqual(metadata.secure_store_ready_counts, {'request': 8})
        self.assertEqual(metadata.completed_native_loads, {'loaded'})
        self.assertEqual(metadata.frontiers['mixed']['layer_ready'], 73)

    def test_workers_use_distinct_progress_temporary_files(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from secure_infer.reuse.connector import publish_progress
        state = dict(assigned=0, ready=0, pending=None, done=False,
                     error=None, loaded_blocks=0, layers=80)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'RUNTIME_RESULT_DIR': directory}):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda n: publish_progress(str(n), state), range(100)))
            self.assertEqual(json.loads((Path(directory)/'reuse_progress.json').read_text())['layers'], 80)
