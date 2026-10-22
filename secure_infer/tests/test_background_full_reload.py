"""Full reload must use background submission without losing layer gating."""
from types import SimpleNamespace
from unittest.mock import patch,Mock
import threading,unittest
from secure_infer.reuse.connector import RuntimeWorker,RuntimeMetadata
class BackgroundFullReloadTests(unittest.TestCase):
    def test_full_reload_keeps_mapping_and_completion_dependency(self):
        worker=RuntimeWorker.__new__(RuntimeWorker);worker.cake_jobs={};worker.lock=threading.Condition()
        worker.names=list(range(40));worker.handler=SimpleNamespace(block_size_factor=1);worker.cake_consuming=None
        metadata=RuntimeMetadata(reqs_to_load={'normal-request':(SimpleNamespace(block_ids=[2,3]),SimpleNamespace(block_ids=[7,8]))},reqs_to_store={})
        sender=Mock()
        with patch('secure_infer.reuse.connector.NativeWorker.start_kv_transfers') as base, \
             patch('secure_infer.reuse.connector.publish_progress'),patch('secure_infer.reuse.connector.trace'), \
             patch('secure_infer.reuse.connector.threading.Thread',return_value=sender):
            worker.start_kv_transfers(metadata)
        self.assertEqual(base.call_args.args[0].reqs_to_load,{})
        state=worker.cake_jobs['normal-request']
        self.assertEqual((state['R'],state['target'],state['planned_target'],state['stop']),(256,0,0,True))
        self.assertEqual(state['source'],[2,3]);self.assertEqual(state['destination'],[7,8])
        self.assertEqual(worker.cake_consuming,('normal-request',state,0));sender.start.assert_called_once()
        self.assertEqual(state['verified'],[256]*40)
if __name__=='__main__':unittest.main()
