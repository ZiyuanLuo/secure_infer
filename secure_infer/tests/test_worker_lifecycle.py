"""Lifecycle checks for the integrated worker, without allocating NPU memory."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from secure_infer.reuse.connector import RuntimeWorker
from secure_infer.kv_offload.native_layerwise import NativeWorker

class WorkerLifecycleTests(unittest.TestCase):
    def test_completed_job_is_retired_after_reporting_completion(self):
        worker=RuntimeWorker.__new__(RuntimeWorker)
        worker.lock=threading.Condition()
        worker.newly_done={'normal-request'}
        worker.cake_jobs={'normal-request':dict(assigned=0,ready=0,pending=None,
            done=True,error=None,loaded_blocks=8,layers=40)}
        with patch.object(NativeWorker,'take_secure_store_metadata',return_value=None):
            result=worker.take_secure_store_metadata()
        self.assertIn('normal-request',result.completed_native_loads)
        self.assertIn('normal-request',result.frontiers)
        self.assertEqual(worker.cake_jobs,{})
        self.assertEqual(worker.newly_done,set())

    def test_copy_inputs_are_retained_until_stream_event(self):
        worker=NativeWorker.__new__(NativeWorker)
        worker.native=SimpleNamespace(load_stream=object())
        references=[object()]
        worker.ops=SimpleNamespace(retained=references)
        worker.retained=[]
        with patch('secure_infer.kv_offload.native_layerwise.torch.npu.Event') as event:
            worker.retain_copy_inputs()
            event.return_value.record.assert_called_once_with(worker.native.load_stream)
            self.assertIs(worker.retained[0][1],references)
            self.assertEqual(worker.ops.retained,[])

if __name__=='__main__':unittest.main()
