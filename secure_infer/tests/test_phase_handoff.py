"""A chunk wave must hand off to layerwise gap filling at STOP."""
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import os,tempfile,threading,time,unittest
from secure_infer.reuse.connector import RuntimeWorker
class PhaseHandoffTests(unittest.TestCase):
    def test_reserved_chunk_then_stop_fills_all_layers_without_overlap(self):
        worker=RuntimeWorker.__new__(RuntimeWorker);worker.lock=threading.Condition();worker.newly_done=set()
        worker.handler=SimpleNamespace(npu_tensors=[(SimpleNamespace(device='cpu'),)]*40,cpu_tensors=[object()]*40)
        worker.names=list(range(40));worker.native=SimpleNamespace(load_stream=object(),kvcaches=[object()]*40,kv_format=SimpleNamespace(value=2),vllm_two_major=False)
        state=dict(R=6144,source=list(range(48)),destination=list(range(48)),ready=6144,
          assigned=6144,pending=None,target=None,planned_target=4096,compute_end=1024,
          stop=False,error=None,submitted=[6144]*40,verified=[6144]*40,events=deque(),
          submissions_done=False,ready_offsets=[0.]*40,layer_ready=0,layers=40,loaded_blocks=0,done=False)
        copies=[];evidence=[]
        def copy(payload,*args):
            m=payload[1];copies.append((worker.ops.layer,int(m[0,0])*128,(int(m[-1,0])+1)*128))
        worker.ops=SimpleNamespace(layer=0,retained=[],path_counts=lambda:{},single_layer_kv_transfer=copy)
        worker.wait_auth=lambda layer:0;worker.retain_copy_inputs=lambda:None
        class Event:
            def record(self,*args):pass
            def synchronize(self):pass
        def stop_after_completed_chunk():
            deadline=time.monotonic()+5
            with worker.lock:
                while state['ready']>4096 and not state['error']:
                    if time.monotonic()>deadline:
                        state['error']='chunk did not complete';worker.lock.notify_all();return
                    worker.lock.wait(timeout=.1)
                evidence.extend(copies)
                state['target']=1024;state['stop']=True;worker.lock.notify_all()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,RUNTIME_RESULT_DIR=directory,RUNTIME_CHECK_LOADED_BYTES='0'), \
                 patch('secure_infer.reuse.connector.torch.npu.set_device'), \
                 patch('secure_infer.reuse.connector.torch.npu.stream',return_value=nullcontext()), \
                 patch('secure_infer.reuse.connector.torch.npu.Event',side_effect=Event), \
                 patch('secure_infer.reuse.connector._retire_h2d_prediction_scope'):
                control=threading.Thread(target=stop_after_completed_chunk);control.start()
                worker.fetch('ordinary-request',state);control.join(timeout=1)
        self.assertFalse(control.is_alive());self.assertIsNone(state['error']);self.assertTrue(state['done'])
        self.assertEqual(evidence,[(i,4096,6144) for i in range(40)])
        self.assertEqual(state['verified'],[1024]*40)
        for i in range(40):self.assertEqual(sorted((a,b) for layer,a,b in copies if layer==i),[(1024,4096),(4096,6144)])
if __name__=='__main__':unittest.main()
