"""Exercise the real loader's STOP handoff without allocating device memory."""
from contextlib import nullcontext
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
import os,tempfile,threading,unittest
from secure_infer.reuse.connector import RuntimeWorker

class HandoffTests(unittest.TestCase):
    def test_stop_fills_earliest_missing_layer_before_finishing_tail(self):
        worker=RuntimeWorker.__new__(RuntimeWorker)
        worker.lock=threading.Condition();worker.newly_done=set()
        worker.handler=SimpleNamespace(npu_tensors=[(SimpleNamespace(device='cpu'),)]*40,
                                        cpu_tensors=[object()]*40)
        worker.names=list(range(40))
        worker.native=SimpleNamespace(load_stream=object(),kvcaches=[object()]*40,
                                     kv_format=SimpleNamespace(value=2),vllm_two_major=False)
        state=dict(R=6144,source=list(range(48)),destination=list(range(48)),ready=6144,
            assigned=6144,pending=None,target=4096,compute_end=1024,stop=False,error=None,
            submitted=[6144]*40,verified=[6144]*40,events=deque(),submissions_done=False,
            ready_offsets=[0.]*40,layer_ready=0,layers=40,loaded_blocks=0,done=False)
        copies=[]
        def copy(payload,*args):
            mapping=payload[1]
            copies.append((worker.ops.layer,int(mapping[0,0])*128,(int(mapping[-1,0])+1)*128))
            if len(copies)==16:
                with worker.lock:state['target']=1024;state['stop']=True
        worker.ops=SimpleNamespace(layer=0,retained=[],single_layer_kv_transfer=copy)
        worker.wait_auth=lambda layer:0
        worker.retain_copy_inputs=lambda:None
        class Event:
            def record(self,*args):pass
            def synchronize(self):pass
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,RUNTIME_RESULT_DIR=directory,RUNTIME_CHECK_LOADED_BYTES='0'), \
                 patch('secure_infer.reuse.connector.torch.npu.set_device'), \
                 patch('secure_infer.reuse.connector.torch.npu.stream',return_value=nullcontext()), \
                 patch('secure_infer.reuse.connector.torch.npu.Event',side_effect=Event), \
                 patch('secure_infer.reuse.connector._retire_h2d_prediction_scope'):
                worker.fetch('normal-request',state)
        self.assertIsNone(state['error']);self.assertTrue(state['done'])
        self.assertEqual(state['verified'],[1024]*40)
        self.assertEqual(copies[16],(0,1024,4096))
        for layer in range(40):
            ranges=sorted((a,b) for i,a,b in copies if i==layer)
            self.assertEqual(ranges[0][0],1024);self.assertEqual(ranges[-1][1],6144)
            self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))
        self.assertEqual(sum((b-a)//128*2 for _,a,b in copies),3200)

if __name__=='__main__':unittest.main()
