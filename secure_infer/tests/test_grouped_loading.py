"""Layer-local groups must copy every K/V block exactly once."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch
import torch
from secure_infer.kv_offload.native_layerwise import NativeTransfer
class GroupedLoadingTests(unittest.TestCase):
    def test_partial_group_preserves_layer_and_mapping(self):
        ops=NativeTransfer.__new__(NativeTransfer)
        stream=object();ops.layer=7;ops.causal_group_blocks=16
        ops.streams=[stream];ops.retained=[]
        counts=[];copies=[]
        ops.begin=lambda n:counts.append(n) or 0;ops.end=lambda:0
        host=[object(),object()];destination=[object(),object()]
        mapping=torch.stack((torch.arange(41),torch.arange(41)+100),dim=1)
        def copy(src,dst,part):copies.append((host.index(src),ops.layer,part.clone()))
        with patch('secure_infer.kv_offload.native_layerwise.torch.npu.current_stream',return_value=stream), \
             patch('secure_infer.kv_offload.native_layerwise.torch.npu.stream',side_effect=lambda _:nullcontext()), \
             patch('secure_infer.kv_offload.native_layerwise.torch.ops._C_ascend.swap_blocks',side_effect=copy,create=True):
            ops.single_layer_kv_transfer((host,mapping),destination,None,False,2,True,False)
        self.assertEqual(counts,[32,32,18]);self.assertEqual(ops.layer,8)
        self.assertEqual(ops.causal_group_blocks,16)
        for kind in range(2):
            records=[part for k,layer,part in copies if k==kind]
            self.assertEqual([len(x) for x in records],[16,16,9])
            combined=torch.cat(records)
            self.assertEqual(sorted(combined[:,0].tolist()),list(range(41)))
            self.assertTrue(torch.equal(combined[:,1],combined[:,0]+100))
        self.assertTrue(all(layer==7 for _,layer,_ in copies))
if __name__=='__main__':unittest.main()
