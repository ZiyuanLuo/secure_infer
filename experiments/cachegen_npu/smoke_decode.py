"""Validate fused entropy decoding and BF16 restoration on a real NPU."""
import argparse
import ctypes as C
import json
import time
import zlib
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--lanes',type=int,default=1)
args=parser.parse_args()
assert args.lanes in (1,2,4,8)
lib=C.CDLL(str(Path('codec_build/libcachegen_entropy_restore.so').resolve()))
restore=lib.entropy_restore_do
restore.argtypes=[C.c_void_p]*5+[C.c_uint32]*3
restore.restype=C.c_int
import torch
import torch_npu
torch.npu.set_device(0)
rows=[]
for name in ('K','V'):
 raw=np.fromfile('data/'+name+'.bf16',dtype='<u2')
 source=(raw.astype(np.uint32)<<16).view(np.float32).reshape(-1,1024)
 for ntokens in (128,512,2048):
  x=source[:ntokens]
  fixtures=[]
  tile=ntokens//args.lanes
  for lane in range(args.lanes):
   values=x[lane*tile:(lane+1)*tile]
   scale=np.max(np.abs(values),axis=-1,keepdims=True)/127
   symbols=np.clip(np.rint(values/np.where(scale>0,scale,1))+127,0,254).astype(np.uint8)
   compressed=zlib.compress(symbols.tobytes(),1)
   payload=scale.astype('<f4').tobytes()+compressed
   inp=torch.from_numpy(np.frombuffer(payload,np.uint8).copy()).npu()
   out=torch.empty((tile,1024),dtype=torch.bfloat16,device='npu')
   scratch=torch.empty((tile,1024),dtype=torch.uint8,device='npu')
   status=torch.ones(1,dtype=torch.int32,device='npu')
   stream=torch.npu.Stream()
   expected=torch.from_numpy((symbols.astype(np.int16)-127).astype(np.float32)*scale).to(torch.bfloat16)
   fixtures.append((inp,out,scratch,status,stream,expected,len(compressed),len(payload)))
  torch.npu.synchronize()
  times=[]
  for repeat in range(4):
   start=time.perf_counter()
   for inp,out,scratch,status,stream,expected,length,payloadlen in fixtures:
    assert restore(stream.npu_stream,inp.data_ptr(),out.data_ptr(),scratch.data_ptr(),status.data_ptr(),tile,1024,length)==0
   for fixture in fixtures: fixture[4].synchronize()
   times.append((time.perf_counter()-start)*1000)
  for inp,out,scratch,status,stream,expected,length,payloadlen in fixtures:
   assert status.item()==1
   assert torch.equal(out.cpu().view(torch.int16),expected.view(torch.int16)),(name,ntokens)
  wire=sum((f[7]+15)//16*16+80 for f in fixtures)
  row=dict(kind=name,tokens=ntokens,lanes=args.lanes,ratio=x.size*2/wire,
           device_restore_ms=times[1:],bf16_matches_quantized_reference=True,
           model_quality_evaluated=False)
  rows.append(row)
  print(json.dumps(row),flush=True)
Path(f'results_cachegen/device_decode_lanes{args.lanes}.json').write_text(json.dumps(rows,indent=2)+'\n')
