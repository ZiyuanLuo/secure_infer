"""Real secure-transfer component benchmark; GEMM consumer, not model E2E."""
import argparse
import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path
import statistics
import struct
import time
import zlib
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--tokens',type=int,default=128)
parser.add_argument('--blocks',type=int,default=12)
parser.add_argument('--compute-repeats',type=int,default=1)
parser.add_argument('--repeats',type=int,default=3)
args=parser.parse_args()
crypto=C.CDLL(str(Path('crypto_build/libruntime_aicpu_ghash.so').resolve()))
codec=C.CDLL(str(Path('codec_build/libcachegen_entropy_restore.so').resolve()))
driver=C.CDLL(str(Path('codec_build/libentropy_driver.so').resolve()))
lz=C.CDLL(ctypes.util.find_library('lz4'))
lz.LZ4_compress_HC.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
class Entry(C.Structure):
 _fields_=[(s,C.c_void_p) for s in ('host','cipher','key','hash','tag','status','aad','plain','restored','scratch','answer')]+[('bytes',C.c_uint64)]+[(s,C.c_uint32) for s in ('tiles','tile','bits','elements')]+[('iv',C.c_uint32*4)]
driver.run_entropy_pipeline.argtypes=[C.POINTER(Entry),C.c_int]+[C.c_void_p]*4+[C.c_int]*3+[C.POINTER(C.c_double),C.POINTER(C.c_float)]
driver.run_entropy_pipeline.restype=C.c_int
import torch
import torch_npu
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
torch.set_num_threads(4)
torch.npu.set_device(0)
key=os.urandom(16)
ecb=Cipher(algorithms.AES(key),modes.ECB()).encryptor()
hash_subkey=ecb.update(bytes(16))+ecb.finalize()
def device_bytes(blob): return torch.from_numpy(np.frombuffer(blob,np.uint8).copy()).npu()
key_d=device_bytes(key);hash_d=device_bytes(hash_subkey)
query=torch.randn((128,128),dtype=torch.bfloat16,device='npu')
a_stream=torch.npu.Stream();c_stream=torch.npu.Stream();d_streams=[torch.npu.Stream() for _ in range(8)]
raw={name:np.fromfile('data/'+name+'.bf16',dtype='<u2') for name in ('K','V')}
rows=[]
for mode_id,mode in enumerate(('bf16','int8','deflate','lz4','bslz4')):
 fixtures=[];encode_ms=0.
 for i in range(args.blocks):
  name='K' if i%2==0 else 'V'
  nt=args.tokens;n=nt*1024
  offset=(i*n)%(raw[name].size-n+1)
  x=(raw[name][offset:offset+n].astype(np.uint32)<<16).view(np.float32).reshape(nt,1024)
  begin=time.perf_counter()
  scale=np.max(np.abs(x),axis=-1,keepdims=True)/127
  ids=np.clip(np.rint(x/np.where(scale>0,scale,1))+127,0,254).astype(np.uint8)
  expected=torch.from_numpy((ids.astype(np.int16)-127).astype(np.float32)*scale).to(torch.bfloat16)
  flag=0
  if mode=='bf16':payload=expected.view(torch.int16).numpy().astype('<i2').tobytes()
  else:
   if mode=='bslz4':
    symbols=np.packbits(np.unpackbits(ids.reshape(-1,128,1),axis=2,bitorder='little').transpose(0,2,1),axis=2,bitorder='little').tobytes()
   else:symbols=ids.tobytes()
   if mode=='int8':data=symbols;flag=0x80000000|len(data)
   elif mode=='deflate':data=zlib.compress(symbols,1);flag=len(data)
   else:
    output=C.create_string_buffer(len(symbols)*2)
    length=lz.LZ4_compress_HC(symbols,output,len(symbols),len(output),1)
    assert length>0
    data=output.raw[:length];flag=0x40000000|len(data)
    if mode=='bslz4':flag|=0x20000000
   payload=scale.astype('<f4').tobytes()+data
  encode_ms+=(time.perf_counter()-begin)*1000
  aad=struct.pack('<8Q',1,i,n*2,len(payload),nt,1024,mode_id,flag)
  nonce=struct.pack('>QI',mode_id+1,i)
  protected=AESGCM(key).encrypt(nonce,payload,aad)
  host=torch.from_numpy(np.frombuffer(protected[:-16],np.uint8).copy()).pin_memory()
  cipher=torch.empty(len(payload),dtype=torch.uint8,device='npu')
  plain=torch.empty_like(cipher)
  restored=plain.view(torch.bfloat16) if mode=='bf16' else torch.empty((nt,1024),dtype=torch.bfloat16,device='npu')
  fixtures.append(dict(host=host,cipher=cipher,plain=plain,restored=restored,
     scratch=torch.empty(n,dtype=torch.uint8,device='npu'),
     tag=device_bytes(protected[-16:]),aad=device_bytes(aad),
     status=torch.zeros(1,dtype=torch.int32,device='npu'),
     answer=torch.empty((128,n//128),dtype=torch.bfloat16,device='npu'),
     flag=flag,expected=expected,iv=struct.unpack('>4I',nonce+struct.pack('>I',2))))
 entries=(Entry*len(fixtures))()
 for i,f in enumerate(fixtures):
  entries[i]=Entry(f['host'].data_ptr(),f['cipher'].data_ptr(),key_d.data_ptr(),hash_d.data_ptr(),
     f['tag'].data_ptr(),f['status'].data_ptr(),f['aad'].data_ptr(),f['plain'].data_ptr(),
     f['restored'].data_ptr(),f['scratch'].data_ptr(),f['answer'].data_ptr(),
     f['cipher'].numel(),args.tokens,1024,f['flag'],args.tokens*1024,(C.c_uint32*4)(*f['iv']))
 for pipeline,lanes in ((0,1),(1,1),(1,4)):
  streams=(C.c_void_p*lanes)(*[s.npu_stream for s in d_streams[:lanes]])
  samples=[]
  torch.npu.synchronize()
  for r in range(args.repeats+1):
   timings=(C.c_double*2)()
   rc=driver.run_entropy_pipeline(entries,len(fixtures),query.data_ptr(),a_stream.npu_stream,
        C.cast(streams,C.c_void_p),c_stream.npu_stream,pipeline,args.compute_repeats,lanes,timings,None)
   assert rc==0,rc
   if r: samples.append(timings[1])
  for f in fixtures:
   assert f['status'].item()==1,(mode,'GCM failed')
   assert torch.equal(f['restored'].reshape_as(f['expected']).cpu().view(torch.int16),f['expected'].view(torch.int16)),(mode,'restore mismatch')
  wire=sum(((f['host'].numel()+15)//16*16)+80 for f in fixtures)
  row=dict(mode=mode,pipeline=bool(pipeline),decode_streams=lanes,tokens_per_block=args.tokens,
     blocks=args.blocks,compute_repeats=args.compute_repeats,median_ms=statistics.median(samples),
     samples_ms=samples,encode_ms=encode_ms,ratio=args.tokens*1024*2*args.blocks/wire,
     authenticated=True,quantized_reference_exact=True,full_model_e2e=False)
  rows.append(row);print(json.dumps(row),flush=True)
 Path('results_cachegen/pipeline_t%d_c%d.json'%(args.tokens,args.compute_repeats)).write_text(json.dumps(rows,indent=2)+'\n')
