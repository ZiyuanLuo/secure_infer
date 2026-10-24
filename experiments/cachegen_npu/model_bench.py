"""Full-model prefix reuse with real secure H2D and layerwise KV restore.

Uses a process-local random benchmark key, not a provisioned TEE session.
No synthetic compute stage. Encoder cost and target request cost are separate.
"""
import argparse
import ctypes as C
import ctypes.util
import gc
import json
import os
from pathlib import Path
import struct
import threading
import time
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--text-file',default='model_bench.py')
parser.add_argument('--prefix',type=int,default=2048)
parser.add_argument('--suffix',type=int,default=512)
parser.add_argument('--output-tokens',type=int,default=16)
parser.add_argument('--chunk',type=int,default=512)
parser.add_argument('--lookahead',type=int,default=3)
parser.add_argument('--modes',default='plain,bf16,lz4')
parser.add_argument('--output',default='model_results.json')
args=parser.parse_args()
assert args.prefix>0 and args.suffix>0 and args.output_tokens>0
crypto=C.CDLL(str(Path('crypto_build/libruntime_aicpu_ghash.so').resolve()))
codec=C.CDLL(str(Path('codec_build/libcachegen_entropy_restore.so').resolve()))
verify=crypto.secure_infer_aicpu_gcm_verify_only_aad_do
verify.argtypes=[C.c_void_p]*6+[C.c_uint64]+[C.c_uint32]*4+[C.c_void_p,C.c_uint64]
decrypt=crypto.secure_infer_aicpu_aes_ctr_do
decrypt.argtypes=[C.c_void_p]*4+[C.c_uint64]+[C.c_uint32]*4
restore=codec.entropy_restore_do
restore.argtypes=[C.c_void_p]*5+[C.c_uint32]*3
lz=C.CDLL(ctypes.util.find_library('lz4'))
native_encoder=None
if Path('libhost_encoder.so').exists():
 native_encoder=C.CDLL(str(Path('libhost_encoder.so').resolve()))
 native_encoder.paci_encode_symbols.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32]+[C.c_void_p]*4+[C.c_int,C.c_int]
lz.LZ4_compress_HC.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
import torch
import torch_npu
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
torch.set_num_threads(4)
torch.manual_seed(123)
torch.npu.set_device(0)
key=os.urandom(16)
ecb=Cipher(algorithms.AES(key),modes.ECB()).encryptor()
h=ecb.update(bytes(16))+ecb.finalize()
def dev(blob):return torch.from_numpy(np.frombuffer(blob,np.uint8).copy()).npu()
kd=dev(key);hd=dev(h)
nonce_counter=0

def prepare_cache(cpu_cache,mode):
 global nonce_counter
 started=time.perf_counter(); layers=[]; raw_bytes=wire=0
 encode_ms=seal_ms=allocate_ms=0.
 for index,(k,v) in enumerate(cpu_cache):
  output=[]; streams=[torch.npu.Stream() for _ in range(4)]
  status_count=2*((len(k)+args.chunk-1)//args.chunk)
  statuses=torch.ones(status_count,dtype=torch.int32,device='npu')
  record_index=0
  for kind,values in enumerate((k,v)):
   nt,heads,dim=values.shape; channels=heads*dim
   destination=torch.empty((nt,heads,dim),dtype=torch.bfloat16,device='npu')
   # Reuse host workspaces across records; their contents are copied into
   # immutable payload bytes before the next encode call.
   if mode=='bslz4_fast':
    scale_work=np.empty(args.chunk,dtype=np.float32)
    ids_work=np.empty(args.chunk*channels,dtype=np.uint8)
    shuffled_work=np.empty_like(ids_work)
    encoded_work=C.create_string_buffer(ids_work.size*2)
   for start in range(0,nt,args.chunk):
    encode_started=time.perf_counter()
    count=min(args.chunk,nt-start)
    raw=values[start:start+count].contiguous().view(torch.int16).numpy().view(np.uint16)
    flag=0
    if mode=='bslz4_fast':
     if native_encoder is None:raise RuntimeError('libhost_encoder.so is required')
     scales=scale_work[:count]
     size=native_encoder.paci_encode_symbols(raw.ctypes.data,count,channels,scales.ctypes.data,ids_work.ctypes.data,shuffled_work.ctypes.data,encoded_work,len(encoded_work),4)
     if size<=0:raise RuntimeError(('native encoder',size))
     payload=scales.astype('<f4',copy=False).tobytes()+C.string_at(encoded_work,size);flag=0x60000000|size
    elif mode in ('plain','bf16','bf16_serial'):
     payload=raw.astype('<u2').tobytes()
    else:
     x=(raw.astype(np.uint32)<<16).view(np.float32).reshape(count,channels)
     scale=np.max(np.abs(x),axis=-1,keepdims=True)/127
     ids=np.clip(np.rint(x/np.where(scale>0,scale,1))+127,0,254).astype(np.uint8)
     if mode=='bslz4':
      shuffled=np.packbits(np.unpackbits(ids.reshape(-1,128,1),axis=2,bitorder='little').transpose(0,2,1),axis=2,bitorder='little')
      symbols=shuffled.tobytes()
     else: symbols=ids.tobytes()
     if mode=='int8': data=symbols;flag=0x80000000|len(data)
     else:
      encoded=C.create_string_buffer(len(symbols)*2)
      size=lz.LZ4_compress_HC(symbols,encoded,len(symbols),len(encoded),1)
      assert size>0
      data=encoded.raw[:size];flag=0x40000000|size
      if mode=='bslz4':flag|=0x20000000
     payload=scale.astype('<f4').tobytes()+data
    encode_ms+=(time.perf_counter()-encode_started)*1000
    nonce_counter+=1;nonce=nonce_counter.to_bytes(12,'big')
    aad=struct.pack('<8Q',index,kind,start,count,channels,len(payload),flag,1)
    seal_started=time.perf_counter()
    sealed=payload if mode=='plain' else AESGCM(key).encrypt(nonce,payload,aad)
    seal_ms+=(time.perf_counter()-seal_started)*1000
    allocate_started=time.perf_counter()
    wire_blob=sealed if mode=='plain' else sealed+aad
    host=torch.from_numpy(np.frombuffer(wire_blob,np.uint8).copy()).pin_memory()
    wire_device=torch.empty(host.numel(),dtype=torch.uint8,device='npu')
    cipher=wire_device[:len(payload)]
    plaintext=cipher if mode=='plain' else torch.empty_like(cipher)
    if mode=='plain':
     tag=torch.empty(16,dtype=torch.uint8,device='npu')
     aad_device=torch.empty(64,dtype=torch.uint8,device='npu')
    else:
     tag=wire_device[len(payload):len(payload)+16]
     aad_device=wire_device[len(payload)+16:len(payload)+80]
    record=dict(host=host,wire_device=wire_device,cipher=cipher,plain=plaintext,
      tag=tag,aad=aad_device,
      status=statuses[record_index:record_index+1],
      scratch=torch.empty(count*channels,dtype=torch.uint8,device='npu'),
      target=destination[start:start+count],count=count,channels=channels,flag=flag,
      iv=struct.unpack('>4I',nonce+struct.pack('>I',2)),stream=streams[len(output)%4])
    record_index+=1
    output.append(record);raw_bytes+=raw.nbytes
    wire+=len(sealed)+64+12+8
    allocate_ms+=(time.perf_counter()-allocate_started)*1000
   output.append(destination)
  # Strip destination sentinels from records and retain K/V outputs.
  destinations=[item for item in output if isinstance(item,torch.Tensor)]
  records=[item for item in output if isinstance(item,dict)]
  layers.append(dict(records=records,outputs=destinations,event=torch.npu.Event()))
 sync_started=time.perf_counter();torch.npu.synchronize()
 return layers,dict(encode_and_stage_ms=(time.perf_counter()-started)*1000,raw_bytes=raw_bytes,
                    wire_bytes=wire,ratio=raw_bytes/wire,
                    host_encode_ms=encode_ms,host_seal_ms=seal_ms,
                    record_allocation_ms=allocate_ms,prepare_sync_ms=(time.perf_counter()-sync_started)*1000)

class Loader:
 def __init__(self,layers,mode):
  self.layers=layers;self.mode=mode;self.next_layer=0;self.consumed=0
  self.condition=threading.Condition();self.error=None;self.ready=set();self.thread=None
  self.auth_stream=torch.npu.Stream();self.ready_stream=torch.npu.Stream();self.done=False
 def submit(self,index):
  layer=self.layers[index]
  auth_done=torch.npu.Event()
  with torch.npu.stream(self.auth_stream):
   for f in layer['records']:
    f['wire_device'].copy_(f['host'],non_blocking=True)
    if self.mode!='plain':
     assert verify(self.auth_stream.npu_stream,f['cipher'].data_ptr(),kd.data_ptr(),hd.data_ptr(),
         f['tag'].data_ptr(),f['status'].data_ptr(),f['cipher'].numel(),*f['iv'],f['aad'].data_ptr(),64)==0
   auth_done.record()
  for f in layer['records']:
   stream=f['stream']
   with torch.npu.stream(stream):
    stream.wait_event(auth_done)
    if self.mode!='plain':
     assert decrypt(stream.npu_stream,f['cipher'].data_ptr(),kd.data_ptr(),f['plain'].data_ptr(),f['plain'].numel(),*f['iv'])==0
    if f['flag']:
     assert restore(stream.npu_stream,f['plain'].data_ptr(),f['target'].data_ptr(),f['scratch'].data_ptr(),
         f['status'].data_ptr(),f['count'],f['channels'],f['flag'])==0
    else:
     f['target'].view(torch.uint8).reshape(-1).copy_(f['plain'],non_blocking=True)
  # A separate event stream waits for all K/V records of this layer.
  with torch.npu.stream(self.ready_stream):
   for f in layer['records']:self.ready_stream.wait_stream(f['stream'])
   layer['event'].record()
 @torch.inference_mode()
 def run(self):
  try:
   torch.npu.set_device(0)
   for i in range(len(self.layers)):
    with self.condition:
     self.condition.wait_for(lambda:i<self.consumed+args.lookahead or self.mode=='bf16_serial')
    self.submit(i)
    with self.condition:self.ready.add(i);self.condition.notify_all()
   self.done=True
  except BaseException as error:
   with self.condition:self.error=error;self.condition.notify_all()
 def start(self):
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def wait(self,index):
  with self.condition:
   self.condition.wait_for(lambda:index in self.ready or self.error is not None)
   if self.error:raise self.error
  self.layers[index]['event'].synchronize()
  # Enforce authentication before the model reads restored KV.
  if any(f['status'].item()!=1 for f in self.layers[index]['records']):
   raise RuntimeError('KV authentication or decoding failed')
  with self.condition:self.consumed=max(self.consumed,index+1);self.condition.notify_all()
  return tuple(x.permute(1,0,2).unsqueeze(0) for x in self.layers[index]['outputs'])

class RestoredCache(DynamicCache):
 def __init__(self,loader,config,shape):
  placeholders=[(torch.empty(shape,device='meta',dtype=torch.bfloat16),torch.empty(shape,device='meta',dtype=torch.bfloat16)) for _ in loader.layers]
  super().__init__(ddp_cache_data=placeholders,config=config)
  self.loader=loader;self.loaded=set()
 def update(self,k,v,layer_idx,cache_kwargs=None):
  if layer_idx not in self.loaded:
   oldk,oldv=self.loader.wait(layer_idx)
   layer=self.layers[layer_idx];layer.keys=oldk;layer.values=oldv
   layer.device=k.device;layer.dtype=k.dtype;self.loaded.add(layer_idx)
  return super().update(k,v,layer_idx,cache_kwargs)

print('loading model',flush=True)
tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
       attn_implementation='sdpa',local_files_only=True).eval().to('npu')
all_tokens=tokenizer.encode(Path(args.text_file).read_text(),add_special_tokens=False)
if len(all_tokens)<args.prefix+args.suffix:
 raise ValueError('text file is shorter than requested prefix and suffix; no repeated padding')
all_tokens=all_tokens[:args.prefix+args.suffix]
prefix=torch.tensor([all_tokens[:args.prefix]],device='npu')
suffix=torch.tensor([all_tokens[args.prefix:]],device='npu')
with torch.inference_mode():
 warm=model(prefix,use_cache=True,logits_to_keep=1)
 cpu_cache=[]
 for layer in warm.past_key_values.layers:
  cpu_cache.append((layer.keys[0].permute(1,0,2).cpu().contiguous(),layer.values[0].permute(1,0,2).cpu().contiguous()))
 shape=warm.past_key_values.layers[0].keys.shape
 del warm;torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
 results=[]
 for mode in args.modes.split(','):
  layers,encoding=prepare_cache(cpu_cache,mode)
  loader=Loader(layers,mode);cache=RestoredCache(loader,model.config,shape)
  torch.npu.synchronize();began=time.perf_counter();loader.start()
  if mode=='bf16_serial':
   for layer_index in range(len(layers)):loader.wait(layer_index)
  output=model(suffix,past_key_values=cache,use_cache=True,logits_to_keep=1)
  next_token=output.logits[:,-1].argmax(-1,keepdim=True)
  torch.npu.synchronize();first=time.perf_counter()
  generated=[int(next_token.item())]
  for _ in range(args.output_tokens-1):
   output=model(next_token,past_key_values=cache,use_cache=True,logits_to_keep=1)
   next_token=output.logits[:,-1].argmax(-1,keepdim=True)
   generated.append(int(next_token.item()))
  torch.npu.synchronize();end=time.perf_counter();loader.thread.join()
  row=dict(mode=mode,prefix=args.prefix,suffix=args.suffix,output_tokens=len(generated),
     ttft_ms=(first-began)*1000,tpot_ms=(end-first)*1000/max(1,len(generated)-1),
     e2e_ms=(end-began)*1000,metadata_in_timed_dma=True,tokens=generated,text=tokenizer.decode(generated),
     loaded_layers=len(cache.loaded),**encoding)
  results.append(row);print(json.dumps(row),flush=True)
  Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
  del output,cache,loader,layers;gc.collect();torch.npu.empty_cache()
