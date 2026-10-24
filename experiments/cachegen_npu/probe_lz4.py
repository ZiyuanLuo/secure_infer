import ctypes as C
import ctypes.util
import numpy as np
lib=C.CDLL(ctypes.util.find_library('lz4'))
import time
lib.LZ4_compress_HC.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
for name in ('K','V'):
 a=np.fromfile('data/'+name+'.bf16',dtype='<u2')
 x=(a.astype(np.uint32)<<16).view(np.float32).reshape(-1,1024)
 for tokens in (128,512,2048):
  v=x[:tokens];s=np.max(np.abs(v),axis=-1,keepdims=True)/127
  q=np.clip(np.rint(v/np.where(s>0,s,1))+127,0,254).astype(np.uint8)
  b=q.tobytes();out=C.create_string_buffer(len(b)*2)
  for level in (1,4,9):
   start=time.perf_counter()
   n=lib.LZ4_compress_HC(b,out,len(b),len(out),level)
   ms=(time.perf_counter()-start)*1000
   assert n>0
   print(name,tokens,level,v.nbytes/2/(n+s.nbytes+80),n,ms,flush=True)
