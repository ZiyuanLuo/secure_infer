import ctypes as C
import ctypes.util
import numpy as np
lib=C.CDLL(ctypes.util.find_library('lz4'))
lib.LZ4_compress_HC.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
for name in ('K','V'):
 raw=np.fromfile('data/'+name+'.bf16',dtype='<u2')
 x=(raw.astype(np.uint32)<<16).view(np.float32).reshape(-1,1024)
 s=np.max(np.abs(x),axis=-1,keepdims=True)/127
 ids=np.clip(np.rint(x/np.where(s>0,s,1))+127,0,254).astype(np.uint8)
 for block in (32,128,256,1024):
  shuffled=np.packbits(np.unpackbits(ids.reshape(-1,block,1),axis=2,bitorder='little').transpose(0,2,1),axis=2,bitorder='little').tobytes()
  out=C.create_string_buffer(len(shuffled)*2)
  n=lib.LZ4_compress_HC(shuffled,out,len(shuffled),len(out),1)
  print(name,block,x.size*2/(n+s.nbytes+100),flush=True)
