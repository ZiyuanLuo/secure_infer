"""Check native encoder against NumPy and independently decode its LZ4 stream."""
import ctypes as C
import ctypes.util
import json
import time
import numpy as np
lib=C.CDLL('./libhost_encoder.so')
fn=lib.paci_encode_symbols
fn.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32]+[C.c_void_p]*4+[C.c_int,C.c_int]
lz=C.CDLL(ctypes.util.find_library('lz4'))
lz.LZ4_decompress_safe.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int]
rng=np.random.default_rng(123)
for tokens,channels in ((1,128),(128,128),(512,256),(2048,1024)):
 x=rng.standard_normal((tokens,channels)).astype(np.float32)
 x[0]=0
 bits=(x.view(np.uint32)>>16).astype(np.uint16)
 x=(bits.astype(np.uint32)<<16).view(np.float32)
 scales=np.empty(tokens,np.float32);ids=np.empty(x.size,np.uint8);shuffled=np.empty_like(ids)
 out=C.create_string_buffer(ids.size*2)
 begin=time.perf_counter()
 length=fn(bits.ctypes.data,tokens,channels,scales.ctypes.data,ids.ctypes.data,shuffled.ctypes.data,out,len(out),4)
 elapsed=(time.perf_counter()-begin)*1000
 assert length>0
 reference_scale=np.max(np.abs(x),axis=-1)/127
 reference_ids=np.clip(np.rint(x/np.where(reference_scale[:,None]>0,reference_scale[:,None],1))+127,0,254).astype(np.uint8)
 assert np.array_equal(scales,reference_scale)
 assert np.array_equal(ids.reshape(tokens,channels),reference_ids)
 reference_shuffle=np.packbits(np.unpackbits(reference_ids.reshape(-1,128,1),axis=2,bitorder='little').transpose(0,2,1),axis=2,bitorder='little')
 assert shuffled.tobytes()==reference_shuffle.tobytes()
 decoded=C.create_string_buffer(ids.size)
 assert lz.LZ4_decompress_safe(out,decoded,length,len(decoded))==ids.size
 assert decoded.raw==shuffled.tobytes()
 print(json.dumps(dict(tokens=tokens,channels=channels,encoder_ms=elapsed,exact_reference=True)),flush=True)
