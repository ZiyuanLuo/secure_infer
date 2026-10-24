"""Check reciprocal-assisted quantization against float32 division.

Includes all finite BF16 bit patterns and values near INT8 rounding boundaries.
"""
import ctypes as C
from pathlib import Path
import numpy as np

lib = C.CDLL(str(Path('libhost_encoder.so').resolve()))
encode = lib.paci_encode_symbols
encode.argtypes = [C.c_void_p,C.c_uint32,C.c_uint32]+[C.c_void_p]*4+[C.c_int,C.c_int]
words = np.arange(65536, dtype=np.uint16)
words = words[(words & 0x7f80) != 0x7f80]
words = np.pad(words, (0, (-len(words))%128)).reshape(-1,128)
rng = np.random.default_rng(917)
fixtures = [words, np.zeros((16,128),np.uint16)]
for exponent in (-120,-60,0,60,120):
    x = rng.uniform(-127,127,(128,128)).astype(np.float32)
    # Mix arbitrary values with values close to half-integer boundaries.
    x[::2] = np.floor(x[::2])+np.float32(.5)
    x[:,0] = 127
    x = np.ldexp(x,exponent-7)
    fixtures.append((x.view(np.uint32)>>16).astype(np.uint16))
checked = 0
for raw in fixtures:
    tokens,channels = raw.shape
    scales = np.empty(tokens,np.float32)
    ids = np.empty(raw.size,np.uint8);shuffled = np.empty_like(ids)
    output = C.create_string_buffer(raw.size*2)
    n = encode(raw.ctypes.data,tokens,channels,scales.ctypes.data,ids.ctypes.data,
               shuffled.ctypes.data,output,len(output),4)
    assert n>0
    values = (raw.astype(np.uint32)<<16).view(np.float32)
    reference_scale = np.max(np.abs(values),axis=1)/np.float32(127)
    reference = np.clip(np.rint(values/np.where(reference_scale[:,None]>0,
                         reference_scale[:,None],np.float32(1)))+127,0,254).astype(np.uint8)
    assert np.array_equal(scales,reference_scale)
    assert np.array_equal(ids.reshape(raw.shape),reference)
    checked += raw.size
print('Exact division-reference symbols:',checked,flush=True)
