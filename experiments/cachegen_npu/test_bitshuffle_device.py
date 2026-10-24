"""Compare AICPU bitshuffle kernels on actual BF16 samples and TP shard widths."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--libraries', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    enc = C.CDLL(str(args.encoder.resolve())).paci_encode_symbols
    enc.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]+[C.c_void_p]*4+[C.c_int,C.c_int]
    kernels = []
    for path in args.libraries:
        lib = C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL)
        fn = lib.entropy_restore_do
        fn.argtypes = [C.c_void_p]*5+[C.c_uint32]*3
        fn.restype = C.c_int
        kernels.append((str(path), lib, fn))
    torch.npu.set_device(0)
    rows = []
    for kind in ('K', 'V'):
        source = np.fromfile(args.data/(kind+'.bf16'), '<u2').reshape(-1,8,128)
        for channels in (128,256,1024):
            for tokens in (128,512):
                fixtures = []
                # Four independent records exercise the same multi-stream restore
                # organization used in the full-model benchmark.
                for lane in range(4):
                    raw = np.ascontiguousarray(source[lane*tokens:(lane+1)*tokens,:channels//128])
                    scales = np.empty(tokens, np.float32)
                    ids = np.empty(raw.size, np.uint8);shuffled = np.empty_like(ids)
                    encoded = C.create_string_buffer(raw.size*2)
                    n = enc(raw.ctypes.data,tokens,channels,scales.ctypes.data,ids.ctypes.data,
                            shuffled.ctypes.data,encoded,len(encoded),4)
                    assert n>0
                    blob = scales.astype('<f4',copy=False).tobytes()+encoded.raw[:n]
                    inp = torch.from_numpy(np.frombuffer(blob,np.uint8).copy()).npu()
                    out = torch.empty((tokens,channels),dtype=torch.bfloat16,device='npu')
                    scratch = torch.empty(raw.size,dtype=torch.uint8,device='npu')
                    status = torch.ones(1,dtype=torch.int32,device='npu')
                    expected = torch.from_numpy((ids.reshape(tokens,channels).astype(np.float32)-127)*scales[:,None]).to(torch.bfloat16)
                    fixtures.append((inp,out,scratch,status,torch.npu.Stream(),n,expected))
                torch.npu.synchronize()
                times = {name:[] for name,_,_ in kernels}
                for repeat in range(4):
                    ordered = kernels if repeat%2==0 else list(reversed(kernels))
                    for name,_,fn in ordered:
                        begin=time.perf_counter()
                        for inp,out,scratch,status,stream,n,expected in fixtures:
                            assert fn(stream.npu_stream,inp.data_ptr(),out.data_ptr(),scratch.data_ptr(),
                                      status.data_ptr(),tokens,channels,0x60000000|n)==0
                        for fixture in fixtures:fixture[4].synchronize()
                        elapsed=(time.perf_counter()-begin)*1000
                        if repeat:times[name].append(elapsed)
                        for inp,out,scratch,status,stream,n,expected in fixtures:
                            assert status.item()==1
                            assert torch.equal(out.cpu().view(torch.int16),expected.view(torch.int16)),(name,kind,tokens,channels)
                row=dict(kind=kind,tokens_per_record=tokens,channels_per_rank=channels,streams=4,
                         elapsed_ms=times,median_ms={k:float(np.median(v)) for k,v in times.items()},
                         exact_quantized_bf16=True)
                rows.append(row);print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
