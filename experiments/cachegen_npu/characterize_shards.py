"""Measure the real encoder on per-rank KV, without gathering tensor-parallel heads.

Input is token-major BF16 K/V. The output is a codec microbenchmark, not
distributed inference or an answer-quality evaluation.
"""
import argparse
import ctypes as C
import ctypes.util
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--tp', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--chunks', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib = C.CDLL(str(args.library.resolve()))
    encode = lib.paci_encode_symbols
    encode.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32] + [C.c_void_p]*4 + [C.c_int, C.c_int]
    lz = C.CDLL(ctypes.util.find_library('lz4'))
    lz.LZ4_decompress_safe.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int]
    rows = []
    for kind in ('K', 'V'):
        source = np.fromfile(args.data / (kind+'.bf16'), dtype='<u2')
        source = source.reshape(-1, args.heads, args.head_dim)
        for tp in args.tp:
            if tp < 1 or args.heads % tp:
                raise ValueError('This benchmark requires equally sharded KV heads')
            channels = args.heads // tp * args.head_dim
            for chunk in args.chunks:
                if chunk < 1:
                    raise ValueError('chunk must be positive')
                raw_bytes = wire_bytes = 0
                square_error = square_value = 0.
                encode_ms = []
                rank_ratios = []
                for rank in range(tp):
                    shard = np.ascontiguousarray(source[:, rank*args.heads//tp:(rank+1)*args.heads//tp])
                    rank_wire = 0
                    for start in range(0, len(source), chunk):
                        raw = shard[start:start+chunk].reshape(-1, channels)
                        tokens = len(raw)
                        scale = np.empty(tokens, np.float32)
                        ids = np.empty(raw.size, np.uint8)
                        shuffled = np.empty_like(ids)
                        output = C.create_string_buffer(raw.size*2)
                        begin = time.perf_counter()
                        n = encode(raw.ctypes.data, tokens, channels, scale.ctypes.data,
                                   ids.ctypes.data, shuffled.ctypes.data, output, len(output), 4)
                        encode_ms.append((time.perf_counter()-begin)*1000)
                        if n <= 0:
                            raise RuntimeError(('encode', n, tp, rank, start))
                        decoded = C.create_string_buffer(raw.size)
                        assert lz.LZ4_decompress_safe(output, decoded, n, len(decoded)) == raw.size
                        assert decoded.raw == shuffled.tobytes()
                        # Independent inverse of the 128-symbol bit shuffle.
                        packed = np.frombuffer(decoded, np.uint8).reshape(-1, 8, 16)
                        bits = np.unpackbits(packed, axis=2, bitorder='little').transpose(0, 2, 1)
                        restored_ids = np.packbits(bits, axis=2, bitorder='little').reshape(-1)
                        assert np.array_equal(restored_ids, ids)
                        values = (raw.astype(np.uint32) << 16).view(np.float32)
                        restored = (ids.reshape(tokens, channels).astype(np.float32)-127)*scale[:, None]
                        square_error += float(np.sum((values-restored)**2, dtype=np.float64))
                        square_value += float(np.sum(values**2, dtype=np.float64))
                        # Same record budget as model_bench: scale, tag, AAD, nonce, index.
                        record_bytes = n + scale.nbytes + 16 + 64 + 12 + 8
                        rank_wire += record_bytes
                    raw_bytes += shard.nbytes
                    wire_bytes += rank_wire
                    rank_ratios.append(shard.nbytes / rank_wire)
                row = dict(kind=kind, tp=tp, channels_per_rank=channels, chunk=chunk,
                           tokens=len(source), raw_bytes=raw_bytes, wire_bytes=wire_bytes,
                           ratio=raw_bytes/wire_bytes, rank_ratios=rank_ratios,
                           nmse=square_error/max(square_value, 1e-30),
                           median_encode_record_ms=float(np.median(encode_ms)),
                           total_encode_ms=sum(encode_ms), exact_symbol_roundtrip=True)
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
