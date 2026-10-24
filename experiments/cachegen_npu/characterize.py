"""Measure codec sizes on real BF16 KV; no inference speed or quality claim."""
import argparse
import json
import math
import struct
import time
import zlib
from pathlib import Path
import numpy as np


def entropy(a, alphabet):
    counts = np.bincount(a.reshape(-1), minlength=alphabet)
    p = counts[counts > 0] / a.size
    return float(-(p * np.log2(p)).sum())


def pack(ids, bits):
    flat = ids.reshape(-1).astype(np.uint32)
    # Independent 8-symbol groups support fixed-offset vector restoration.
    flat = np.pad(flat, (0, (-flat.size) % 8)).reshape(-1, 8)
    words = np.zeros(len(flat), dtype=np.uint64)
    for i in range(8):
        words |= flat[:, i].astype(np.uint64) << (i * bits)
    return np.stack([(words >> (8 * b)).astype(np.uint8) for b in range(bits)], axis=1).tobytes()


def unpack(data, bits, count):
    data = np.frombuffer(data, np.uint8).reshape(-1, bits)
    words = np.zeros(len(data), dtype=np.uint64)
    for b in range(bits):
        words |= data[:, b].astype(np.uint64) << (8 * b)
    return np.stack([(words >> (i * bits)) & ((1 << bits)-1) for i in range(8)], axis=1).reshape(-1)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in ('K', 'V'):
        raw = np.fromfile(args.data / (name + '.bf16'), dtype='<u2')
        x = (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, 1024)
        for tokens in (128, 512, 2048):
            original = raw[:tokens * 1024].tobytes()
            for mode in ('raw', 'byte_shuffle', 'token_xor_shuffle'):
                words = raw[:tokens * 1024].reshape(tokens, 1024)
                if mode == 'token_xor_shuffle':
                    words = np.concatenate((words[:1], words[1:] ^ words[:-1]), axis=0)
                payload = words.astype('<u2').tobytes()
                if mode != 'raw':
                    payload = np.frombuffer(payload, np.uint8).reshape(-1, 2).T.copy().tobytes()
                began = time.perf_counter()
                encoded = zlib.compress(payload, 1)
                enc_ms = (time.perf_counter()-began)*1000
                began = time.perf_counter()
                restored = zlib.decompress(encoded)
                dec_ms = (time.perf_counter()-began)*1000
                assert restored == payload
                result = np.frombuffer(restored, np.uint8)
                if mode != 'raw':
                    result = result.reshape(2, -1).T.copy().reshape(-1)
                result = np.frombuffer(result.tobytes(), '<u2').reshape(tokens, 1024)
                if mode == 'token_xor_shuffle':
                    result = np.bitwise_xor.accumulate(result, axis=0)
                assert result.tobytes() == original
                wire = math.ceil(len(encoded)/16)*16 + 80
                rows.append(dict(kind=name, tokens=tokens, codec='lossless_zlib_'+mode,
                                 raw_bytes=len(original), wire_bytes=wire, ratio=len(original)/wire,
                                 host_encode_ms=enc_ms, host_decode_ms=dec_ms,
                                 bitwise_correct=True))
            for bits in (4, 5, 6, 8):
                # CacheGen's published per-token max-absolute quantization rule.
                # Padding is one unused code; zero groups are handled explicitly.
                values = x[:tokens]
                maximum = np.max(np.abs(values), axis=-1, keepdims=True)
                limit = 2**(bits-1)-1
                scale = maximum / limit
                safe = np.where(scale > 0, scale, 1)
                ids = np.clip(np.rint(values / safe) + limit, 0, 2*limit).astype(np.uint8)
                payload = pack(ids, bits)
                decoded = unpack(payload, bits, ids.size).reshape(ids.shape).astype(np.int16)
                assert np.array_equal(decoded, ids)
                recovered = (decoded-limit).astype(np.float32) * scale
                nmse = float(np.mean((recovered-values)**2)/max(float(np.mean(values**2)), 1e-30))
                meta_bytes = tokens*4 + 80
                wire = math.ceil((len(payload)+tokens*4)/16)*16 + 80
                # Real DEFLATE byte count is a comparison, not CacheGen arithmetic coding.
                compressed = zlib.compress(payload, 1)
                entropy_wire = math.ceil((len(compressed)+tokens*4)/16)*16+80
                channel_entropy = np.mean([entropy(ids[:,c], 2**bits) for c in range(1024)])
                rows.append(dict(kind=name,tokens=tokens,codec='cachegen_rule_fixed_pack',bits=bits,
                                 raw_bytes=len(original),wire_bytes=wire,ratio=len(original)/wire,
                                 deflate_wire_bytes=entropy_wire,deflate_ratio=len(original)/entropy_wire,
                                 symbol_entropy_bits=entropy(ids,2**bits),
                                 mean_channel_entropy_bits=float(channel_entropy),
                                 nmse=nmse,max_abs_error=float(np.max(np.abs(recovered-values))),
                                 quantized_symbol_roundtrip=True,bitwise_correct=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    for r in rows:
        if r['tokens']==2048:
            print(r['kind'], r['codec'], r.get('bits',''),round(r['ratio'],3),
                  'deflate=',round(r.get('deflate_ratio',0),3),'nmse=',r.get('nmse',0),flush=True)


if __name__ == '__main__':
    main()
