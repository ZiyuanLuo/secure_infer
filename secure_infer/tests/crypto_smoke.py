"""Check the project's device AES-GCM kernels against host AES-GCM."""
import argparse
import ctypes as ct
import json
import os
import struct
from pathlib import Path

import torch
import torch_npu
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    library = ct.CDLL(str(Path(args.library).resolve()))
    ptr, size, word = ct.c_void_p, ct.c_uint64, ct.c_uint32
    encrypt = library.secure_infer_aicpu_gcm_encrypt_tag_do
    encrypt.argtypes = [ptr]*6 + [size] + [word]*4 + [ptr, size]
    decrypt = library.secure_infer_aicpu_gcm_decrypt_verify_aad_do
    decrypt.argtypes = [ptr]*7 + [size] + [word]*4 + [ptr, size]
    keystream = library.secure_infer_aicpu_keystream_do
    keystream.argtypes = [ptr]*3 + [size] + [word]*4
    verify_xor = library.secure_infer_aicpu_gcm_verify_xor_aad_do
    verify_xor.argtypes = [ptr]*8 + [size] + [word]*4 + [ptr, size]

    def device_bytes(value):
        return torch.tensor(list(value), dtype=torch.uint8, device='npu')

    def host_bytes(value):
        return value.cpu().numpy().tobytes()

    for device in map(int, args.devices.split(',')):
        torch.npu.set_device(device)
        for count in (16, 31, 32768, 262144):
            key, nonce, aad = os.urandom(16), os.urandom(12), os.urandom(64)
            plain = os.urandom(count)
            reference = AESGCM(key).encrypt(nonce, plain, aad)
            ecb = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
            h = ecb.update(bytes(16)) + ecb.finalize()
            dk, dh, da = map(device_bytes, (key, h, aad))
            source = device_bytes(plain)
            ciphertext = torch.empty_like(source)
            tag = torch.empty(16, dtype=torch.uint8, device='npu')
            stream = torch.npu.current_stream()
            iv = (*struct.unpack('>III', nonce), 2)
            torch.npu.synchronize()
            assert encrypt(stream.npu_stream, source.data_ptr(), dk.data_ptr(), dh.data_ptr(), ciphertext.data_ptr(), tag.data_ptr(), count, *iv, da.data_ptr(), len(aad)) == 0
            stream.synchronize()
            protected = host_bytes(ciphertext) + host_bytes(tag)
            assert protected == reference, ('device encryption mismatch', device, count)
            assert AESGCM(key).decrypt(nonce, protected, aad) == plain
            dc, dt = device_bytes(reference[:-16]), device_bytes(reference[-16:])
            restored = torch.empty_like(source)
            status = torch.zeros(1, dtype=torch.int32, device='npu')
            assert decrypt(stream.npu_stream, dc.data_ptr(), dk.data_ptr(), dh.data_ptr(), dt.data_ptr(), restored.data_ptr(), status.data_ptr(), count, *iv, da.data_ptr(), len(aad)) == 0
            stream.synchronize()
            assert status.item() == 1 and host_bytes(restored) == plain, ('device decryption mismatch', device, count)
            ks = torch.empty_like(source)
            assert keystream(stream.npu_stream, dk.data_ptr(), ks.data_ptr(), count, *iv) == 0
            assert verify_xor(stream.npu_stream, dc.data_ptr(), dk.data_ptr(), dh.data_ptr(), dt.data_ptr(), ks.data_ptr(), restored.data_ptr(), status.data_ptr(), count, *iv, da.data_ptr(), len(aad)) == 0
            stream.synchronize()
            assert status.item() == 1 and host_bytes(restored) == plain, ('precomputed keystream mismatch', device, count)
            print(json.dumps(dict(device=device, bytes=count, device_encrypt_host_decrypt='PASS', host_encrypt_device_decrypt='PASS', precomputed_keystream='PASS')), flush=True)


if __name__ == '__main__':
    main()
