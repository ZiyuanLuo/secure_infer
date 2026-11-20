"""Authenticated H2D submission and KV recovery on separate NPU streams."""
import os
import queue
import threading
import time
import torch
from .transport_codec import PackedTransfer, layer_bits, settings
from .optimized_decode import decode_into


class PipelinedTransfer(PackedTransfer):
    def __init__(self, original, store):
        super().__init__(original, store)
        self.use_stages = bool(int(os.environ.get('KV_STAGE_PIPELINE', '1')))
        self.recovery_stream = None
        self.index_buffer = None
        self.worker = None

    def prepare(self, caches):
        if not self.use_stages:
            return
        self.device = caches[0][0].device
        self.recovery_stream = torch.npu.Stream()
        self.capacity = min(self.store.handler.cpu_tensors[0][0].shape[0], caches[0][0].shape[0])
        self.index_buffer = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        mode, _ = settings()
        if mode == 'none':
            return
        # Fixed-capacity storage avoids allocation for every new q.
        for layer, tensors in enumerate(caches):
            for kind, destination in enumerate(tensors):
                bits = layer_bits(mode, layer, len(caches))
                self._reserve(layer, kind, bits, destination)

    def _reserve(self, layer, kind, bits, destination):
        elements = destination[0].numel()
        qb = elements * bits // 8
        sb = elements // 64 * 4
        width = (qb + 2 * sb + 16 + 15) // 16 * 16
        key = (layer, kind, bits)
        if key not in self.buffers:
            self.buffers[key] = torch.empty((self.capacity, width), dtype=torch.uint8, device=destination.device)
        temporary_key = (kind, tuple(destination.shape[1:]))
        if temporary_key not in self.temporary_buffers:
            self.temporary_buffers[temporary_key] = torch.empty(
                self.capacity * elements, dtype=torch.float32, device=destination.device)

    def start_request(self, request_id, mode, layers, on_ready, on_error):
        self.request_id = request_id
        self.mode = mode
        self.layers = layers
        self.records = []
        self.stage_records = []
        self.on_ready = on_ready
        self.on_error = on_error
        self.recovery_error = None
        self.cancelled = threading.Event()
        self.request_mapping = None
        self.anchor = None
        self.pending = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._recover_loop, name='kv-recovery', daemon=True)
        self.worker.start()

    def _put(self, value):
        while True:
            if self.recovery_error is not None:
                raise RuntimeError(self.recovery_error)
            try:
                self.pending.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def transfer(self, payload, destinations):
        if not self.use_stages or self.mode == 'none':
            return super().transfer(payload, destinations)
        old = self.original
        host, mapping = payload
        layer, b = old.layer, len(mapping)
        assert b <= self.capacity
        bits = layer_bits(self.mode, layer, self.layers)
        if self.request_mapping is None:
            local = torch.stack((mapping[:, 0], torch.arange(b)), dim=1).long()
            index_host = mapping[:, 1].contiguous().pin_memory()
            indices = self.index_buffer[:b]
            indices.copy_(index_host, non_blocking=True)
            self.request_mapping = (local, indices, mapping[:, 0].tolist())
            old.retained.extend([local, index_host, indices])
        local, indices, source_ids = self.request_mapping
        packets = []
        for kind, destination in enumerate(destinations):
            self._reserve(layer, kind, bits, destination)
            cpu, qb, sb = self.store.get(layer, kind, bits, source_ids)
            gpu = self.buffers[layer, kind, bits][:b]
            assert gpu.shape[1] == cpu.shape[1]
            packets.append((kind, cpu, gpu, qb, sb, destination))
        start = self.load_start
        if self.anchor is None:
            self.anchor = start
        assert old.begin(2 * b) == 0
        try:
            for kind, cpu, gpu, qb, sb, destination in packets:
                torch.ops._C_ascend.swap_blocks(cpu, gpu, local)
        finally:
            assert old.end() == 0
        copied = torch.npu.Event(enable_timing=True)
        copied.record(torch.npu.current_stream())
        self._put(dict(layer=layer, bits=bits, blocks=b, packets=packets,
                       indices=indices, start=start, copied=copied,
                       original_bytes=sum(b * x[0].numel() * x.element_size() for x in host)))
        old.layer += 1

    def _recover_one(self, task):
        # Verify before any decompression is submitted. H2D submission can
        # continue on its own thread while this verifier waits.
        layer = task['layer']
        auth_begin = time.monotonic_ns()
        assert self.wait_auth(layer) == 0, 'Packed KV authentication failed'
        auth_ms = (time.monotonic_ns() - auth_begin) / 1e6
        host_begin = time.monotonic_ns()
        with torch.npu.stream(self.recovery_stream):
            self.recovery_stream.wait_event(task['copied'])
            before = torch.npu.Event(enable_timing=True)
            before.record(self.recovery_stream)
            for kind, cpu, gpu, qb, sb, destination in task['packets']:
                temporary = self.temporary_buffers[kind, tuple(destination.shape[1:])]
                decode_into(gpu, destination, task['indices'], task['bits'], kind, qb, sb,
                            temporary=temporary[:task['blocks'] * destination[0].numel()])
            after = torch.npu.Event(enable_timing=True)
            after.record(self.recovery_stream)
        self.records.append(dict(layer=layer, bits=task['bits'], blocks=task['blocks'],
            original_bytes=task['original_bytes'],
            wire_bytes=sum(task['blocks'] * cpu.shape[1] for _, cpu, *_ in task['packets']),
            encode_cpu_ms=0., reload_reencoded=False, optimized_decode=True,
            decode_host_submit_ms=(time.monotonic_ns() - host_begin) / 1e6,
            authentication_host_wait_ms=auth_ms))
        self.stage_records.append(dict(layer=layer, transfer_start=task['start'],
                                      transfer_end=task['copied'], decode_start=before, decode_end=after))
        self.on_ready(layer, after, task['start'])

    def _recover_loop(self):
        try:
            torch.npu.set_device(self.device)
            while True:
                task = self.pending.get()
                if task is None or self.cancelled.is_set():
                    return
                self._recover_one(task)
        except BaseException as error:
            self.recovery_error = repr(error)
            self.on_error(self.recovery_error)

    def finish_request(self):
        if self.worker is None:
            return
        self._put(None)
        self.worker.join(timeout=120)
        assert not self.worker.is_alive(), 'KV recovery submitter timeout'
        if self.recovery_error is not None:
            raise RuntimeError(self.recovery_error)

    def abort_request(self):
        if self.worker is None:
            return
        self.cancelled.set()
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=120)
        if self.worker.is_alive():
            raise RuntimeError('KV recovery worker could not be drained')
        self.recovery_stream.synchronize()

    def flush(self):
        if self.use_stages and self.mode != 'none':
            from .native_layerwise import log
            stages = []
            for record in self.stage_records:
                stages.append(dict(layer=record['layer'], **{
                    key + '_ms': self.anchor.elapsed_time(value)
                    for key, value in record.items() if key != 'layer'}))
            log('recovery_stages.jsonl', dict(request_id=self.request_id, stages=stages))
        return super().flush()
