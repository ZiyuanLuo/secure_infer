"""Native layer transfer integrated with existing offload storage and crypto."""
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from .secure_connector import (SecureInferOffloadingConnectorWorker,
    SecureInferOffloadingConnectorScheduler, SecureInferSecureStoreMetadata)

def log(name, row):
    with (Path(os.environ["RUNTIME_RESULT_DIR"])/name).open("a") as out:
        out.write(json.dumps(row)+"\n")

@dataclass
class NativeMetadata(SecureInferSecureStoreMetadata):
    completed_native_loads: set[str] = field(default_factory=set)
    native_load_ready_counts: dict[str, int] = field(default_factory=dict)

    def aggregate(self, other):
        super().aggregate(other)
        self.completed_native_loads.update(other.completed_native_loads)
        for rid,count in other.native_load_ready_counts.items():
            self.native_load_ready_counts[rid]=self.native_load_ready_counts.get(rid,0)+count
        return self

class NativeScheduler(SecureInferOffloadingConnectorScheduler):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if tokens is not None:
            limit=max(0, ((request.num_tokens-1)//self.offloaded_block_size)
                      *self.offloaded_block_size-num_computed_tokens)
            tokens=min(tokens, limit)
        if not hasattr(self, '_native_local'):
            self._native_local={}
        self._native_local[request.request_id]=num_computed_tokens
        return tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
            # Same synchronous-hit connector contract as native layerwise backends.
            # Limit destination range to external blocks; suffix blocks are computed.
            self._requests[request.request_id]=request
            self._request_block_ids[request.request_id]=[]
            if not num_external_tokens:
                self._native_local.pop(request.request_id, None)
                return
            block_ids=blocks.get_block_ids()[0]
            local=self._native_local.pop(request.request_id)//self.gpu_block_size
            external=num_external_tokens//self.gpu_block_size
            assert num_external_tokens%self.gpu_block_size==0
            end=local+external
            assert len(block_ids)>=end
            num_blocks=(local*self.gpu_block_size+num_external_tokens)//self.offloaded_block_size
            # The native helper returns an iterator. Both prepare_load and later
            # completion bookkeeping need the same hashes, so materialize once.
            hashes=list(self._get_block_hashes(request,start_idx=local*self.gpu_block_size//self.offloaded_block_size,end_idx=num_blocks))
            src=self.manager.prepare_load(hashes)
            self._reqs_to_load[request.request_id]=(src,GPULoadStoreSpec(block_ids[local:end]))
            self._reqs_being_loaded[request.request_id].update(hashes)
            self._next_stored_block_idx[request.request_id]=num_blocks
            if self._blocks_being_loaded is not None:
                self._blocks_being_loaded.update(hashes)

    def update_connector_output(self, output):
            super().update_connector_output(output)
            meta=getattr(output, "kv_connector_worker_meta", None)
            if isinstance(meta,NativeMetadata):
                if not hasattr(self, '_native_load_worker_counts'):
                    self._native_load_worker_counts={}
                counts=meta.native_load_ready_counts or {rid:1 for rid in meta.completed_native_loads}
                for req,count in counts.items():
                    self._native_load_worker_counts[req]=self._native_load_worker_counts.get(req,0)+count
                    if self._native_load_worker_counts[req]<self._secure_ready_expected_workers:continue
                    del self._native_load_worker_counts[req]
                    hashes=self._reqs_being_loaded.pop(req,None)
                    if hashes:
                        if self._blocks_being_loaded is not None:
                            self._blocks_being_loaded.difference_update(hashes)
                        self.manager.complete_load(hashes)
                        log('native_load_retirements.jsonl',dict(request_id=req,blocks=len(hashes)))

class NativeTransfer:
    def __init__(self, handler):
        self.streams=handler._h2d_slice_streams
        self.retained=[]
        self.layer=0
        self.rt=ctypes.CDLL(None)
        self.begin=self.rt.secure_infer_h2d_layer_group_begin
        self.begin.argtypes=[ctypes.c_size_t]
        self.end=self.rt.secure_infer_h2d_layer_group_end
    def single_layer_kv_transfer(self, payload, destination):
        host,mapping=payload
        group=getattr(self,'causal_group_blocks',0)
        if group and len(mapping)>group:
            # Preserve each reverse chunk's operator grouping and launch count,
            # but submit all such groups for this layer before the next layer.
            layer=self.layer
            self.causal_group_blocks=0
            try:
                end=len(mapping)
                while end>0:
                    start=max(0,end-group)
                    self.layer=layer
                    self.single_layer_kv_transfer((host,mapping[start:end]),destination)
                    end=start
            finally:
                self.causal_group_blocks=group
                self.layer=layer+1
            return
        parent=torch.npu.current_stream()
        for stream in self.streams:
            if stream!=parent:
                stream.wait_stream(parent)
        assert self.begin(2*len(mapping))==0
        try:
            for src,dst in zip(host,destination):
                for j,stream in enumerate(self.streams):
                    part=mapping[len(mapping)*j//len(self.streams):len(mapping)*(j+1)//len(self.streams)]
                    self.retained.append(part)
                    with torch.npu.stream(stream):
                        torch.ops._C_ascend.swap_blocks(src,dst,part)
        finally:
            assert self.end()==0
        for stream in self.streams:
            if stream!=parent:
                parent.wait_stream(stream)
        self.layer+=1

class NativeWorker(SecureInferOffloadingConnectorWorker):
    def __init__(self,spec):
        super().__init__(spec)
        self.retained=[]
        self.wait_auth=ctypes.CDLL(None).secure_infer_recovery_wait_layer_authentication
        self.wait_auth.argtypes=[ctypes.c_uint32]
        self.wait_auth.restype=ctypes.c_int

    def retain_copy_inputs(self):
        event=torch.npu.Event()
        event.record(self.native.load_stream)
        self.retained.append((event,self.ops.retained))
        self.ops.retained=[]

    def get_finished(self,finished):
        result=super().get_finished(finished)
        if hasattr(self,'store_producer_events'):
            self.store_producer_events={job:event for job,event in self.store_producer_events.items()
                if job in self._jobs}
        self.retained=[(event,refs) for event,refs in self.retained if not event.query()]
        return result

    def register_kv_caches(self,caches):
        super().register_kv_caches(caches)
        self.names=list(caches)

        self.handler=self.spec._handler
        # The store submitter runs on another host thread. Capture the actual
        # producer stream before queueing; a background thread's default stream
        # does not order D2H after the foreground model writes.
        self.store_producer_events={}
        queue_store=self.handler._queue_d2h_submission
        submit=self.handler._submit_transfer
        def queue_after_producer(job_id,spec):
            event=torch.npu.Event()
            event.record(torch.npu.current_stream())
            self.store_producer_events[job_id]=event
            return queue_store(job_id,spec)
        def submit_after_producer(job_id,spec,*args,**kwargs):
            event=self.store_producer_events.get(job_id)
            if event is not None:
                self.handler.d2h_stream.wait_event(event)
            return submit(job_id,spec,*args,**kwargs)
        self.handler._queue_d2h_submission=queue_after_producer
        self.handler._submit_transfer=submit_after_producer
        self.ops=NativeTransfer(self.handler)
        self.ops.causal_group_blocks=int(os.environ.get("RUNTIME_LOAD_CHUNK","2048"))//128
        self.native=SimpleNamespace(num_layers=len(caches),load_stream=self.handler.h2d_stream,
            kvcaches=self.handler.npu_tensors)
        from secure_infer.reuse.timing_hooks import install
        install(self)
