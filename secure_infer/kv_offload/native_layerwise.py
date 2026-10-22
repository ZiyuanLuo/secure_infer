"""Native layer generator integrated with existing offload storage and crypto."""
import ast
import ctypes
from dataclasses import dataclass, field
import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
except ImportError:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata as KVConnectorWorkerMetadata
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm_ascend.kv_offload.cpu_npu import expand_block_ids
from secure_infer.kv_offload.npu_predict import NPUOffloadingSpec, _retire_h2d_prediction_scope
from .secure_connector import (SecureInferOffloadingConnector, SecureInferOffloadingConnectorWorker,
    SecureInferOffloadingConnectorScheduler, SecureInferOffloadingConnectorMetadata,
    SecureInferSecureStoreMetadata)

from .layerwise_generator import LayerwiseGenerator

MODE="native_secure"

def log(name, row):
    with (Path(os.environ["RUNTIME_RESULT_DIR"])/name).open("a") as out:
        out.write(json.dumps(row)+"\n")

@dataclass
class NativeMetadata(SecureInferSecureStoreMetadata):
    completed_native_loads: set[str] = field(default_factory=set)

    def aggregate(self, other):
        super().aggregate(other)
        self.completed_native_loads.update(other.completed_native_loads)
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
                for req in meta.completed_native_loads:
                    hashes=self._reqs_being_loaded.pop(req,None)
                    if hashes:
                        if self._blocks_being_loaded is not None:
                            self._blocks_being_loaded.difference_update(hashes)
                        self.manager.complete_load(hashes)
                        log('native_load_retirements.jsonl',dict(request_id=req,blocks=len(hashes)))

class NativeTransfer:
    def __init__(self, handler):
        self.handler=handler
        self.streams=handler._h2d_slice_streams
        self.retained=[]
        self.layer=0
        self.rt=ctypes.CDLL(None)
        self.begin=self.rt.secure_infer_h2d_layer_group_begin
        self.begin.argtypes=[ctypes.c_size_t]
        self.end=self.rt.secure_infer_h2d_layer_group_end
        # The unchanged Fig.1 binary has no diagnostic batching symbol.
        # Batch execution remains disabled by the original protocol.
        self.stats=None
        self.cpu_stats=self.rt.secure_infer_get_transfer_prefetch_stats
        self.cpu_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]*3
        self.cpu_stats.restype=None
        self.key_stats=self.rt.secure_infer_get_gcm_keystream_stats
        self.key_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]*7
        self.key_stats.restype=None

    def path_counts(self):
        # Read-only cumulative counters; no event wait, prediction or status consumption.
        cpu=[ctypes.c_uint64() for _ in range(3)]
        key=[ctypes.c_uint64() for _ in range(7)]
        self.cpu_stats(*(ctypes.byref(x) for x in cpu))
        self.key_stats(*(ctypes.byref(x) for x in key))
        names=['cpu_ready','cpu_busy','cpu_missing','key_ready','key_busy','key_missing',
            'key_aicpu_submitted','key_aicore_submitted','key_dropped','key_switches']
        return dict(zip(names,[x.value for x in cpu+key]))

    def batch_counts(self):
        launches=ctypes.c_uint64()
        records=ctypes.c_uint64()
        if self.stats is not None:
            self.stats(ctypes.byref(launches),ctypes.byref(records))
        return launches.value,records.value

    def single_layer_kv_transfer(self, payload, destination, slots, to_cpu, kv_format, token_major, two_major):
        assert not to_cpu
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
                    self.single_layer_kv_transfer((host,mapping[start:end]),destination,
                        slots,to_cpu,kv_format,token_major,two_major)
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
        self.active=None
        self.completed=set()
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

    def check_loaded_bytes(self, active):
        if os.environ.get("RUNTIME_CHECK_LOADED_BYTES")!="1":
            return
        host,mapping=active["payload"][0].tensor
        src=mapping[:,0].long()
        dst=mapping[:,1].long()
        self.native.load_stream.synchronize()
        for layer,(host,device) in enumerate(zip(self.handler.cpu_tensors,self.handler.npu_tensors)):
            for kind,(h,d) in enumerate(zip(host,device)):
                assert torch.equal(h[src],d[dst.to(device=d.device)].cpu()),(layer,kind)
        log("byte_checks.jsonl",dict(request_id=active["req"],layers=len(self.names),records=2*len(src)*len(self.names),exact=True))

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
        self.native=LayerwiseGenerator(self.ops)
        self.native.use_gpu=False
        self.native.num_layers=len(caches)
        from secure_infer.reuse.timing_hooks import install
        install(self)
        self.native.load_stream=self.handler.h2d_stream
        self.native.kvcaches=self.handler.npu_tensors
        self.native.kv_format=SimpleNamespace(value=2)
        self.native.vllm_two_major=False
        self.native.initialize_kvcaches_ptr=lambda **kwargs:None

    def start_kv_transfers(self,metadata):
        assert self.active is None
        super().start_kv_transfers(SecureInferOffloadingConnectorMetadata(reqs_to_load={},reqs_to_store=metadata.reqs_to_store))
        if not metadata.reqs_to_load:
            return
        assert len(metadata.reqs_to_load)==1,'This sequential benchmark requires one reload per model step'
        req,spec=next(iter(metadata.reqs_to_load.items()))
        job=self._generate_job_id()
        src,dst=spec
        factor=self.handler.block_size_factor
        mapping=np.empty((len(src.block_ids)*factor,2),dtype=np.int64)
        expand_block_ids(src.block_ids,factor,mapping[:,0])
        expand_block_ids(dst.block_ids,1,mapping[:,1])
        tensor=torch.from_numpy(mapping)
        self.handler._record_h2d_geometry(job_id=job,logical_block_count=len(src.block_ids),physical_block_count=len(mapping))
        log('jobs.jsonl',dict(job_id=job,request_id=req))
        self.ops.layer=0
        before=self.ops.batch_counts()
        path_before=self.ops.path_counts()
        self.ops.retained=[tensor]
        payload=[SimpleNamespace(tensor=(host,tensor)) for host in self.handler.cpu_tensors]
        gen=self.native.batched_to_gpu([0],[len(mapping)],slot_mapping=torch.arange(len(mapping)),sync=True)
        next(gen)
        gen.send([payload[0]])
        self.active=dict(req=req,job=job,payload=payload,gen=gen,index=0,batch_before=before,path_before=path_before,expected_records=2*len(mapping)*len(self.names))
        if MODE=='serial':
            for item in payload[1:]:
                gen.send([item])
            next(gen)
            self.native.load_stream.synchronize()
            for layer in range(len(self.names)):
                assert self.wait_auth(layer)==0
                log('layer_waits.jsonl',dict(request_id=req,layer=layer,mode=MODE,completed=True))
            gen.close()
            after=self.ops.batch_counts()
            log('batch_stats.jsonl',dict(request_id=req,job_id=job,mode=MODE,
                expected_records=self.active['expected_records'],batch_launches=after[0]-before[0],
                batched_records=after[1]-before[1]))
            counters=self.ops.path_counts()
            log('path_counters.jsonl',dict(request_id=req,job_id=job,
                records=self.active['expected_records'],**{k:v-path_before[k] for k,v in counters.items()}))
            self.completed.add(req)
            _retire_h2d_prediction_scope()
            self.check_loaded_bytes(self.active)
            self.active=None
            self.retain_copy_inputs()

    def wait_for_layer_load(self,name):
        if self.active is None:
            return
        a=self.active
        layer=a['index']
        assert name==self.names[layer],(name,self.names[layer])
        if layer+1<len(self.names):
            a['gen'].send([a['payload'][layer+1]])
        else:
            next(a['gen'])
        if MODE!='plain':
            assert self.wait_auth(layer)==0,'Authentication failure blocks consumption'
        log('layer_waits.jsonl',dict(request_id=a['req'],layer=layer,mode=MODE,completed=True))
        a['index']+=1
        if a['index']==len(self.names):
            after=self.ops.batch_counts()
            log('batch_stats.jsonl',dict(request_id=a['req'],job_id=a['job'],mode=MODE,
                expected_records=a['expected_records'],batch_launches=after[0]-a['batch_before'][0],
                batched_records=after[1]-a['batch_before'][1]))
            counters=self.ops.path_counts()
            log('path_counters.jsonl',dict(request_id=a['req'],job_id=a['job'],
                records=a['expected_records'],**{k:v-a['path_before'][k] for k,v in counters.items()}))
            a['gen'].close()
            self.completed.add(a['req'])
            _retire_h2d_prediction_scope()
            self.check_loaded_bytes(self.active)
            self.active=None
            self.retain_copy_inputs()

    def take_secure_store_metadata(self):
        store=super().take_secure_store_metadata()
        if not store and not self.completed:
            return None
        result=NativeMetadata(secure_store_ready_counts=store.secure_store_ready_counts if store else {},completed_native_loads=self.completed)
        self.completed=set()
        return result

class NativeLayerConnector(SecureInferOffloadingConnector):
    def __init__(self,config,role,kv_cache_config=None):
        KVConnectorBase_V1.__init__(self,config,role,kv_cache_config)
        spec=OffloadingSpecFactory.create_spec(config,kv_cache_config)
        self.connector_scheduler=NativeScheduler(spec) if role==KVConnectorRole.SCHEDULER else None
        self.connector_worker=NativeWorker(spec) if role==KVConnectorRole.WORKER else None

    def wait_for_layer_load(self,layer_name):
        if self.connector_worker:
            self.connector_worker.wait_for_layer_load(layer_name)
