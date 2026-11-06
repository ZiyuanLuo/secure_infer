"""Static suffix-aware prefix reuse on native layerwise secure reload."""
from dataclasses import dataclass, field, replace
import os
import threading
from collections import deque
import time
import numpy as np
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole
from secure_infer.kv_offload.native_layerwise import (
    NativeScheduler, NativeWorker, NativeMetadata, log)
from secure_infer.kv_offload.npu_predict import _retire_h2d_prediction_scope
from secure_infer.kv_offload.secure_connector import (
    SecureInferOffloadingConnector, SecureInferOffloadingConnectorMetadata)
from . import entry_gate

CHUNK=int(os.environ.get('RUNTIME_COMPUTE_CHUNK','1024'))
FETCH=int(os.environ.get('RUNTIME_LOAD_CHUNK','2048'))

def trace(event, **row):
    log('reuse_timeline.jsonl', dict(event=event,time_ns=time.monotonic_ns(),**row))

@dataclass
class RuntimeMetadata(SecureInferOffloadingConnectorMetadata):
    kickoff: dict=field(default_factory=dict)
    controls: dict=field(default_factory=dict)

class RuntimeStorageScheduler(NativeScheduler):
    def __init__(self,spec):
        super().__init__(spec)
        config=spec.vllm_config.scheduler_config
        if self.gpu_block_size != 128 or self.offloaded_block_size != 128:
            raise ValueError('RUNTIME native loader currently requires 128-token blocks')
        if CHUNK % 128 or FETCH % 128 or CHUNK<=0 or FETCH<=0:
            raise ValueError('RUNTIME compute/load chunks must be positive multiples of 128')
        self.entry_layers=spec.vllm_config.model_config.hf_config.num_hidden_layers
        self.reuse_requests={}; self.kickoff={}; self.preempted_loads=set()
        self.scale_scheduler_config=config
        self.ordinary_prefill_threshold=config.long_prefill_token_threshold

    def get_num_new_matched_tokens(self,request,num_computed_tokens):
        if request.request_id in self.preempted_loads:
            # Its earlier load may still be in flight; recompute instead of
            # starting a second load into newly allocated blocks.
            return 0,False
        tokens,_=super().get_num_new_matched_tokens(request,num_computed_tokens)
        if not tokens:
            return tokens,False
        # A device-resident head of num_computed_tokens stays in place; the
        # split applies to the host-resident tokens that follow it.
        base=num_computed_tokens
        suffix=request.num_prompt_tokens-base-tokens
        decision=entry_gate.choose(tokens,suffix,CHUNK,self.entry_layers,base)
        trace('entry_cost_decision',request_id=request.request_id,N=tokens,
              U=suffix,D=base,**decision)
        if decision['policy']=='load':return tokens,False
        if decision['policy']=='compute':return 0,False
        q=decision['q']
        limit=self.scale_scheduler_config.long_prefill_token_threshold
        self.scale_scheduler_config.long_prefill_token_threshold=min(CHUNK,q,limit) if limit>0 else min(CHUNK,q)
        # R and q are absolute positions: recompute [base, base+q), reload [base+q, base+tokens).
        self.reuse_requests[request.request_id]=dict(R=base+tokens,q=base+q,D=base,suffix_started=False)
        return 0,False

    def update_state_after_alloc(self,request,blocks,num_external_tokens):
        super().update_state_after_alloc(request,blocks,num_external_tokens)
        state=self.reuse_requests.get(request.request_id)
        if state is None:return
        ids=blocks.get_block_ids()[0]; prefix=state['R']; head=state['D']//self.offloaded_block_size
        assert prefix%self.offloaded_block_size==0 and state['D']%self.offloaded_block_size==0
        hashes=list(self._get_block_hashes(request,start_idx=head,end_idx=prefix//self.offloaded_block_size))
        source=self.manager.prepare_load(hashes)
        self._reqs_being_loaded[request.request_id].update(hashes)
        if self._blocks_being_loaded is not None:self._blocks_being_loaded.update(hashes)
        self._next_stored_block_idx[request.request_id]=prefix//self.offloaded_block_size
        assert len(ids)>=prefix//self.gpu_block_size
        self.kickoff[request.request_id]=dict(R=prefix,q=state['q'],
            # Source is indexed by absolute block position; the device-resident
            # head has no host source and is never read (reloads start at q >= D).
            source=[-1]*head+list(source.block_ids),destination=list(ids[:prefix//self.gpu_block_size]))

    def build_connector_meta(self,output):
        base=super().build_connector_meta(output); controls={}
        for rid,count in output.num_scheduled_tokens.items():
            state=self.reuse_requests.get(rid)
            if state:
                request=self._requests[rid]
                controls[rid]=dict(start=request.num_computed_tokens,end=request.num_computed_tokens+count,
                    R=state['R'],q=state['q'],consume=state['suffix_started'])
                trace('compute_schedule',request_id=rid,**controls[rid])
        result=RuntimeMetadata(reqs_to_load=base.reqs_to_load,reqs_to_store=base.reqs_to_store,
            kickoff=self.kickoff,controls=controls)
        self.kickoff={}
        return result

    def request_finished(self,request,block_ids):
        result=super().request_finished(request,block_ids)
        self.reuse_requests.pop(request.request_id,None)
        self.preempted_loads.discard(request.request_id)
        return result

class RuntimeScheduler(Scheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ordinary_prefill_threshold=self.scheduler_config.long_prefill_token_threshold

    def _preempt_request(self,request,timestamp):
        # vLLM frees a preempted request's blocks without telling the reload
        # thread; a load still in flight must not be repeated into new blocks.
        cs=self.connector.connector_scheduler if self.connector else None
        if isinstance(cs,RuntimeStorageScheduler) and getattr(cs,"_reqs_being_loaded",{}).get(request.request_id):
            trace('loading_request_preempted',request_id=request.request_id,
                  num_computed=request.num_computed_tokens,
                  reuse=cs.reuse_requests.get(request.request_id))
            cs.preempted_loads.add(request.request_id)
            cs.reuse_requests.pop(request.request_id,None)
        return super()._preempt_request(request,timestamp)

    def schedule(self):
        cs=self.connector.connector_scheduler if self.connector else None
        self.scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold
        if isinstance(cs,RuntimeStorageScheduler):
            # vLLM preempts from the end of the running list; keep requests
            # with an in-flight host load at the front (stable order otherwise).
            if isinstance(self.running,list):
                self.running.sort(key=lambda r:not getattr(cs,"_reqs_being_loaded",{}).get(r.request_id))
            for request in self.running:
                state=cs.reuse_requests.get(request.request_id)
                if not state or state['suffix_started']:continue
                if request.num_computed_tokens>=state['q']:
                    assert request.num_computed_tokens==state['q']
                    state['suffix_started']=True
                    request.num_computed_tokens=state['R']
                    trace('prefix_recompute_complete',request_id=request.request_id,
                          compute=state['q'],R=state['R'])
                else:
                    limit=self.scheduler_config.long_prefill_token_threshold
                    remaining=min(CHUNK,state['q']-request.num_computed_tokens)
                    self.scheduler_config.long_prefill_token_threshold=min(limit,remaining) if limit>0 else remaining
        allocate=self.kv_cache_manager.allocate_slots
        def allocate_for_tail(request,num_new_tokens,*args,**kwargs):
            state=cs.reuse_requests.get(request.request_id) if isinstance(cs,RuntimeStorageScheduler) else None
            # A request whose host load is in flight reserves blocks for its whole
            # prompt up front, so it never allocates (and triggers preemption of
            # itself) while the load still writes into its blocks.
            if state or getattr(cs,"_reqs_being_loaded",{}).get(request.request_id):
                total=getattr(request,"num_prompt_tokens",None)
                if total is None:
                    total=state['R'] if state else request.num_computed_tokens
                needed=max(0,total-request.num_computed_tokens-num_new_tokens)
                kwargs['num_lookahead_tokens']=max(kwargs.get('num_lookahead_tokens',0),needed)
            return allocate(request,num_new_tokens,*args,**kwargs)
        self.kv_cache_manager.allocate_slots=allocate_for_tail
        try:return super().schedule()
        finally:
            self.kv_cache_manager.allocate_slots=allocate
            self.scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold

class RuntimeWorker(NativeWorker):
    def __init__(self,spec):
        super().__init__(spec)
        self.load_jobs={}
        self.lock=threading.Condition()
        self.newly_done=set()
        self.consuming={}
        self.submission_lock=threading.Lock()

    def snapshot(self,state):
        return dict(q=state['q'],done=state['done'],error=state['error'],
                    verified=list(state['verified']))

    def observe_layers(self,req_id,state):
        # Publish each layer only after its device writes and authentication finish.
        try:
            torch.npu.set_device(self.handler.npu_tensors[0][0].device)
            while True:
                with self.lock:
                    while not state['events'] and not state['submissions_done'] and not state['error']:
                        self.lock.wait()
                    if state['error']:return
                    if not state['events']:break
                    layer,event=state['events'].popleft()
                event.synchronize()
                assert self.wait_auth(layer)==0,'Authentication failure blocks consumption'
                if os.environ.get('RUNTIME_CHECK_LOADED_BYTES')=='1':
                    source=torch.as_tensor(state['source'][state['q']//128:],dtype=torch.long)
                    destination=torch.as_tensor(state['destination'][state['q']//128:],dtype=torch.long)
                    for host,device in zip(self.handler.cpu_tensors[layer],self.handler.npu_tensors[layer]):
                        assert torch.equal(host[source],device[destination.to(device=device.device)].cpu())
                with self.lock:
                    assert state['verified'][layer]==state['R']
                    state['verified'][layer]=state['q']
                    trace('layer_ready',request_id=req_id,layer=layer,start=state['q'],end=state['R'])
                    self.lock.notify_all()
        except BaseException as error:
            with self.lock:
                state['error']=repr(error)
                trace('fetch_error',request_id=req_id,error=repr(error))
                self.lock.notify_all()

    def fetch(self,req_id,state):
        # NativeTransfer and crypto prediction scope are shared by this worker.
        # Queue requests through one submitter while compute batches run concurrently.
        with self.submission_lock:
            self.fetch_serialized(req_id,state)

    def fetch_serialized(self,req_id,state):
        observer=threading.Thread(target=self.observe_layers,args=(req_id,state),daemon=True)
        try:
            torch.npu.set_device(self.handler.npu_tensors[0][0].device)
            observer.start()
            start,end=state['q'],state['R']
            mapping=np.column_stack((state['source'][start//128:end//128],
                                    state['destination'][start//128:end//128])).astype(np.int64)
            tensor=torch.from_numpy(mapping)
            self.ops.retained.append(tensor)
            trace('fetch_claim',request_id=req_id,start=start,end=end,execution_order='layer')
            with torch.npu.stream(self.native.load_stream):
                for layer in range(len(self.names)):
                    with self.lock:
                        if state['error']:raise RuntimeError(state['error'])
                    self.ops.layer=layer
                    self.ops.causal_group_blocks=FETCH//128
                    self.ops.single_layer_kv_transfer((self.handler.cpu_tensors[layer],tensor),
                        self.native.kvcaches[layer])
                    event=torch.npu.Event()
                    event.record(self.native.load_stream)
                    log('load_submissions.jsonl',dict(request_id=req_id,layer=layer,
                        start=start,end=end,records=2*len(mapping)))
                    with self.lock:
                        state['events'].append((layer,event))
                        self.lock.notify_all()
            _retire_h2d_prediction_scope()
            self.retain_copy_inputs()
        except BaseException as error:
            with self.lock:
                state['error']=repr(error)
                trace('fetch_error',request_id=req_id,error=repr(error))
                self.lock.notify_all()
        finally:
            with self.lock:
                state['submissions_done']=True
                self.lock.notify_all()
            if observer.is_alive():observer.join(timeout=120)
            with self.lock:
                if observer.is_alive():state['error']='authentication observer timeout'
                state['done']=True
                self.newly_done.add(req_id)
                trace('fetch_done',request_id=req_id,**self.snapshot(state))
                self.lock.notify_all()

    def start_kv_transfers(self,metadata):
        if not isinstance(metadata,RuntimeMetadata):
            return super().start_kv_transfers(metadata)
        self.consuming={}
        kickoff=dict(metadata.kickoff)
        controls=dict(metadata.controls)
        for req_id,(source,destination) in metadata.reqs_to_load.items():
            assert self.handler.block_size_factor==1
            assert len(source.block_ids)==len(destination.block_ids)
            prefix=len(source.block_ids)*128
            kickoff[req_id]=dict(R=prefix,q=0,source=list(source.block_ids),destination=list(destination.block_ids))
            controls[req_id]=dict(start=prefix,end=prefix,R=prefix,q=0,consume=True)
        metadata=replace(metadata,reqs_to_load={})
        # Retain ordinary asynchronous stores; all reloads use the background submitter.
        super().start_kv_transfers(metadata)
        for req_id,kick in kickoff.items():
            assert req_id not in self.load_jobs
            state=dict(**kick,done=False,error=None,verified=[kick['R'] for _ in self.names],
                events=deque(),submissions_done=False)
            self.load_jobs[req_id]=state
            state['thread']=threading.Thread(target=self.fetch,args=(req_id,state),daemon=True)
            state['thread'].start()
            trace('fetch_started',request_id=req_id,R=kick['R'],q=kick['q'])
        for req_id,control in controls.items():
            state=self.load_jobs.get(req_id)
            if state:
                assert control['q']==state['q']
                assert control['end']<=state['q'] or control['start']>=state['R']
                if control['consume']:
                    self.consuming[req_id]=state
                    trace('tail_consume_begin',request_id=req_id,q=state['q'])

    def wait_for_layer_load(self,name):
        layer=self.names.index(name)
        # Attention consumes a batch: every reloading request in that batch
        # must have authenticated KV for this layer before it can run.
        for req_id,state in self.consuming.items():
            begin=time.monotonic_ns()
            with self.lock:
                while state['verified'][layer]>state['q'] and not state['error']:
                    self.lock.wait(timeout=30)
                assert not state['error'],state['error']
                assert state['verified'][layer]==state['q']
            trace('native_layer_wait_end',request_id=req_id,layer=layer,
                wait_ns=time.monotonic_ns()-begin,authenticated=True,written=True)
            if layer==len(self.names)-1:
                state['thread'].join(timeout=120)
                assert not state['thread'].is_alive()
                assert not state['error'],state['error']
        if layer==len(self.names)-1:self.consuming={}

    def take_secure_store_metadata(self):
        base=super().take_secure_store_metadata()
        with self.lock:
            for state in self.load_jobs.values():
                if state['error']:raise RuntimeError(state['error'])
            done=self.newly_done
            self.newly_done=set()
            for rid in done:self.load_jobs.pop(rid,None)
        if base is None and not done:return None
        return NativeMetadata(secure_store_ready_counts=base.secure_store_ready_counts if base else {},
                              completed_native_loads=done,
                              native_load_ready_counts={rid:1 for rid in done})

class RuntimeConnector(SecureInferOffloadingConnector):
    def __init__(self,config,role,kv_cache_config=None):
        KVConnectorBase_V1.__init__(self,config,role,kv_cache_config)
        spec=OffloadingSpecFactory.create_spec(config,kv_cache_config)
        self.connector_scheduler=RuntimeStorageScheduler(spec) if role==KVConnectorRole.SCHEDULER else None
        self.connector_worker=RuntimeWorker(spec) if role==KVConnectorRole.WORKER else None

    def wait_for_layer_load(self,layer_name):
        if self.connector_worker:self.connector_worker.wait_for_layer_load(layer_name)

    def get_kv_connector_stats(self):
        from vllm.v1.outputs import KVConnectorOutput
        if 'kv_connector_worker_meta' in KVConnectorOutput.__annotations__:
            return super().get_kv_connector_stats()
        from .legacy_metadata import RuntimeControlStats
        metadata=self.build_connector_worker_meta()
        return RuntimeControlStats({'metadata': metadata}) if metadata is not None else None

    def update_connector_output(self,output):
        from .legacy_metadata import RuntimeControlStats
        stats=output.kv_connector_stats
        if isinstance(stats,RuntimeControlStats):
            output.kv_connector_worker_meta=stats.data.get('metadata')
            output.kv_connector_stats=None
        return super().update_connector_output(output)
