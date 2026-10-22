"""RUNTIME prefix reuse on native layerwise secure loading.

Admission uses the framework's host-cache match, never encoded request IDs.
The current loader supports one active request (max_num_seqs=1).
"""
from dataclasses import dataclass, field, replace
import json
import os
import threading
from collections import deque
import time
from pathlib import Path
import numpy as np
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole
from secure_infer.kv_offload.native_layerwise import (
    NativeScheduler, NativeWorker, NativeMetadata, NativeLayerConnector, log,
    SecureInferOffloadingConnectorMetadata, _retire_h2d_prediction_scope)
from . import entry_gate
from .stop_more import decide

CHUNK=int(os.environ.get('RUNTIME_COMPUTE_CHUNK','1024'))
FETCH=int(os.environ.get('RUNTIME_LOAD_CHUNK','2048'))

def trace(event, **row):
    log('reuse_timeline.jsonl', dict(event=event,time_ns=time.monotonic_ns(),**row))

def progress_path():
    return Path(os.environ['RUNTIME_RESULT_DIR'])/'reuse_progress.json'

def publish_progress(req_id, state):
    row=dict(request_id=req_id,assigned=state['assigned'],ready=state['ready'],
        pending=state['pending'],done=state['done'],error=state['error'],
        loaded_blocks=state['loaded_blocks'],time_ns=time.monotonic_ns(),
        claim_ns=state.get('claim_ns'),last_service_ms=state.get('last_service_ms'),
        last_load_tokens=state.get('last_load_tokens'),layers=state['layers'],
        layer_ready=state.get('layer_ready',0),ready_offsets=state.get('ready_offsets',[]))
    path=progress_path(); temporary=path.with_name(
        f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(row)); os.replace(temporary,path)

@dataclass
class RuntimeMetadata(SecureInferOffloadingConnectorMetadata):
    kickoff: dict=field(default_factory=dict)
    controls: dict=field(default_factory=dict)

@dataclass
class RuntimeWorkerMetadata(NativeMetadata):
    frontiers: dict=field(default_factory=dict)
    def aggregate(self,other):
        super().aggregate(other)
        self.frontiers.update(other.frontiers)
        return self

class RuntimeStorageScheduler(NativeScheduler):
    def __init__(self,spec):
        super().__init__(spec)
        config=spec.vllm_config.scheduler_config
        if config.max_num_seqs != 1:
            raise ValueError('RUNTIME native loader currently requires max_num_seqs=1')
        if self.gpu_block_size != 128 or self.offloaded_block_size != 128:
            raise ValueError('RUNTIME native loader currently requires 128-token blocks')
        if CHUNK % 128 or FETCH % 128 or CHUNK<=0 or FETCH<=0:
            raise ValueError('RUNTIME compute/load chunks must be positive multiples of 128')
        self.entry_layers=spec.vllm_config.model_config.hf_config.num_hidden_layers
        self.cake={}; self.kickoff={}; self.frontiers={}
        self.scale_scheduler_config=config
        self.ordinary_prefill_threshold=config.long_prefill_token_threshold

    def get_num_new_matched_tokens(self,request,num_computed_tokens):
        tokens,_=super().get_num_new_matched_tokens(request,num_computed_tokens)
        # Partial device hits retain the native path; q applies to a host prefix.
        if num_computed_tokens or not tokens:
            return tokens,False
        decision=entry_gate.choose(tokens,request.num_prompt_tokens-tokens,CHUNK,self.entry_layers)
        request.entry_model=decision['model']
        request.entry_selected_policy=decision['policy']
        trace('entry_cost_decision',request_id=request.request_id,N=tokens,
              U=request.num_prompt_tokens-tokens,**decision)
        if decision['policy']=='load':
            self.scale_scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold
            return tokens,False
        if decision['policy']=='compute':
            self.scale_scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold
            return 0,False
        self.scale_scheduler_config.long_prefill_token_threshold=min(CHUNK,decision['predicted_q'])
        self.cake[request.request_id]=dict(R=tokens,stopped=False,target=None,
                                         meet_compute=None,policy='mixed',started_ns=time.monotonic_ns(),
                                         provisional_q=decision['predicted_q'])
        return 0,False

    def update_state_after_alloc(self,request,blocks,num_external_tokens):
        super().update_state_after_alloc(request,blocks,num_external_tokens)
        state=self.cake.get(request.request_id)
        if state is None:return
        ids=blocks.get_block_ids()[0]; prefix=state['R']
        assert prefix%self.offloaded_block_size==0
        hashes=list(self._get_block_hashes(request,end_idx=prefix//self.offloaded_block_size))
        source=self.manager.prepare_load(hashes)
        self._reqs_being_loaded[request.request_id].update(hashes)
        if self._blocks_being_loaded is not None:self._blocks_being_loaded.update(hashes)
        self._next_stored_block_idx[request.request_id]=prefix//self.offloaded_block_size
        assert len(ids)>=prefix//self.gpu_block_size
        self.kickoff[request.request_id]=dict(R=prefix,source=list(source.block_ids),
            destination=list(ids[:prefix//self.gpu_block_size]),target=None,planned_target=state['provisional_q'])

    def build_connector_meta(self,output):
        base=super().build_connector_meta(output); controls={}
        for rid,count in output.num_scheduled_tokens.items():
            state=self.cake.get(rid)
            if state:
                request=self._requests[rid]
                controls[rid]=dict(start=request.num_computed_tokens,end=request.num_computed_tokens+count,
                    R=state['R'],stop=state['stopped'],meet_compute=state['meet_compute'],dynamic_target=True)
                trace('compute_schedule',request_id=rid,**controls[rid])
        result=RuntimeMetadata(reqs_to_load=base.reqs_to_load,reqs_to_store=base.reqs_to_store,
            layerwise_load_req_ids=base.layerwise_load_req_ids,kickoff=self.kickoff,controls=controls)
        self.kickoff={}
        return result

    def update_connector_output(self,output):
        super().update_connector_output(output)
        meta=getattr(output, "kv_connector_worker_meta", None)
        if isinstance(meta,RuntimeWorkerMetadata):self.frontiers.update(meta.frontiers)

    def request_finished(self,request,block_ids):
        result=super().request_finished(request,block_ids)
        self.cake.pop(request.request_id,None); self.frontiers.pop(request.request_id,None)
        return result

class RuntimeScheduler(Scheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ordinary_prefill_threshold=self.scheduler_config.long_prefill_token_threshold

    def schedule(self):
        cs=self.connector.connector_scheduler if self.connector else None
        self.scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold
        if isinstance(cs,RuntimeStorageScheduler):
            for request in self.running:
                state=cs.cake.get(request.request_id)
                if not state or state['stopped']:continue
                observed=cs.frontiers.get(request.request_id)
                if observed:
                    if observed['error']:raise RuntimeError(observed['error'])
                    early=False
                    if observed['assigned']>request.num_computed_tokens:
                        model=request.entry_model
                        delta=min(CHUNK,state['R']-request.num_computed_tokens)
                        distance=state['provisional_q']-request.num_computed_tokens
                        if distance>0:delta=min(delta,distance)
                        compute=entry_gate.compute_ms(model,delta,request.num_computed_tokens)
                        decision=decide(q=request.num_computed_tokens,r=state['R'],
                            u=request.num_prompt_tokens-state['R'],observed=observed,
                            now_ns=time.monotonic_ns(),step_ms=compute,step_tokens=delta,
                            layers=observed['layers'],suffix_ms_override=entry_gate.compute_ms(
                                model,request.num_prompt_tokens-state['R'],state['R']),
                            next_compute_ms_override=compute,error_margin_ms=model['residual_ms'],
                            queued_compute_ms=max(0.,entry_gate.head_ms(model,request.num_computed_tokens,CHUNK)-
                                (time.monotonic_ns()-state['started_ns'])/1e6))
                        early=decision['stop']
                        if decision['reason']=='insufficient current-request observations':
                            early=request.num_computed_tokens>=state['provisional_q']
                        trace('boundary_cost_decision',request_id=request.request_id,**decision)
                    if early or observed['assigned']<=request.num_computed_tokens:
                        q=request.num_computed_tokens
                        state.update(stopped=True,meet_compute=q)
                        request.num_computed_tokens=state['R']
                        trace('meet_decision',request_id=request.request_id,compute=q,R=state['R'],
                              reason='stop' if early else 'coverage')
                if not observed and request.num_computed_tokens>=state['provisional_q']:
                    q=request.num_computed_tokens
                    state.update(stopped=True,meet_compute=q)
                    request.num_computed_tokens=state['R']
                    trace('meet_decision',request_id=request.request_id,compute=q,R=state['R'],
                          reason='entry estimate pending progress feedback')
                if not state['stopped']:
                    distance=state['provisional_q']-request.num_computed_tokens
                    self.scheduler_config.long_prefill_token_threshold=min(CHUNK,state['R']-request.num_computed_tokens,
                        distance if distance>0 else CHUNK)
        allocate=self.kv_cache_manager.allocate_slots
        def allocate_for_tail(request,num_new_tokens,*args,**kwargs):
            state=cs.cake.get(request.request_id) if isinstance(cs,RuntimeStorageScheduler) else None
            if state:
                needed=max(0,state['R']-request.num_computed_tokens-num_new_tokens)
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
        self.cake_jobs={}
        self.lock=threading.Condition()
        self.newly_done=set()
        self.cake_consuming=None

    def snapshot(self,state):
        return dict(assigned=state['assigned'],ready=state['ready'],pending=state['pending'],
            done=state['done'],error=state['error'],loaded_blocks=state['loaded_blocks'],
            claim_ns=state.get('claim_ns'),last_service_ms=state.get('last_service_ms'),
            last_ready_ms=state.get('last_ready_ms'),last_load_tokens=state.get('last_load_tokens'),
            layers=state['layers'],layer_ready=state.get('layer_ready',0),ready_offsets=state.get('ready_offsets',[]),
            verified=state.get('verified',[]))

    def observe_layers(self,req_id,state):
        # A single observer owns authentication status consumption. Each event
        # covers one submitted layer/range, including ranges added after STOP.
        try:
            torch.npu.set_device(self.handler.npu_tensors[0][0].device)
            while True:
                with self.lock:
                    while not state['events'] and not state['submissions_done'] and not state['error']:
                        self.lock.wait()
                    if state['error']:return
                    if not state['events']:break
                    layer,start,end,event=state['events'].popleft()
                event.synchronize()
                assert self.wait_auth(layer)==0,'Authentication failure blocks consumption'
                if os.environ.get('RUNTIME_CHECK_LOADED_BYTES')=='1':
                    source=torch.as_tensor(state['source'][start//128:end//128],dtype=torch.long)
                    destination=torch.as_tensor(state['destination'][start//128:end//128],dtype=torch.long)
                    for host,device in zip(self.handler.cpu_tensors[layer],self.handler.npu_tensors[layer]):
                        assert torch.equal(host[source],device[destination.to(device=device.device)].cpu())
                with self.lock:
                    assert state['verified'][layer]==end
                    state['verified'][layer]=start
                    state['ready']=max(state['verified'])
                    target=state['pending'][0] if state['pending'] else state['ready']
                    for i,frontier in enumerate(state['verified']):
                        if frontier<=target and not state['ready_offsets'][i]:
                            state['ready_offsets'][i]=(time.monotonic_ns()-state['claim_ns'])/1e6
                    state['layer_ready']=next((i for i,x in enumerate(state['verified']) if x>target),len(self.names))
                    trace('layer_ready',request_id=req_id,layer=layer,start=start,end=end)
                    self.lock.notify_all()
        except BaseException as error:
            with self.lock:
                state['error']=repr(error)
                trace('fetch_error',request_id=req_id,error=repr(error))
                self.lock.notify_all()

    def fetch(self,req_id,state):
        observer=threading.Thread(target=self.observe_layers,args=(req_id,state),daemon=True)
        try:
            torch.npu.set_device(self.handler.npu_tensors[0][0].device)
            observer.start()
            with torch.npu.stream(self.native.load_stream):
                while True:
                    with self.lock:
                        if state['error']:raise RuntimeError(state['error'])
                        if state['target'] is not None:
                            start=state['target']
                            if state['ready']<=start:
                                if state['stop']:break
                                self.lock.wait()
                                continue
                        elif state['pending'] is None:
                            end=state['ready']
                            if end<=state['compute_end']:break
                            floor=state['planned_target']
                            if end<=floor:
                                self.lock.wait()
                                continue
                            start=max(floor,state['compute_end'],end-FETCH)
                        else:
                            start=state['pending'][0]
                        # STOP can widen the remaining range at a layer
                        # boundary. Already submitted layers remain valid.
                        if state['pending'] is None or state['pending'][0]!=start:
                            end=max(state['submitted'])
                            state['pending']=(start,end)
                            state['assigned']=start
                            state['claim_ns']=time.monotonic_ns()
                            state['ready_offsets']=[0.0 for _ in self.names]
                            state['layer_ready']=0
                            trace('fetch_claim',request_id=req_id,start=start,end=end,
                                  execution_order='layer' if state['target'] is not None else 'chunk')
                        layer=next((i for i,x in enumerate(state['submitted']) if x>start),None)
                        if layer is None:
                            while state['ready']>start and not state['error']:
                                if state['target'] is not None and state['target']<start:break
                                self.lock.wait()
                            if state['error']:raise RuntimeError(state['error'])
                            if state['target'] is not None and state['target']<start:continue
                            elapsed=(time.monotonic_ns()-state['claim_ns'])/1e6
                            state['last_service_ms']=elapsed
                            state['last_load_tokens']=state['pending'][1]-start
                            state['loaded_blocks']=(state['R']-state['ready'])//128
                            trace('fetch_usable',request_id=req_id,start=start,end=state['pending'][1])
                            state['pending']=None
                            publish_progress(req_id,state)
                            self.lock.notify_all()
                            continue
                        end=state['submitted'][layer]
                        state['submitted'][layer]=start
                    mapping=np.column_stack((state['source'][start//128:end//128],
                        state['destination'][start//128:end//128])).astype(np.int64)
                    tensor=torch.from_numpy(mapping)
                    self.ops.retained.append(tensor)
                    self.ops.layer=layer
                    self.ops.causal_group_blocks=FETCH//128
                    self.ops.single_layer_kv_transfer((self.handler.cpu_tensors[layer],tensor),
                        self.native.kvcaches[layer],torch.arange(len(mapping)),False,
                        self.native.kv_format.value,True,self.native.vllm_two_major)
                    event=torch.npu.Event()
                    event.record(self.native.load_stream)
                    log('load_submissions.jsonl',dict(request_id=req_id,layer=layer,
                        start=start,end=end,records=2*len(mapping)))
                    with self.lock:
                        state['events'].append((layer,start,end,event))
                        self.lock.notify_all()
            self.ops.causal_group_blocks=FETCH//128
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
                state['loaded_blocks']=(state['R']-state['ready'])//128
                self.newly_done.add(req_id)
                publish_progress(req_id,state)
                trace('fetch_done',request_id=req_id,**self.snapshot(state))
                self.lock.notify_all()

    def start_kv_transfers(self,metadata):
        if not isinstance(metadata,RuntimeMetadata):
            return super().start_kv_transfers(metadata)
        # Use the same background submitter for full and partial reloads.
        # The model thread retains only the layer authentication/write wait.
        kickoff=dict(metadata.kickoff)
        controls=dict(metadata.controls)
        for req_id,(source,destination) in metadata.reqs_to_load.items():
            assert self.handler.block_size_factor==1
            assert len(source.block_ids)==len(destination.block_ids)
            prefix=len(source.block_ids)*128
            kickoff[req_id]=dict(R=prefix,source=list(source.block_ids),
                destination=list(destination.block_ids),target=0,planned_target=0)
            controls[req_id]=dict(start=prefix,end=prefix,R=prefix,stop=True,
                meet_compute=0,dynamic_target=False)
        metadata=replace(metadata,reqs_to_load={})
        for req_id,control in controls.items():
            state=self.cake_jobs.get(req_id)
            if state:
                with self.lock:
                    state['compute_end']=control['end']
                    state['stop']=control['stop']
                    if control['stop'] and control.get('dynamic_target'):
                        state['target']=control['meet_compute']
                    pending=state['pending']
                    # Complete any previous tail write before a scheduled chunk overwrites it.
                    while pending and pending[0]<control['end'] and pending[1]>control['start']:
                        trace('overlap_write_wait',request_id=req_id,start=control['start'],end=control['end'],pending=pending)
                        self.lock.wait(timeout=30)
                        if state['error']:raise RuntimeError(state['error'])
                        pending=state['pending']
                    trace('compute_ready_to_run',request_id=req_id,
                        start=control['start'],end=control['end'],
                        ready=state['ready'],pending=state['pending'])
                    self.lock.notify_all()
                if control['stop'] and not state.get('gate_started') and control['meet_compute']<state['R']:
                    state['gate_started']=True
                    self.cake_consuming=(req_id,state,control['meet_compute'])
                    trace('tail_consume_begin',request_id=req_id,meet_compute=control['meet_compute'])
        # Retain ordinary stores and the unchanged native generator for load-only.
        super().start_kv_transfers(metadata)
        for req_id,kick in kickoff.items():
            assert not self.cake_jobs or all(s['done'] for s in self.cake_jobs.values())
            control=controls[req_id]
            state=dict(**kick,assigned=kick['R'],ready=kick['R'],pending=None,
                compute_end=control['end'],stop=control['stop'],done=False,error=None,loaded_blocks=0,
                dynamic_target=control['dynamic_target'],layers=len(self.names),
                layer_ready=0,
                diagnostic_drain=('-cakesync-' in req_id or '-halfsync-' in req_id),
                submitted=[kick['R'] for _ in self.names],verified=[kick['R'] for _ in self.names],
                events=deque(),submissions_done=False,ready_offsets=[0.0 for _ in self.names])
            self.cake_jobs[req_id]=state
            if control['stop'] and control['meet_compute']<state['R']:
                state['gate_started']=True
                self.cake_consuming=(req_id,state,control['meet_compute'])
            publish_progress(req_id,state)
            state['thread']=threading.Thread(target=self.fetch,args=(req_id,state),daemon=True)
            state['thread'].start()
            trace('fetch_started',request_id=req_id,R=kick['R'],compute_end=control['end'])

    def wait_for_layer_load(self,name):
        super().wait_for_layer_load(name)
        if self.cake_consuming is None:
            return
        req_id,state,boundary=self.cake_consuming
        layer=self.names.index(name)
        begin=time.monotonic_ns()
        with self.lock:
            while state['verified'][layer]>boundary and not state['error']:
                self.lock.wait(timeout=30)
            assert not state['error'],state['error']
            assert state['verified'][layer]<=boundary
        trace('native_layer_wait_end',request_id=req_id,layer=layer,
            wait_ns=time.monotonic_ns()-begin,
            authenticated=True,written=True)
        if layer==len(self.names)-1:
            trace('consumption_wait_start',request_id=req_id)
            state['thread'].join(timeout=120)
            assert not state['thread'].is_alive()
            assert not state['error'],state['error']
            assert state['ready']<=boundary,(state['ready'],boundary)
            trace('consumption_wait_end',request_id=req_id,**self.snapshot(state))
            if state['diagnostic_drain']:
                trace('diagnostic_device_drain_begin',request_id=req_id)
                torch.npu.synchronize()
                trace('diagnostic_device_drain_end',request_id=req_id)
            self.cake_consuming=None

    def take_secure_store_metadata(self):
        base=super().take_secure_store_metadata()
        with self.lock:
            frontiers={req:self.snapshot(s) for req,s in self.cake_jobs.items()}
            done=self.newly_done
            self.newly_done=set()
            for rid in done:
                self.cake_jobs.pop(rid, None)
        if base is None and not frontiers and not done:
            return None
        return RuntimeWorkerMetadata(secure_store_ready_counts=base.secure_store_ready_counts if base else {},
            completed_native_loads=(base.completed_native_loads if base else set())|done,
            frontiers=frontiers)
class RuntimeConnector(NativeLayerConnector):
    def __init__(self,config,role,kv_cache_config=None):
        KVConnectorBase_V1.__init__(self,config,role,kv_cache_config)
        spec=OffloadingSpecFactory.create_spec(config,kv_cache_config)
        self.connector_scheduler=RuntimeStorageScheduler(spec) if role==KVConnectorRole.SCHEDULER else None
        self.connector_worker=RuntimeWorker(spec) if role==KVConnectorRole.WORKER else None


    def get_kv_connector_stats(self):
        from vllm.v1.outputs import KVConnectorOutput
        if 'kv_connector_worker_meta' in KVConnectorOutput.__annotations__:
            return super().get_kv_connector_stats()
        from .legacy_metadata import RuntimeControlStats
        metadata = self.build_connector_worker_meta()
        return RuntimeControlStats({'metadata': metadata}) if metadata is not None else None

    def update_connector_output(self, output):
        from .legacy_metadata import RuntimeControlStats
        stats = output.kv_connector_stats
        if isinstance(stats, RuntimeControlStats):
            output.kv_connector_worker_meta = stats.data.get('metadata')
            output.kv_connector_stats = None
        return super().update_connector_output(output)
