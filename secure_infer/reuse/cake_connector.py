"""Paper-derived Cake bidirectional execution on the Ascend secure executor."""
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
from .cake_progress import Progress
from secure_infer.kv_offload.transport_codec import settings

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
        self.progress_world_size=int(getattr(
            spec.vllm_config.parallel_config,'world_size',1))
        self.reuse_requests={}; self.kickoff={}; self.preempted_loads=set()
        self.scale_scheduler_config=config
        self.ordinary_prefill_threshold=config.long_prefill_token_threshold

    def get_num_new_matched_tokens(self,request,num_computed_tokens):
        if request.request_id in self.preempted_loads:
            # Its earlier load may still be in flight; recompute instead of
            # starting a second load into newly allocated blocks.
            return 0,False
        existing=self.reuse_requests.get(request.request_id)
        if existing:
            # A prior schedule may have failed allocation after creating the
            # reuse decision.  Reapply the split on every retry because the
            # parent scheduler restores its ordinary threshold at the end of
            # each schedule call.
            base=num_computed_tokens
            remaining=max(0,existing['q']-base)
            limit=self.scale_scheduler_config.long_prefill_token_threshold
            threshold=min(CHUNK,remaining,limit) if limit>0 else min(CHUNK,remaining)
            self.scale_scheduler_config.long_prefill_token_threshold=threshold
            if os.environ.get('RUNTIME_SCHED_TRACE')=='1':
                trace('entry_threshold_reapply',request_id=request.request_id,
                      config_id=id(self.scale_scheduler_config),
                      threshold=threshold,ordinary=limit)
            return 0,False
        tokens,_=super().get_num_new_matched_tokens(request,num_computed_tokens)
        if not tokens:
            return tokens,False
        # A device-resident head of num_computed_tokens stays in place; the
        # split applies to the host-resident tokens that follow it.
        base=num_computed_tokens
        suffix=request.num_prompt_tokens-base-tokens
        q=min(CHUNK,tokens)
        limit=self.scale_scheduler_config.long_prefill_token_threshold
        threshold=min(CHUNK,q,limit) if limit>0 else min(CHUNK,q)
        self.scale_scheduler_config.long_prefill_token_threshold=threshold
        if os.environ.get('RUNTIME_SCHED_TRACE')=='1':
            trace('entry_threshold_set',request_id=request.request_id,
                  config_id=id(self.scale_scheduler_config),
                  threshold=threshold,ordinary=limit)
        self.reuse_requests[request.request_id]=dict(R=base+tokens,q=base+q,D=base,suffix_started=False)
        trace('entry_cost_decision',request_id=request.request_id,N=tokens,U=suffix,D=base,
              policy='hybrid',q=q,threshold=threshold,
              reason='Cake bidirectional execution; no fitted split')
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
        destination=[int(block_id) for block_id in ids[:prefix//self.gpu_block_size]]
        state['destination']=destination
        trace('cake_destination_allocated',request_id=request.request_id,
              R=prefix,D=state['D'],blocks=destination)
        Progress(request.request_id).create(
            state['D'],state['q'],prefix,self.progress_world_size)
        self.kickoff[request.request_id]=dict(R=prefix,q=state['q'],
            # Source is indexed by absolute block position; the device-resident
            # head has no host source and is never read (reloads start at q >= D).
            source=[-1]*head+list(source.block_ids),destination=destination)

    def build_connector_meta(self,output):
        base=super().build_connector_meta(output); controls={}
        for rid,count in output.num_scheduled_tokens.items():
            state=self.reuse_requests.get(rid)
            if state:
                request=self._requests[rid]
                controls[rid]=dict(
                    start=request.num_computed_tokens,
                    end=request.num_computed_tokens+count,
                    R=state['R'],q=state['q'],consume=state['suffix_started'])
                start_block=controls[rid]['start']//self.gpu_block_size
                end_block=(controls[rid]['end']+self.gpu_block_size-1)//self.gpu_block_size
                trace('compute_destination_blocks',request_id=rid,
                      start=controls[rid]['start'],end=controls[rid]['end'],
                      blocks=state.get('destination',[])[start_block:end_block])
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
        if os.environ.get('RUNTIME_SCHED_TRACE')=='1':
            cs=self.connector.connector_scheduler if self.connector else None
            trace('scheduler_config_identity',scheduler_config_id=id(self.scheduler_config),
                  connector_config_id=id(getattr(cs,'scale_scheduler_config',None)),
                  scheduler_threshold=self.scheduler_config.long_prefill_token_threshold,
                  connector_threshold=getattr(getattr(cs,'scale_scheduler_config',None),
                                              'long_prefill_token_threshold',None),
                  parent_schedule_file=Scheduler.schedule.__code__.co_filename,
                  parent_schedule_line=Scheduler.schedule.__code__.co_firstlineno)

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
        if os.environ.get('RUNTIME_SCHED_TRACE')=='1':
            trace('scheduler_before_parent',scheduler_config_id=id(self.scheduler_config),
                  connector_config_id=id(getattr(cs,'scale_scheduler_config',None)),
                  scheduler_threshold=self.scheduler_config.long_prefill_token_threshold,
                  connector_threshold=getattr(getattr(cs,'scale_scheduler_config',None),
                                              'long_prefill_token_threshold',None),
                  waiting=len(self.waiting),running=len(self.running))
        if isinstance(cs,RuntimeStorageScheduler):
            # vLLM preempts from the end of the running list; keep requests
            # with an in-flight host load at the front (stable order otherwise).
            if isinstance(self.running,list):
                self.running.sort(key=lambda r:(not getattr(cs,"_reqs_being_loaded",{}).get(r.request_id), r.request_id in cs.reuse_requests))
            for request in self.running:
                state=cs.reuse_requests.get(request.request_id)
                if not state or state['suffix_started']:continue
                if request.num_computed_tokens>=state['q']:
                    progress=Progress(request.request_id)
                    target=progress.compute(CHUNK)
                    if target is not None:
                        state['q']=target
                        limit=self.scheduler_config.long_prefill_token_threshold
                        remaining=target-request.num_computed_tokens
                        self.scheduler_config.long_prefill_token_threshold=min(limit,remaining) if limit>0 else remaining
                    else:
                        state['suffix_started']=True
                        request.num_computed_tokens=state['R']
                        trace('cake_meet',request_id=request.request_id,compute=state['q'],R=state['R'],progress=progress.read())
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
            if os.environ.get('RUNTIME_SCHED_TRACE')=='1' and state:
                trace('allocate_for_tail',request_id=request.request_id,
                      num_new_tokens=num_new_tokens,
                      threshold=self.scheduler_config.long_prefill_token_threshold,
                      lookahead=kwargs.get('num_lookahead_tokens',0),
                      computed=request.num_computed_tokens)
            return allocate(request,num_new_tokens,*args,**kwargs)
        self.kv_cache_manager.allocate_slots=allocate_for_tail
        try:return super().schedule()
        finally:
            self.kv_cache_manager.allocate_slots=allocate
            self.scheduler_config.long_prefill_token_threshold=self.ordinary_prefill_threshold

class RuntimeWorker(NativeWorker):
    def __init__(self,spec):
        super().__init__(spec)
        # OffloadingConnectorWorker does not expose the engine rank.  Use the
        # vLLM parallel rank so each TP worker owns a distinct progress cursor
        # for its own KV shard.
        self.rank=int(getattr(spec.vllm_config.parallel_config,'rank',0))
        self.progress_world_size=int(getattr(
            spec.vllm_config.parallel_config,'world_size',1))
        self.load_jobs={}
        self.lock=threading.Condition()
        self.newly_done=set()
        self.consuming={}
        self.submission_lock=threading.Lock()

    def snapshot(self,state):
        return dict(q=state['q'],done=state['done'],error=state['error'],
                    verified=list(state['verified']))

    def fetch(self,req_id,state):
        # NativeTransfer and crypto prediction scope are shared by this worker.
        # Queue requests through one submitter while compute batches run concurrently.
        with self.submission_lock,self.codec_store.io_lock:
            self.fetch_serialized(req_id,state)

    def fetch_serialized(self,req_id,state):
        progress=Progress(req_id,rank=getattr(self,'rank',0))
        try:
            torch.npu.set_device(self.handler.npu_tensors[0][0].device)
            # The model runner zeroes newly allocated KV blocks on its
            # foreground stream before it starts the connector.  H2D uses
            # independent streams, so explicitly order those streams after
            # that producer work; otherwise the zero kernel can race a
            # reload into the same destination block.
            mode,_=settings()
            self.codec_adapter.mode=mode
            self.codec_adapter.request_id=req_id
            staged=self.codec_adapter.use_stages and mode!='none'
            producer_event=state.get('producer_event')
            if producer_event is not None:
                self.native.load_stream.wait_event(producer_event)
                for stream in self.ops.streams:
                    stream.wait_event(producer_event)
            while True:
                claim=progress.load(FETCH)
                if claim is None:break
                start,end=claim
                mapping=np.column_stack((state['source'][start//128:end//128],state['destination'][start//128:end//128])).astype(np.int64)
                tensor=torch.from_numpy(mapping)
                self.ops.retained.append(tensor)
                trace('cake_load_claim',request_id=req_id,start=start,end=end,
                      destination=state['destination'][start//128:end//128])
                if staged:
                    self.restore_compressed_claim(req_id,state,mode,start,end,tensor)
                    progress.ready(start,end)
                    trace('cake_chunk_ready',request_id=req_id,start=start,end=end)
                    continue
                def record_layer_event():
                    # NativeTransfer stripes one layer over several H2D
                    # streams.  Join those streams on the connector's load
                    # stream before publishing a layer-ready event.
                    with torch.npu.stream(self.native.load_stream):
                        for stream in self.ops.streams:
                            if stream==self.native.load_stream:continue
                            stream_event=torch.npu.Event()
                            stream_event.record(stream)
                            self.ops.retained.append(stream_event)
                            self.native.load_stream.wait_event(stream_event)
                        event=torch.npu.Event()
                        event.record(self.native.load_stream)
                        self.ops.retained.append(event)
                    return event

                layer_events=[]
                with torch.npu.stream(self.native.load_stream):
                    for layer in range(len(self.names)):
                        self.ops.layer=layer
                        self.ops.causal_group_blocks=FETCH//128
                        self.ops.single_layer_kv_transfer((self.handler.cpu_tensors[layer],tensor),self.native.kvcaches[layer])
                        log('load_submissions.jsonl',dict(request_id=req_id,layer=layer,start=start,end=end,records=2*len(mapping)))
                        layer_events.append(record_layer_event())
                for layer,layer_event in enumerate(layer_events):
                    layer_event.synchronize()
                    assert self.wait_auth(layer)==0,'Authentication failure blocks consumption'
                    with self.lock:
                        state['ready_start'][layer]=start
                        self.lock.notify_all()
                progress.ready(start,end)
                trace('cake_chunk_ready',request_id=req_id,start=start,end=end)
            _retire_h2d_prediction_scope()
            self.retain_copy_inputs()
        except BaseException as error:
            state['error']=repr(error)
            progress.error(repr(error))
            trace('fetch_error',request_id=req_id,error=repr(error))
        finally:
            with self.lock:
                state['done']=True
                state['verified']=[0 for _ in self.names]
                self.newly_done.add(req_id)
                trace('fetch_done',request_id=req_id,**self.snapshot(state))
                self.lock.notify_all()

    def restore_compressed_claim(self,req_id,state,mode,start,end,mapping):
        # Preserve Cake's tail claim and completion boundaries. Only replace
        # the transport within a claimed range by the common 4-bit pipeline.
        adapter=self.codec_adapter
        layer_events=[]
        def submitted(layer,event,_):
            assert layer==len(layer_events)
            layer_events.append(event)
        def failed(message):
            with self.lock:
                state['error']=message
                self.lock.notify_all()
        adapter.start_request(req_id,mode,len(self.names),submitted,failed)
        try:
            with torch.npu.stream(self.native.load_stream):
                for layer in range(len(self.names)):
                    self.ops.layer=layer
                    start_event=torch.npu.Event(enable_timing=True)
                    start_event.record(self.native.load_stream)
                    adapter.load_start=start_event
                    self.ops.single_layer_kv_transfer((self.handler.cpu_tensors[layer],mapping),
                                                      self.native.kvcaches[layer])
                    log('load_submissions.jsonl',dict(request_id=req_id,layer=layer,start=start,
                                                      end=end,records=2*len(mapping)))
            adapter.finish_request()
            assert len(layer_events)==len(self.names)
            for layer,event in enumerate(layer_events):
                event.synchronize()
                with self.lock:
                    state['ready_start'][layer]=start
                    self.lock.notify_all()
            adapter.flush()
            trace('cake_compressed_claim',request_id=req_id,start=start,end=end,codec=mode)
        except BaseException:
            adapter.abort_request()
            self.native.load_stream.synchronize()
            raise

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
            Progress(req_id).create(0,0,prefix,self.progress_world_size)
            controls[req_id]=dict(start=prefix,end=prefix,R=prefix,q=0,consume=True)
        metadata=replace(metadata,reqs_to_load={})
        # Retain ordinary asynchronous stores; all reloads use the background submitter.
        super().start_kv_transfers(metadata)
        producer_event=None
        if kickoff:
            producer_event=torch.npu.Event()
            producer_event.record(torch.npu.current_stream())
        for req_id,kick in kickoff.items():
            assert req_id not in self.load_jobs
            state=dict(**kick,done=False,error=None,verified=[kick['R'] for _ in self.names],
                events=deque(),submissions_done=False,producer_event=producer_event,
                ready_start=[kick['R']]*len(self.names))
            self.load_jobs[req_id]=state
            state['thread']=threading.Thread(target=self.fetch,args=(req_id,state),daemon=True)
            state['thread'].start()
            trace('fetch_started',request_id=req_id,R=kick['R'],q=kick['q'])
        for req_id,control in controls.items():
            state=self.load_jobs.get(req_id)
            if state:
                # Ownership is coordinated by Progress across scheduler/worker.
                state['q']=control['q']
                if control['consume']:
                    self.consuming[req_id]=state
                    trace('tail_consume_begin',request_id=req_id,q=state['q'])

    def wait_for_layer_load(self,name):
        layer=self.names.index(name)
        # Recomputation owns the head and never reads the reloaded tail.
        # Suffix attention needs every tail chunk, not merely the first one
        # to complete on this layer. Track the descending ready frontier.
        for state in list(self.load_jobs.values()):
            assert not state['error'],state['error']
        for req_id,state in list(self.consuming.items()):
            begin=time.monotonic_ns()
            with self.lock:
                while state['ready_start'][layer]>state['q'] and not state['done'] and not state['error']:
                    self.lock.wait(timeout=30)
                assert not state['error'],state['error']
                assert state['ready_start'][layer]<=state['q'], 'Cake tail is incomplete'
            trace('native_layer_wait_end',request_id=req_id,layer=layer,
                  wait_ns=time.monotonic_ns()-begin,authenticated=True,written=True)
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
