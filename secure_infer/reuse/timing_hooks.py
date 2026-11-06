"""Observation-only NPU stream markers, collected after forward completion."""
import time
import torch

_installed=False
_traces={}
_current=None

def marker():
    event=torch.npu.Event(enable_timing=True)
    event.record(torch.npu.current_stream())
    return event

def flush():
    from secure_infer.kv_offload.native_layerwise import log
    for trace in _traces.values():
        if trace['written'] or not trace['suffix_complete']:
            continue
        if not all(x['end'].query() for name in ['steps','loads','gates'] for x in trace[name]):
            continue
        anchor=trace['anchor']
        row=dict(request_id=trace['request_id'], layers=trace['layers'], timing_window='model_forward',
                 rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
        for name in ['steps','loads','gates']:
            row[name]=[dict(**{k:v for k,v in x.items() if k not in ['start','end']},
                start_ms=anchor.elapsed_time(x['start']),end_ms=anchor.elapsed_time(x['end']),
                stream_span_ms=x['start'].elapsed_time(x['end'])) for x in trace[name]]
        log('device_timeline.jsonl',row)
        trace['written']=True
    for rid in [rid for rid, trace in _traces.items() if trace['written']]:
        del _traces[rid]

def install(worker):
    global _installed
    if _installed:return
    _installed=True
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from secure_infer.kv_offload.native_layerwise import NativeTransfer
    from .connector import RuntimeWorker
    old_execute=NPUModelRunner.execute_model
    old_forward=NPUModelRunner._model_forward
    def execute(self,scheduler_output,*args,**kwargs):
        global _current
        flush()
        positions={r.req_id:r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
        cached=scheduler_output.scheduled_cached_reqs
        positions.update(zip(cached.req_ids,cached.num_computed_tokens))
        new_lengths={r.req_id:len(r.prompt_token_ids) for r in scheduler_output.scheduled_new_reqs}
        measured=[rid for rid,n in scheduler_output.num_scheduled_tokens.items()
            if n>0 and positions[rid] < (new_lengths[rid] if rid in new_lengths else self.requests[rid].num_prompt_tokens)]
        if len(measured)!=1:
            # Two prefills in one step are not an individual compute-model
            # observation.  Co-scheduled decode tokens are allowed; their small
            # cost is charged to this step and recorded as co_decode.
            for rid in measured:_traces.pop(rid,None)
            _current=None
            self._paci_timing_step=None
            return old_execute(self,scheduler_output,*args,**kwargs)
        rid=measured[0]
        trace=_traces.get(rid)
        if trace is None:
            trace=dict(request_id=rid,layers=worker.native.num_layers,anchor=marker(),steps=[],loads=[],gates=[],
                suffix_complete=False,written=False)
            _traces[rid]=trace
        _current=trace
        count=scheduler_output.num_scheduled_tokens[rid]
        position=positions[rid]
        self._paci_timing_step=dict(trace=trace,position=position,tokens=count,
            co_decode=len(scheduler_output.num_scheduled_tokens)-1,
            prompt_tokens=(new_lengths[rid] if rid in new_lengths else self.requests[rid].num_prompt_tokens))
        try:
            result=old_execute(self,scheduler_output,*args,**kwargs)
        finally:
            self._paci_timing_step=None
        state=self.requests[rid]
        if position+count>=state.num_prompt_tokens:trace['suffix_complete']=True
        return result
    def forward(self,*args,**kwargs):
        step=getattr(self,'_paci_timing_step',None)
        if step is None:
            return old_forward(self,*args,**kwargs)
        before=marker();host_start=time.monotonic_ns()
        result=old_forward(self,*args,**kwargs)
        after=marker()
        step['trace']['steps'].append(dict(start=before,end=after,
            position=step['position'],tokens=step['tokens'],
            prompt_tokens=step['prompt_tokens'],co_decode=step['co_decode'],
            host_call_ms=(time.monotonic_ns()-host_start)/1e6,phase='prefill'))
        return result
    NPUModelRunner.execute_model=execute
    NPUModelRunner._model_forward=forward
    old_copy=NativeTransfer.single_layer_kv_transfer
    def copy(self,payload,destination):
        trace=_current
        if getattr(self,'_timing_copy_active',False) or trace is None or trace['written']:
            return old_copy(self,payload,destination)
        before=marker();host_start=time.monotonic_ns()
        layer=self.layer%worker.native.num_layers
        self._timing_copy_active=True
        try:result=old_copy(self,payload,destination)
        finally:self._timing_copy_active=False
        after=marker()
        trace['loads'].append(dict(start=before,end=after,layer=layer,
            blocks=len(payload[1]),host_submit_ms=(time.monotonic_ns()-host_start)/1e6))
        return result
    NativeTransfer.single_layer_kv_transfer=copy
    def wrap_wait(old):
        def wait(self,name):
            trace=_current
            if trace is None or trace['written'] or not self.consuming:
                return old(self,name)
            before=marker();host_start=time.monotonic_ns()
            result=old(self,name)
            after=marker()
            trace['gates'].append(dict(start=before,end=after,layer=self.names.index(name),
                host_callback_ms=(time.monotonic_ns()-host_start)/1e6))
            return result
        return wait
    RuntimeWorker.wait_for_layer_load=wrap_wait(RuntimeWorker.wait_for_layer_load)
