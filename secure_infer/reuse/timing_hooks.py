"""Observation-only NPU stream markers, collected after forward completion."""
import time
import re
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
        row=dict(request_id=trace['request_id'], layers=trace['layers'])
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
    from secure_infer.kv_offload.native_layerwise import NativeTransfer, NativeWorker
    from .connector import RuntimeWorker
    old_execute=NPUModelRunner.execute_model
    def execute(self,scheduler_output,*args,**kwargs):
        global _current
        flush()
        measured=[rid for rid,n in scheduler_output.num_scheduled_tokens.items()
            if n>1]
        if not measured:return old_execute(self,scheduler_output,*args,**kwargs)
        assert len(measured)==1
        rid=measured[0]
        trace=_traces.get(rid)
        if trace is None:
            trace=dict(request_id=rid,layers=worker.native.num_layers,anchor=marker(),steps=[],loads=[],gates=[],
                suffix_complete=False,written=False)
            _traces[rid]=trace
        _current=trace
        before=marker();host_start=time.monotonic_ns()
        result=old_execute(self,scheduler_output,*args,**kwargs)
        after=marker();host_end=time.monotonic_ns()
        state=self.requests[rid]
        count=scheduler_output.num_scheduled_tokens[rid]
        position=state.num_computed_tokens
        trace['steps'].append(dict(start=before,end=after,position=position,tokens=count,
            prompt_tokens=state.num_prompt_tokens,host_call_ms=(host_end-host_start)/1e6,
            phase='prefill'))
        if position+count>=state.num_prompt_tokens:trace['suffix_complete']=True
        return result
    NPUModelRunner.execute_model=execute
    old_copy=NativeTransfer.single_layer_kv_transfer
    def copy(self,payload,*args,**kwargs):
        trace=_current
        if getattr(self,'_timing_copy_active',False) or trace is None or trace['written']:
            return old_copy(self,payload,*args,**kwargs)
        before=marker();host_start=time.monotonic_ns()
        layer=self.layer%worker.native.num_layers
        self._timing_copy_active=True
        try:result=old_copy(self,payload,*args,**kwargs)
        finally:self._timing_copy_active=False
        after=marker()
        trace['loads'].append(dict(start=before,end=after,layer=layer,
            blocks=len(payload[1]),host_submit_ms=(time.monotonic_ns()-host_start)/1e6))
        return result
    NativeTransfer.single_layer_kv_transfer=copy
    def wrap_wait(cls,old):
        def wait(self,name):
            trace=_current
            active=self.active is not None if cls is NativeWorker else self.cake_consuming is not None
            if trace is None or trace['written'] or not active:return old(self,name)
            before=marker();host_start=time.monotonic_ns()
            result=old(self,name)
            after=marker()
            trace['gates'].append(dict(start=before,end=after,layer=self.names.index(name),
                host_callback_ms=(time.monotonic_ns()-host_start)/1e6))
            return result
        return wait
    NativeWorker.wait_for_layer_load=wrap_wait(NativeWorker,NativeWorker.wait_for_layer_load)
    RuntimeWorker.wait_for_layer_load=wrap_wait(RuntimeWorker,RuntimeWorker.wait_for_layer_load)
