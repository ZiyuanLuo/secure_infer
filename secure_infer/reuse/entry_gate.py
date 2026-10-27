"""Entry decisions from this arm's previously completed device observations.

No profile is loaded. Coefficients describe effective stream intervals with
explicit layer waits subtracted; they are not isolated kernel measurements.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from .completion_model import pipeline_finish
from scipy.optimize import nnls

_model_cache = None
_model_key = None
_compute = []
_loads = []
_timeline_offset = 0


def root():
    return Path(os.environ['RUNTIME_RESULT_DIR'])


def update():
    path = root()/'device_timeline.jsonl'
    if not path.exists():
        return
    global _timeline_offset
    with path.open() as source:
        source.seek(_timeline_offset)
        lines=source.readlines()
        _timeline_offset=source.tell()
    for line in lines:
        row = json.loads(line)
        rid = row['request_id']
        # flush() emits this trace only after every marker and the complete
        # prefill have physically finished. Only completed own observations train.
        observations = []
        for step in row['steps']:
            gates = [g for g in row['gates']
                     if step['start_ms'] <= g['start_ms'] and
                     g['end_ms'] <= step['end_ms']]
            blocked = sum(g['stream_span_ms'] for g in gates)
            duration = max(0., step['stream_span_ms']-blocked)
            if duration > 0:
                sample = dict(request_id=rid, tokens=step['tokens'],
                              position=step['position'], service_ms=duration,
                              raw_ms=step['stream_span_ms'], gate_ms=blocked)
                _compute.append(sample)
                observations.append(dict(kind='compute', **sample))
        # Each copy marker encloses the physical layer service. Sum active
        # intervals, excluding idle gaps caused by native consumption callbacks.
        # Only complete layer groups are eligible for the load fit.
        by_layer={i:[] for i in range(row['layers'])}
        for load in row['loads']:
            by_layer[load['layer']].append(load)
        counts=[sum(x['blocks'] for x in by_layer[i]) for i in range(row['layers'])]
        if counts[0]>0 and len(set(counts))==1:
            sample=dict(request_id=rid,tokens=counts[0]*128,layers=row['layers'],
                        service_ms=sum(x['stream_span_ms'] for x in row['loads']))
            _loads.append(sample)
            observations.append(dict(kind='load',**sample))
        with (root()/'entry_training.jsonl').open('a') as out:
            for sample in observations:
                out.write(json.dumps(dict(observed_ns=time.monotonic_ns(), **sample))+'\n')
    del _compute[:-64]
    del _loads[:-32]


def costs():
    global _model_cache, _model_key
    key=(tuple(id(x) for x in _compute),tuple(id(x) for x in _loads))
    if key == _model_key:
        return _model_cache
    if not _compute or not _loads:
        return None
    x = np.array([[1., s['tokens']/1024,
                   s['tokens']/1024*(s['position']+s['tokens']/2)/1024]
                  for s in _compute])
    if np.linalg.matrix_rank(x) < 3:
        return None
    y = np.array([s['service_ms'] for s in _compute])
    step, a, b = nnls(x, y)[0]
    load_x = np.array([[1., s['tokens']/1024] for s in _loads])
    load_y = np.array([s['service_ms'] for s in _loads])
    if len({s['tokens'] for s in _loads}) > 1:
        phi, lam = nnls(load_x, load_y)[0]
    else:
        # A single size cannot identify an independent startup cost.
        phi = 0.
        lam = float(np.median(load_y/load_x[:, 1]))
    sizes=sorted({s['tokens'] for s in _loads})
    values=np.maximum.accumulate([float(np.median([s['service_ms'] for s in _loads if s['tokens']==n])) for n in sizes])
    curve=list(zip(sizes,values.tolist()))
    model=dict(step_ms=float(step),a=float(a),b=float(b),phi=float(phi),lam=float(lam),
               load_curve=curve,compute_samples=len(_compute),load_samples=len(_loads),
               training_request_ids=sorted({s['request_id'] for s in _compute+_loads}))
    _model_key=key
    _model_cache=model
    return model


def load_ms(model,n):
    if n<=0:return 0.
    curve=model.get('load_curve')
    if not curve:return model['phi']+model['lam']*n/1024
    xs=[0.]+[float(x) for x,_ in curve]
    ys=[0.]+[float(y) for _,y in curve]
    if n<=xs[-1]:return float(np.interp(n,xs,ys))
    # Do not extrapolate a noisy terminal plateau as free additional bytes.
    slope=max(0.,(ys[-1]-ys[-2])/(xs[-1]-xs[-2]),model.get('lam',0.)/1024)
    return ys[-1]+slope*(n-xs[-1])


def compute_ms(model, n, context=0):
    return (model.get('step_ms',0.) if n>0 else 0.)+model['a']*n/1024 + model['b']*(n/1024)*(context+n/2)/1024


def head_ms(model,n,chunk):
    return (model.get('step_ms',0.)*int(np.ceil(n/chunk))+
            model['a']*n/1024+model['b']*(n/1024)**2/2)

def choose(r, u, chunk, layers):
    begin = time.monotonic_ns()
    update()
    model = costs()
    if model is None:
        return dict(policy='load', entry_mode='native', q=0,
                    reason='cold native load; no identifiable online model',
                    offline_profile_used=False, model=None,
                    decision_ns=time.monotonic_ns()-begin)
    qs = np.array(sorted(set(range(0,r,128)) | {r}), dtype=np.int64)
    suffix = compute_ms(model,u,r)
    head = model.get('step_ms',0.)*np.ceil(qs/chunk)+model['a']*qs/1024+model['b']*(qs/1024)**2/2
    ns=r-qs
    curve=model.get('load_curve')
    if curve:
        xs=np.array([0.]+[float(x) for x,_ in curve]);ys=np.array([0.]+[float(y) for _,y in curve])
        service=np.interp(ns,xs,ys)
        slope=max(0.,(ys[-1]-ys[-2])/(xs[-1]-xs[-2]),model.get('lam',0.)/1024)
        service=np.where(ns>xs[-1],ys[-1]+slope*(ns-xs[-1]),service)
    else:
        service=np.where(ns>0,model['phi']+model['lam']*ns/1024,0.)
    # Load observations include their setup cost. Equal layer service/compute
    # remains an approximation; the complete-recompute endpoint has no load.
    pipeline = pipeline_finish(suffix,service,layers)
    finish = np.maximum(head+suffix,pipeline)
    finish[-1] = compute_ms(model,r+u)
    best = int(np.argmin(finish))
    best_q = int(qs[best])
    gain = float(finish[0]-finish[best])
    if best_q == 0:
        policy,mode,q = 'load','native',0
    elif best_q == r:
        policy,mode,q = 'compute','compute',r
    else:
        # Keep the existing live pending-write/coverage controller. Predicted
        # q is not forced as an oracle split; it only admits the hybrid path.
        policy,mode,q = 'mixed','mixed',None
    return dict(policy=policy, entry_mode=mode, q=q,
                predicted_q=best_q, predicted_native_ms=float(finish[0]),
                predicted_best_ms=float(finish[best]), predicted_gain_ms=gain,
                suffix_ms=suffix, model=model, offline_profile_used=False,
                reason='online entry three-way comparison',
                decision_ns=time.monotonic_ns()-begin)
