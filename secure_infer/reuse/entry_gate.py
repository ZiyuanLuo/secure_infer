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
_rank_timelines = {}


def root():
    return Path(os.environ['RUNTIME_RESULT_DIR'])


def _blocked_ms(row, step):
    # Layer-wait time recorded inside one compute step.
    return sum(g['stream_span_ms'] for g in row['gates']
               if step['start_ms'] <= g['start_ms'] and
               g['end_ms'] <= step['end_ms'])


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
        world = row.get('world_size', 1)
        if world > 1:
            ranks = _rank_timelines.setdefault(rid, {})
            ranks[row['rank']] = row
            if len(ranks) < world:
                continue
            peers = list(_rank_timelines.pop(rid).values())
            # The slowest TP rank determines completion, rather than the
            # sum or average of rank-local observations.
            row = dict(max(peers, key=lambda r:
                sum(x['stream_span_ms'] for x in r['steps'])))
            row['loads'] = max(peers, key=lambda r:
                sum(x['stream_span_ms'] for x in r['loads']))['loads']
            # Layer waits are recorded by the rank whose host callback met them,
            # which differs between requests, while collectives tie the step
            # spans of all ranks together.  Subtracting only the waits of the
            # slowest-looking rank leaves the wait inside the compute sample
            # whenever another rank observed it, so each step is charged the
            # longest wait any rank recorded for it.
            if all(len(p['steps']) == len(row['steps']) for p in peers):
                row['blocked_ms'] = [
                    max(_blocked_ms(p, p['steps'][i]) for p in peers)
                    for i in range(len(row['steps']))]
        # flush() emits this trace only after every marker and the complete
        # prefill have physically finished. Only completed own observations train.
        for index, step in enumerate(row['steps']):
            # A standalone short tail exposes input/launch startup differently
            # from a chunk in a continuous prefill. Keep full-pass observations
            # and all steps from multi-chunk requests in the calibration window.
            if len(row['steps']) == 1 and step['tokens'] < int(os.environ.get('RUNTIME_COMPUTE_CHUNK','1024')):
                continue
            blocked = (row['blocked_ms'][index] if 'blocked_ms' in row
                       else _blocked_ms(row, step))
            duration = max(0., step['stream_span_ms']-blocked)
            if duration > 0:
                sample = dict(request_id=rid, tokens=step['tokens'],
                              position=step['position'], service_ms=duration,
                              raw_ms=step['stream_span_ms'], gate_ms=blocked)
                # Retain recent observations rather than replacing a shape
                # with its latest measurement from a different execution state.
                _compute.append(sample)
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
    model=dict(step_ms=float(step),a=float(a),b=float(b),phi=float(phi),lam=float(lam),
               compute_samples=len(_compute),load_samples=len(_loads),
               training_request_ids=sorted({s['request_id'] for s in _compute+_loads}),
)
    _model_key=key
    _model_cache=model
    return model


def load_ms(model, n):
    values = np.asarray(n, dtype=float)
    result = np.where(values > 0,
                      model['phi'] + model['lam'] * values / 1024,
                      0.)
    return result.item() if result.ndim == 0 else result


def compute_ms(model, n, context=0):
    return (model.get('step_ms',0.) if n>0 else 0.)+model['a']*n/1024 + model['b']*(n/1024)*(context+n/2)/1024


def head_ms(model, n, chunk):
    values = np.asarray(n, dtype=float)
    result = (model.get('step_ms', 0.) * np.ceil(values / chunk) +
              model['a'] * values / 1024 +
              model['b'] * (values / 1024) ** 2 / 2)
    return result.item() if result.ndim == 0 else result

def choose(r, u, chunk, layers):
    begin = time.monotonic_ns()
    force_q = os.environ.get('RUNTIME_FORCE_Q', '')
    if force_q != '':
        fq = int(force_q)
        assert 0 <= fq <= r and fq % 128 == 0, (fq, r)
        if fq == 0:
            policy, mode, q = 'load', 'native', 0
        elif fq == r:
            policy, mode, q = 'compute', 'compute', r
        else:
            policy, mode, q = 'hybrid', 'hybrid', fq
        return dict(policy=policy, entry_mode=mode, q=q,
                    predicted_q=q, predicted_native_ms=0.0,
                    predicted_best_ms=0.0, predicted_gain_ms=0.0,
                    suffix_ms=0.0, model=None, offline_profile_used=False,
                    reason='forced by RUNTIME_FORCE_Q',
                    decision_ns=time.monotonic_ns()-begin)
    update()
    model = costs()
    if model is None:
        return dict(policy='load', entry_mode='native', q=0,
                    reason='cold native load; no identifiable online model',
                    offline_profile_used=False, model=None,
                    decision_ns=time.monotonic_ns()-begin)
    qs = np.array(sorted(set(range(0,r,128)) | {r}), dtype=np.int64)
    suffix = compute_ms(model,u,r)
    head = head_ms(model, qs, chunk)
    ns=r-qs
    service = load_ms(model, ns)
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
        policy,mode,q = 'hybrid','hybrid',best_q
    return dict(policy=policy, entry_mode=mode, q=q,
                predicted_q=best_q, predicted_native_ms=float(finish[0]),
                predicted_best_ms=float(finish[best]), predicted_gain_ms=gain,
                suffix_ms=suffix, model=model, offline_profile_used=False,
                reason='minimum estimated TTFT before prefill',
                decision_ns=time.monotonic_ns()-begin)
