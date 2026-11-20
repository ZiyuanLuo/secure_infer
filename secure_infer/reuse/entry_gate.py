"""Entry decisions from this arm's previously completed device observations.

No profile is loaded. Coefficients describe effective stream intervals with
explicit layer waits subtracted; they are not isolated kernel measurements.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import nnls

_model_cache = None
_model_key = None
_compute = []
_loads = []
_timeline_offset = 0
_load_offset = 0
_rank_timelines = {}


def active_config():
    path=os.environ.get('KV_CODEC_CONFIG','')
    cfg=json.loads(Path(path).read_text())if path else{}
    return cfg.get('codec',os.environ.get('KV_CODEC','none')),bool(cfg.get('pipeline',int(os.environ.get('KV_RECOVERY_PIPELINE','1'))))


def root():
    return Path(os.environ['RUNTIME_RESULT_DIR'])


def _blocked_ms(row, step):
    # Layer-wait time recorded inside one compute step.
    return sum(g['stream_span_ms'] for g in row['gates']
               if step['start_ms'] <= g['start_ms'] and
               g['end_ms'] <= step['end_ms'])


def retain_compute_samples(samples, limit=256, per_shape=4):
    counts = {}
    kept = []
    for sample in reversed(samples):
        shape = (sample['tokens'],sample['position'],sample.get('gated',False))
        count = counts.get(shape,0)
        if count < per_shape:
            kept.append(sample)
            counts[shape] = count+1
            if len(kept)==limit:
                break
    return list(reversed(kept))


def update():
    global _load_offset
    loads_path=root()/'controller_load_costs.jsonl'
    if loads_path.exists():
        with loads_path.open() as source:
            source.seek(_load_offset)
            _loads.extend(json.loads(line) for line in source.readlines())
            _load_offset=source.tell()
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
            gated = any(step['start_ms'] <= g['start_ms'] and g['end_ms'] <= step['end_ms'] for g in row['gates'])
            blocked = (row['blocked_ms'][index] if 'blocked_ms' in row
                       else _blocked_ms(row, step))
            duration = max(0., step['stream_span_ms']-blocked)
            if duration > 0:
                sample = dict(request_id=rid, tokens=step['tokens'],
                              position=step['position'], service_ms=duration,
                              raw_ms=step['stream_span_ms'], gate_ms=blocked, gated=gated)
                # Retain recent observations rather than replacing a shape
                # with its latest measurement from a different execution state.
                _compute.append(sample)
    _compute[:] = retain_compute_samples(_compute)
    del _loads[:-128]


def load_profile(rows):
    # Equal weight per observed size. Merge neighboring groups whose medians
    # are out of order so a noisier sample cannot reward loading more data.
    samples = {}
    for row in rows:
        if row['tokens'] <= 0:
            continue
        samples.setdefault(row['tokens'], []).append(row['service_ms'])
    sizes = sorted(samples)
    groups = []
    for n in sizes:
        groups.append((1, float(np.median(samples[n]))))
        while len(groups) > 1 and groups[-2][1] > groups[-1][1]:
            count, value = groups.pop()
            previous_count, previous_value = groups.pop()
            total = count + previous_count
            groups.append((total, (count*value + previous_count*previous_value)/total))
    times = [value for count,value in groups for _ in range(count)]
    return list(zip(sizes,times))


def forward_profile(samples):
    observations = {}
    for sample in samples:
        shape = (sample['tokens'], sample['position'], bool(sample.get('gated')))
        observations.setdefault(shape, []).append(sample['service_ms'])
    ungated = [(n,x) for n,x,gated in observations if not gated]
    if len(ungated) < 3:
        return None
    core = np.asarray([[1.,n/1024,(n/1024)*(x+n/2)/1024] for n,x in ungated])
    if np.linalg.matrix_rank(core) < 3:
        return None
    times = np.asarray([np.median(observations[n,x,False]) for n,x in ungated])
    coefficients = nnls(core,times)[0]
    weight = 1/np.sqrt(np.maximum(core @ coefficients,1.))
    step,a,b = nnls(core*weight[:,None],times*weight)[0]
    # Gate handling can add a fixed cost even after explicit waits are
    # removed. Estimate it separately from the intrinsic compute curve.
    residuals = [np.median(values)-(step+a*n/1024+b*(n/1024)*(x+n/2)/1024)
                 for (n,x,gated),values in observations.items() if gated]
    gated = max(0.,float(np.median(residuals))) if residuals else 0.
    return step,a,b,gated


def costs():
    global _model_cache, _model_key
    codec,pipeline=active_config()
    staged = codec != 'none' and bool(int(os.environ.get('KV_STAGE_PIPELINE','1')))
    eligible = [r for r in _loads if r['codec']==codec and r['pipeline']==pipeline
                and r.get('staged_recovery',staged)==staged]
    warm_loads = [r for r in eligible if not r.get('startup')]
    selected_loads = warm_loads if len({r['tokens'] for r in warm_loads}) >= 2 else eligible
    key=(codec,pipeline,staged,tuple(id(x) for x in _compute),tuple(id(x) for x in eligible))
    if key == _model_key:
        return _model_cache
    if not _compute or not selected_loads:
        return None
    startup_ids = {r['request_id'] for r in eligible if r.get('startup')}
    compute = [s for s in _compute if s['request_id'] not in startup_ids]
    forward = forward_profile(compute)
    points = load_profile(selected_loads)
    if forward is None or not points:
        return None
    step,a,b,gated = map(float,forward)
    layers = selected_loads[-1].get('layers')
    profiles = [np.asarray(s['ready_ms'],dtype=float)/s['service_ms'] for s in selected_loads
                if s.get('ready_ms') and len(s['ready_ms'])==layers and s['service_ms']>0]
    # Logs without completed layer events can still seed the load curve.
    layer_ready = np.median(profiles,axis=0).tolist() if profiles else []
    model=dict(step_ms=step,a=a,b=b,gated_step_ms=gated,load_points=points,layer_ready=layer_ready,
               startup_load_samples=len(startup_ids),compute_samples=len(compute),load_samples=len(selected_loads),
               training_request_ids=sorted({s['request_id'] for s in compute+selected_loads}))
    _model_key=key
    _model_cache=model
    return model


def load_ms(model, n):
    values = np.asarray(n, dtype=float)
    if 'load_points' not in model:
        # Historical experiment snapshots use the previous affine curve.
        result = model['phi'] + model['lam'] * values / 1024
    else:
        sizes,times = map(np.asarray,zip(*model['load_points']))
        slope = (times[-1]-times[0])/(sizes[-1]-sizes[0]) if len(sizes)>1 else times[0]/sizes[0]
        result = np.interp(values,sizes,times)
        result += slope*(np.minimum(0.,values-sizes[0])+np.maximum(0.,values-sizes[-1]))
    result = np.where(values>0,np.maximum(0.,result),0.)
    return result.item() if result.ndim == 0 else result


def compute_ms(model, n, context=0, step_tokens=None, with_reload=False):
    steps = int(np.ceil(n / step_tokens)) if n > 0 and step_tokens else int(n > 0)
    extra = model.get('gated_step_ms',0.) if with_reload and n>0 else 0.
    return extra+steps*model.get('step_ms',0.)+model['a']*n/1024 + model['b']*(n/1024)*(context+n/2)/1024


def head_ms(model, n, chunk, base=0):
    # Recomputing n tokens that start at context position base.
    values = np.asarray(n, dtype=float)
    result = (model.get('step_ms', 0.) * np.ceil(values / chunk) +
              model['a'] * values / 1024 +
              model['b'] * (values / 1024) ** 2 / 2 +
              model['b'] * (values / 1024) * base / 1024)
    return result.item() if result.ndim == 0 else result


def finish_ms(model,r,u,chunk,layers,base,pending,step_tokens,pipeline):
    qs = np.array(sorted(set(range(0,r,128)) | {r}),dtype=np.int64)
    wait = model['a']*pending/1024
    first_tokens = min(u,step_tokens) if step_tokens else u
    first = compute_ms(model,first_tokens,base+r,with_reload=True)
    remaining = compute_ms(model,u-first_tokens,base+r+first_tokens,step_tokens=step_tokens)
    head = head_ms(model,qs,min(chunk,step_tokens) if step_tokens else chunk,base)
    service = load_ms(model,r-qs)
    begin = wait+head
    if pipeline and u:
        progress = model.get('layer_ready',[])
        if len(progress)!=layers:
            progress = np.arange(1,layers+1)/layers
        for fraction in progress:
            begin = np.maximum(begin,service*fraction)+first/layers
        finish = begin+remaining
    else:
        finish = np.maximum(begin,service)+first+remaining
    finish[-1] = wait+compute_ms(model,r+u,base,step_tokens=step_tokens)
    return qs,finish,service,first+remaining,wait

def choose(r, u, chunk, layers, base=0, pending=0, step_tokens=None):
    # r host-resident prefix tokens start at position base; the first base
    # tokens are already in device HBM.
    begin = time.monotonic_ns()
    force_q = os.environ.get('RUNTIME_FORCE_Q', '')
    probe_path = os.environ.get('RUNTIME_PROBE_Q_FILE', '')
    if probe_path and Path(probe_path).exists():
        force_q = str(json.loads(Path(probe_path).read_text())['q'])
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
    recovery_pipeline=active_config()[1]
    qs,finish,service,suffix,wait = finish_ms(model,r,u,chunk,layers,base,pending,step_tokens,recovery_pipeline)
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
                predicted_native_recovery_ms=float(service[0]),
                predicted_best_ms=float(finish[best]), predicted_gain_ms=gain,
                predicted_full_recompute_ms=float(finish[-1]),
                suffix_ms=suffix, pending_tokens=pending,wait_ms=wait,model=model, offline_profile_used=False,
                forward_token_budget=step_tokens,recovery_pipeline=recovery_pipeline,
                codec=active_config()[0],
                suffix_forward_steps=int(np.ceil(u/step_tokens)) if step_tokens else int(u>0),
                full_recompute_forward_steps=int(np.ceil((r+u)/step_tokens)) if step_tokens else int(r+u>0),
                reason='minimum estimated TTFT before prefill',
                decision_ns=time.monotonic_ns()-begin)
