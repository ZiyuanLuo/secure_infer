"""Suffix-aware stopping from this request's measured progress; no profile file."""


from .completion_model import pipeline_finish

def decide(*, q, r, u, observed, now_ns, step_ms, step_tokens, layers,
           layer_after_stop=True, compare_next=True, suffix_ms_override=None,
           next_compute_ms_override=None, error_margin_ms=0., queued_compute_ms=0.):
    result = dict(stop=False, reason='insufficient current-request observations',
                  q=q, R=r, U=u, offline_profile_used=False)
    if step_tokens <= 0 or step_ms <= 0:
        return result
    # This is effective engine-step time, not an isolated compute kernel time.
    per_token = step_ms / step_tokens
    suffix_ms = per_token * u if suffix_ms_override is None else suffix_ms_override
    service_ms = observed.get('last_service_ms')
    service_tokens = observed.get('last_load_tokens')
    pending = observed.get('pending')
    offsets = observed.get('ready_offsets', [])
    ready_layers = observed.get('layer_ready', 0)
    if service_ms and service_tokens:
        load_per_token = service_ms / service_tokens
        source = 'completed current-request load'
    elif pending and ready_layers and len(offsets) >= ready_layers:
        # Cold-start extrapolation from authentication/write-ready layers.
        service_tokens = pending[1] - pending[0]
        service_ms = offsets[ready_layers - 1] * layers / ready_layers
        load_per_token = service_ms / service_tokens
        source = 'partial current-request layer delivery'
    else:
        return result
    if pending:
        age = max(0., (now_ns - observed['claim_ns']) / 1e6)
        pending_ms = max(0., load_per_token * (pending[1]-pending[0]) - age)
        gap = max(0, pending[0] - q)
    else:
        pending_ms = 0.
        gap = max(0, observed['ready'] - q)
    remaining_ms = pending_ms + load_per_token * gap
    credit_ms = suffix_ms * (1 - 1 / layers)
    stop_finish = max(queued_compute_ms+suffix_ms,
                      float(pipeline_finish(suffix_ms,remaining_ms,layers)))
    # Next nonpreemptible compute step takes time; it removes that many
    # tokens from the unsubmitted gap. The already committed load is retained.
    next_tokens = min(step_tokens, max(0, r-q))
    next_compute = (per_token * next_tokens if next_compute_ms_override is None
                    else next_compute_ms_override)
    compute_start=queued_compute_ms
    if pending and pending[0] < q+next_tokens and pending[1] > q:
        # The next head step overwrites part of an already committed load;
        # preserve the existing write-before-overwrite dependency.
        compute_start=max(compute_start,pending_ms)
    more_finish = max(compute_start + next_compute + suffix_ms,
                      float(pipeline_finish(suffix_ms,
                          pending_ms+load_per_token*max(0,gap-next_tokens),layers)))
    stop = stop_finish-more_finish <= error_margin_ms if compare_next else remaining_ms <= credit_ms
    result.update(stop=stop,
                  reason='online STOP/MORE comparison' if compare_next else
                         'suffix credit versus remaining secure load',
                  compute_per_token_ms=per_token, load_per_token_ms=load_per_token,
                  suffix_ms=suffix_ms, credit_ms=credit_ms,queued_compute_ms=queued_compute_ms,
                  remaining_ms=remaining_ms, pending_remaining_ms=pending_ms,
                  uncovered_tokens=gap, observation_source=source,
                  layer_after_stop=layer_after_stop,
                  stop_finish_ms=stop_finish, more_finish_ms=more_finish, error_margin_ms=error_margin_ms)
    result['history_suffix_used'] = suffix_ms_override is not None
    return result
