"""Version-checked scheduler adapter for SealFlow layer-wise KV loading.

vLLM exposes ``scheduler_cls`` as a configurable class path, but its current
scheduler does not expose a small hook for admitting a request before an
asynchronous KV load completes.  This module derives a scheduler from the
installed vLLM implementation and applies a small, checked transformation to
the two affected methods.  Framework source files remain untouched, and an
unsupported vLLM version fails during import instead of silently changing
scheduling semantics.
"""

from __future__ import annotations

import inspect
import json
import os
import textwrap
import time

from vllm.v1.core.sched import async_scheduler as _async
from vllm.v1.core.sched import scheduler as _base


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SealFlow scheduler adapter expected one {label} site, found {count}; "
            "the installed vLLM scheduler is not supported"
        )
    return source.replace(old, new, 1)


def _compile_method(owner: type, name: str, transforms: list[tuple[str, str, str]]):
    source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    for old, new, label in transforms:
        source = _replace_once(source, old, new, label)
    namespace = dict(vars(_base))
    namespace.update({"os": os, "json": json, "time": time})
    exec(compile(source, f"<sealflow:{name}>", "exec"), namespace)
    return namespace[name]


_SCHEDULE_TRANSFORMS = [
    (
        """                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
""",
        """                    num_lookahead_tokens=self.num_lookahead_tokens,
                    delay_cache_blocks=self._has_pending_layerwise_load(request),
                )
""",
        "running request cache publication while H2D is pending",
    ),
    (
        """                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                        req_index -= 1
                else:
                    preempted_req = self.running.pop()
""",
        """                # An outstanding H2D still owns its destination blocks.
                # Select a victim whose receive has already completed.
                preemptible_reqs = [
                    req for req in self.running
                    if not self._has_pending_layerwise_load(req)
                ]
                if len(preemptible_reqs) != len(self.running):
                    logger.info(
                        "RUNTIME preemption selection: protected_h2d=%d eligible=%d",
                        len(self.running) - len(preemptible_reqs),
                        len(preemptible_reqs),
                    )
                if not preemptible_reqs:
                    # Existing worker work will deliver the completion.
                    break
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        preemptible_reqs,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                else:
                    preempted_req = preemptible_reqs[-1]
                removed_before_current = (
                    self.running.index(preempted_req) < req_index
                )
                self.running.remove(preempted_req)
                if preempted_req in scheduled_running_reqs:
                    preempted_req_id = preempted_req.request_id
                    scheduled_running_reqs.remove(preempted_req)
                    token_budget += num_scheduled_tokens.pop(preempted_req_id)
                    req_to_new_blocks.pop(preempted_req_id)
                    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                    preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                        preempted_req_id, None
                    )
                    if preempted_encoder_inputs:
                        num_embeds_to_restore = sum(
                            preempted_req.get_num_encoder_embeds(i)
                            for i in preempted_encoder_inputs
                        )
                        encoder_compute_budget += num_embeds_to_restore
                if removed_before_current:
                    req_index -= 1
""",
        "preserve in-flight H2D blocks and restore scheduled work for either policy",
    ),
    (
        """            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
""",
        """            num_external_computed_tokens = 0
            load_kv_async = False
            layerwise_load = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
""",
        "layer-wise state initialization",
    ),
    (
        """                    request.num_external_computed_tokens = ext_tokens
                    num_external_computed_tokens = ext_tokens

                    connector_prefix_cache_queries = (
""",
        """                    request.num_external_computed_tokens = ext_tokens
                    num_external_computed_tokens = ext_tokens
                    residual_tokens = (
                        request.num_tokens
                        - num_new_local_computed_tokens
                        - ext_tokens
                    )
                    layerwise_policy = getattr(
                        self.connector, "should_execute_layerwise_load", None
                    )
                    if layerwise_policy is not None:
                        layerwise_candidate = layerwise_policy(
                            ext_tokens, residual_tokens
                        )
                    else:
                        layerwise_candidate = getattr(
                            self.connector,
                            "can_execute_layerwise_load",
                            lambda: False,
                        )()
                    layerwise_load = bool(
                        load_kv_async
                        and ext_tokens > 0
                        and residual_tokens > 0
                        and layerwise_candidate
                    )

                    connector_prefix_cache_queries = (
""",
        "layer-wise admission policy",
    ),
    (
        """            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget

            if load_kv_async:
""",
        """            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget
            layerwise_h2d_pending = bool(
                self.connector is not None
                and getattr(
                    self.connector, "is_layerwise_load_request", lambda _: False
                )(request.request_id)
            )

            if load_kv_async and not layerwise_load:
""",
        "request admission branch",
    ),
    (
        "delay_cache_blocks=load_kv_async,",
        "delay_cache_blocks=load_kv_async or layerwise_h2d_pending,",
        "delayed cache publication",
    ),
    (
        """            if self.connector is not None:
                self.connector.update_state_after_alloc(
""",
        """            if self.connector is not None:
                if layerwise_load:
                    activate = getattr(
                        self.connector, "activate_layerwise_load", None
                    )
                    if activate is None:
                        raise RuntimeError(
                            "connector enabled layer-wise H2D without "
                            "activation support"
                        )
                    activate(request.request_id, num_external_computed_tokens)
                self.connector.update_state_after_alloc(
""",
        "connector activation",
    ),
    (
        """            request = request_queue.pop_request()
            if load_kv_async:
""",
        """            request = request_queue.pop_request()
            if load_kv_async and not layerwise_load:
""",
        "remote-KV waiting state",
    ),
]


_FINISHED_TRANSFORMS = [
    (
        """        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(self.requests[req_id])
""",
        """        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        elif (
            self.connector is not None
            and getattr(
                self.connector, "is_layerwise_load_request", lambda _: False
            )(req_id)
        ):
            safe_to_publish = self.connector.complete_layerwise_load(req_id)
            if req.status == RequestStatus.RUNNING and safe_to_publish:
                self.kv_cache_manager.cache_blocks(req, req.num_computed_tokens)
            elif req.status == RequestStatus.RUNNING:
                # Keep request-private blocks alive until the attention hook
                # reaches the output commit check and aborts this request.
                # Prevent the failed reload from becoming globally reusable.
                pass
            elif RequestStatus.is_finished(req.status):
                self._free_blocks(req)
            else:
                raise RuntimeError(
                    "layer-wise KV load completed for unexpected request "
                    f"state {req.status.name}: {req_id}"
                )
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(self.requests[req_id])
""",
        "layer-wise load completion",
    ),
    (
        """    for req_id in kv_connector_output.finished_sending or ():
        logger.debug("Finished sending KV transfer for request %s", req_id)
        assert req_id in self.requests
        self._free_blocks(self.requests[req_id])
""",
        """    for req_id in kv_connector_output.finished_sending or ():
        logger.debug("Finished sending KV transfer for request %s", req_id)
        if req_id not in self.requests:
            # A failed eager request may be removed after device-side D2H
            # reads finish, while its host-cache completion arrives later.
            continue
        self._free_blocks(self.requests[req_id])
""",
        "late D2H completion after eager request abort",
    ),
]


_OUTPUT_TRANSFORMS = [
    (
        """    failed_kv_load_req_ids = None
    if kv_connector_output and kv_connector_output.invalid_block_ids:
""",
        """    # Sampling may finish before the independent verifier streams.
    # Keep the token inside EngineCore until this request's reload passes.
    failed_commit_req_ids = set()
    if self.connector is not None and sampled_token_ids:
        for req_id in num_scheduled_tokens:
            req_index = model_runner_output.req_id_to_index[req_id]
            if sampled_token_ids[req_index]:
                committed = self.connector.wait_for_layerwise_commit(
                    req_id, "output"
                )
                if not committed:
                    failed_commit_req_ids.add(req_id)

    failed_kv_load_req_ids = None
    if kv_connector_output and kv_connector_output.invalid_block_ids:
""",
        "eager output commit dependency",
    ),
    (
        """    failed_kv_load_req_ids = None
    if kv_connector_output and kv_connector_output.invalid_block_ids:
""",
        """    failed_kv_load_req_ids = failed_commit_req_ids or None
    fault_flag = os.environ.get("PIPE_GCM_FAULT_FLAG_PATH", "")
    if (not os.environ.get("PIPE_GCM_COMMIT_DIR")
            and fault_flag and os.path.exists(fault_flag)):
        # The speculative forward may have consumed unauthenticated KV,
        # but no output is committed. The legacy single-request recovery
        # control uses the process-wide fault flag.
        failed_kv_load_req_ids = {
            req_id for req_id in num_scheduled_tokens
            if (request := self.requests.get(req_id)) is not None
            and not request.is_finished()
        }
    if failed_kv_load_req_ids:
        event_path = os.environ.get("PIPE_GCM_FAULT_EVENT_PATH", "")
        if event_path:
            with open(event_path, "a", encoding="utf-8") as event_file:
                event_file.write(json.dumps({
                    "phase": "dependent_state_invalidated",
                    "realtime_ns": time.time_ns(),
                    "request_count": len(failed_kv_load_req_ids),
                }, sort_keys=True) + "\\n")
    if (
        failed_kv_load_req_ids is None
        and kv_connector_output
        and kv_connector_output.invalid_block_ids
    ):
""",
        "speculative authentication-failure containment",
    ),
    (
        """        requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
""",
        """        requests = [
            request for req_id in failed_kv_load_req_ids
            if (request := self.requests.get(req_id)) is not None
        ]
""",
        "speculative failure request lookup",
    ),
    (
        """        self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
""",
        """        failed_device_block_ids = set()
        for request in requests:
            (request_block_ids,) = self.kv_cache_manager.get_block_ids(
                request.request_id
            )
            failed_device_block_ids.update(request_block_ids)
        if failed_device_block_ids:
            # Remove the failed request's blocks from the local prefix cache
            # before freeing the request.  Otherwise a retry could be served
            # from a stale device hit and would not exercise trusted H2D
            # recovery.
            self.kv_cache_manager.evict_blocks(failed_device_block_ids)
        self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
        # finish_requests has removed the requests from the scheduler
        # queues and released their private KV blocks.  Record this point
        # separately from invalidation so recovery timing can distinguish
        # notification from actual cleanup.
        cleanup_event_path = os.environ.get("PIPE_GCM_FAULT_EVENT_PATH", "")
        if cleanup_event_path:
            with open(cleanup_event_path, "a", encoding="utf-8") as event_file:
                event_file.write(json.dumps({
                    "phase": "dependent_state_cleanup_complete",
                    "realtime_ns": time.time_ns(),
                    "request_count": len(requests),
                }, sort_keys=True) + chr(10))
""",
        "speculative state cleanup event",
    )
]


class SealFlowScheduler(_async.AsyncScheduler):
    """Async vLLM scheduler with connector-controlled layer-wise KV admission.

    vLLM enables asynchronous scheduling by default for this configuration.
    The adapter must therefore retain ``AsyncScheduler`` state updates while
    changing the inherited scheduling and KV-completion methods.
    """

    schedule = _compile_method(_base.Scheduler, "schedule", _SCHEDULE_TRANSFORMS)
    _update_from_kv_xfer_finished = _compile_method(
        _base.Scheduler,
        "_update_from_kv_xfer_finished",
        _FINISHED_TRANSFORMS,
    )
    update_from_output = _compile_method(
        _base.Scheduler,
        "update_from_output",
        _OUTPUT_TRANSFORMS,
    )

    def _has_pending_layerwise_load(self, request) -> bool:
        return bool(
            self.connector is not None
            and getattr(
                self.connector, "is_layerwise_load_request", lambda _: False
            )(request.request_id)
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # Forced preemption also frees KV blocks. Return the native API's
        # unsuccessful-reset result until outstanding receives retire.
        if reset_running_requests and any(
            self._has_pending_layerwise_load(req) for req in self.running
        ):
            return False
        return super().reset_prefix_cache(reset_running_requests, reset_connector)
