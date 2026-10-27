"""Security-aware two-phase KV offloading connector for SecureInfer-ACL.

This module extends vLLM's OffloadingConnector without modifying framework
source.  A secure D2H store has two different completion semantics:

* DEVICE_COPY_DONE: encrypted DMA no longer reads the source NPU KV blocks.
  vLLM may release those blocks.
* SECURE_STORE_READY: host decryption and authentication have completed.
  Only now may the CPU cache be published.

The stock connector collapses both states into ``finished_sending``.  This
connector keeps ``finished_sending`` for device release and carries the later
cache-ready notification through KVConnectorWorkerMetadata, an existing vLLM
extension point.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
import time

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
except ImportError:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata as KVConnectorWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorMetadata,
    OffloadingConnectorScheduler,
    OffloadingConnectorWorker,
)
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput

from .attention_plugin import install_attention_plugin


# ``kv_connector_module_path`` loads this module in every vLLM process.  Use
# that supported extension point to install the generic Ascend attention wait
# without editing the vLLM-Ascend package.
if os.getenv("PIPE_GCM_H2D_LAYERWISE", "0") == "1":
    install_attention_plugin()


def _layerwise_h2d_enabled() -> bool:
    """Return whether the verified-layer H2D experiment is enabled.

    Keep the switch local to the connector as well as the NPU handler: the
    scheduler and worker live in different processes and must independently
    agree on the transaction semantics.  The default retains the stock
    request-atomic load path.
    """
    return os.getenv("PIPE_GCM_H2D_LAYERWISE", "0") == "1"


@dataclass
class SecureInferSecureStoreMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler notification for plaintext-ready CPU KV."""

    secure_store_ready_counts: dict[str, int] = field(default_factory=dict)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "SecureInferSecureStoreMetadata":
        if not isinstance(other, SecureInferSecureStoreMetadata):
            raise TypeError(
                "cannot aggregate SecureInfer secure-store metadata with "
                f"{type(other)!r}"
            )
        for req_id, count in other.secure_store_ready_counts.items():
            self.secure_store_ready_counts[req_id] = (
                self.secure_store_ready_counts.get(req_id, 0) + count
            )
        return self


@dataclass
class SecureInferOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler-to-worker transfer policy for each external reload.

    The environment variable only advertises that verified-layer execution is
    available.  The scheduler makes the per-request profitability decision;
    carrying the selected request IDs here prevents an atomic fallback from
    being mistaken for a layerwise job by the worker process.
    """

    layerwise_load_req_ids: set[str] = field(default_factory=set)


class SecureInferOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Offloading worker with separate device and secure-store completion."""

    def __init__(self, spec):
        super().__init__(spec)
        self._device_store_jobs: dict[str, set[int]] = defaultdict(set)
        self._device_reqs_with_store: set[str] = set()
        self._cache_reqs_with_store: set[str] = set()
        self._device_done_reqs: set[str] = set()
        self._cache_done_reqs: set[str] = set()
        self._finished_reqs_waiting_for_device: set[str] = set()
        self._finished_reqs_waiting_for_cache: set[str] = set()
        self._secure_store_ready_notifications: set[str] = set()
        # The two device/host completion queues are polled independently.
        # Neither notification is guaranteed to be observed first, so retain
        # a store job until both sides have reported completion.
        self._device_finished_store_jobs: set[int] = set()
        self._cache_finished_store_jobs: set[int] = set()
        # Evaluation control for the legacy coupled lifecycle: hold
        # finished_sending until both the encrypted device copy and secure CPU
        # publication complete.  The default remains the production two-phase
        # behavior, so this knob cannot affect normal serving accidentally.
        self._couple_cache_ready = (
            os.getenv("PIPE_D2H_COUPLE_CACHE_READY", "0") == "1"
        )

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        first_new_job = len(self._unsubmitted_store_jobs)
        super().prepare_store_kv(metadata)

        for job_id, _ in self._unsubmitted_store_jobs[first_new_job:]:
            req_id, store = self._jobs[job_id]
            assert store
            self._device_store_jobs[req_id].add(job_id)
            self._device_reqs_with_store.add(req_id)
            self._cache_reqs_with_store.add(req_id)
            # A request can produce multiple deferred store waves.  A newly
            # accepted job invalidates a phase-complete state from an earlier
            # wave until this job reaches the corresponding completion point.
            self._device_done_reqs.discard(req_id)
            self._cache_done_reqs.discard(req_id)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        if not isinstance(metadata, SecureInferOffloadingConnectorMetadata):
            raise TypeError(
                "SecureInfer worker requires per-request transfer metadata, got "
                f"{type(metadata)!r}"
            )

        # Preserve the base connector's deferred D2H behavior.
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        pending_loads: list[tuple[str, int, object, bool]] = []
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            is_layerwise = req_id in metadata.layerwise_load_req_ids
            if os.getenv("PIPE_GCM_COMMIT_DIR") and not is_layerwise:
                raise RuntimeError(
                    "eager-use commit control does not cover request-atomic H2D"
                )
            for handler in self.worker.handlers:
                set_policy = getattr(handler, "set_h2d_job_layerwise", None)
                if set_policy is not None:
                    set_policy(job_id, is_layerwise)
                if is_layerwise:
                    set_request = getattr(handler, "set_h2d_job_request", None)
                    if set_request is not None:
                        set_request(job_id, req_id)
            pending_loads.append((req_id, job_id, transfer_spec, is_layerwise))

        # All loads in this metadata object feed the same model execution
        # step.  Publish the complete membership before starting any
        # background submitter so a layer wave cannot advance merely because
        # another request's Python thread has not started yet.
        layerwise_job_ids = [
            job_id for _, job_id, _, is_layerwise in pending_loads
            if is_layerwise
        ]
        for handler in self.worker.handlers:
            set_batch = getattr(handler, "set_h2d_batch_layerwise_jobs", None)
            if set_batch is not None:
                set_batch(layerwise_job_ids)

        layerwise_batch = [
            (job_id, transfer_spec)
            for _, job_id, transfer_spec, is_layerwise in pending_loads
            if is_layerwise
        ]
        batch_submitted = False
        if len(layerwise_batch) > 1:
            first_spec = layerwise_batch[0][1]
            src, dst = first_spec
            transfer_type = (src.medium(), dst.medium())
            handler = self.worker.transfer_type_to_handler.get(transfer_type)
            submit_batch = getattr(
                handler, "transfer_layerwise_batch_async", None
            )
            if submit_batch is not None:
                batch_submitted = bool(submit_batch(layerwise_batch))

        layerwise_batch_ids = {job_id for job_id, _ in layerwise_batch}
        for _, job_id, transfer_spec, _ in pending_loads:
            if batch_submitted and job_id in layerwise_batch_ids:
                continue
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success

    def _get_device_finished_jobs(self) -> list[int]:
        finished: list[int] = []
        for handler in self.worker.handlers:
            get_device_finished = getattr(handler, "get_device_finished", None)
            if get_device_finished is not None:
                finished.extend(get_device_finished())
        return finished

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Attach all active verified-H2D events for ``layer_name``.

        Each registered handler owns the concrete tensor/event mapping.  The
        base offloading connector leaves this hook empty, which is correct for
        request-atomic loads but would defeat the layer pipeline.
        """
        if not _layerwise_h2d_enabled():
            return
        for handler in self.worker.handlers:
            wait_for_layer = getattr(handler, "wait_for_layer_load", None)
            if wait_for_layer is not None:
                wait_for_layer(layer_name)

    def _publish_device_done(
        self, req_id: str, finished_sending: set[str]
    ) -> None:
        self._finished_reqs_waiting_for_device.discard(req_id)
        self._device_done_reqs.discard(req_id)
        self._device_reqs_with_store.discard(req_id)
        finished_sending.add(req_id)
        logger.info(
            "SecureInfer DEVICE_COPY_DONE request=%s: NPU blocks may be released",
            req_id,
        )

    def _publish_cache_ready(self, req_id: str) -> None:
        self._finished_reqs_waiting_for_cache.discard(req_id)
        self._cache_done_reqs.discard(req_id)
        self._cache_reqs_with_store.discard(req_id)
        pending_jobs = self._store_jobs.get(req_id)
        if pending_jobs is not None:
            assert not pending_jobs
            del self._store_jobs[req_id]
        self._secure_store_ready_notifications.add(req_id)
        logger.info(
            "SecureInfer SECURE_STORE_READY request=%s: CPU cache may be published",
            req_id,
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        # Poll the source-safe event first.  Cache-ready may also be visible in
        # this engine step, but scheduler-side processing must never publish a
        # CPU entry without first making the device completion observable.
        for job_id in self._get_device_finished_jobs():
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError(
                    f"DEVICE_COPY_DONE arrived for unknown SecureInfer job {job_id}"
                )
            req_id, store = job
            if not store:
                raise RuntimeError(
                    f"DEVICE_COPY_DONE arrived for non-store job {job_id}"
                )
            req_jobs = self._device_store_jobs[req_id]
            req_jobs.remove(job_id)
            self._device_finished_store_jobs.add(job_id)
            if job_id in self._cache_finished_store_jobs:
                self._cache_finished_store_jobs.remove(job_id)
                self._device_finished_store_jobs.remove(job_id)
                self._jobs.pop(job_id)
            if not req_jobs:
                del self._device_store_jobs[req_id]
                self._device_done_reqs.add(req_id)
                if (
                    req_id in self._finished_reqs_waiting_for_device
                    and (
                        not self._couple_cache_ready
                        or req_id not in self._cache_reqs_with_store
                    )
                ):
                    self._publish_device_done(req_id, finished_sending)

        # Stock TransferResult now means SECURE_STORE_READY for D2H and demand
        # completion for H2D.  Keep the existing load behavior unchanged.
        for transfer_result in self.worker.get_finished():
            if hasattr(transfer_result, "job_id"):
                job_id = transfer_result.job_id
                transfer_success = transfer_result.success
                transfer_time = transfer_result.transfer_time
                transfer_size = transfer_result.transfer_size
                transfer_type = transfer_result.transfer_type
            else:
                # vLLM 0.14 returns the declared tuple[int, bool] form.
                job_id, transfer_success = transfer_result
                transfer_time = None
                transfer_size = None
                transfer_type = None
            assert transfer_success
            job = self._jobs.get(job_id)
            if job is None:
                raise RuntimeError(
                    f"SECURE_STORE_READY arrived for unknown SecureInfer job {job_id}"
                )
            req_id, store = job
            if (
                transfer_time
                and transfer_size is not None
                and transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_size,
                    time=transfer_time,
                    transfer_type=transfer_type,
                )

            if store:
                self._cache_finished_store_jobs.add(job_id)
                if job_id in self._device_finished_store_jobs:
                    self._device_finished_store_jobs.remove(job_id)
                    self._cache_finished_store_jobs.remove(job_id)
                    self._jobs.pop(job_id)
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if not req_jobs:
                    self._cache_done_reqs.add(req_id)
                    if req_id in self._finished_reqs_waiting_for_cache:
                        self._publish_cache_ready(req_id)
                        # In the evaluation-only coupled lifecycle, device
                        # completion may have arrived first and deliberately
                        # withheld finished_sending.  Cache readiness is the
                        # final dependency, so wake that request here rather
                        # than relying on finished_req_ids to be replayed.
                        if (
                            self._couple_cache_ready
                            and req_id in self._device_done_reqs
                            and req_id in self._finished_reqs_waiting_for_device
                        ):
                            self._publish_device_done(
                                req_id, finished_sending
                            )
            else:
                self._jobs.pop(job_id)
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        # A request output may return independently.  Source blocks wait only
        # for DEVICE_COPY_DONE; cache publication waits for the stronger
        # SECURE_STORE_READY state.
        for req_id in finished_req_ids:
            if req_id in self._device_reqs_with_store:
                cache_ready_for_coupled_control = (
                    not self._couple_cache_ready
                    or req_id not in self._cache_reqs_with_store
                )
                if (
                    req_id in self._device_done_reqs
                    and cache_ready_for_coupled_control
                ):
                    self._publish_device_done(req_id, finished_sending)
                else:
                    self._finished_reqs_waiting_for_device.add(req_id)

            if req_id in self._cache_reqs_with_store:
                if req_id in self._cache_done_reqs:
                    self._publish_cache_ready(req_id)
                    # Both phase-complete events may precede the scheduler's
                    # first finished_req_ids notification.  In that ordering,
                    # the device gate above deliberately waits for cache
                    # publication, so publishing the cache here must also
                    # release the already-complete device phase.  Without this
                    # wake-up, finished_req_ids is not replayed and NPU blocks
                    # remain pinned indefinitely in the coupled control run.
                    if (
                        self._couple_cache_ready
                        and req_id in self._device_done_reqs
                        and req_id in self._finished_reqs_waiting_for_device
                    ):
                        self._publish_device_done(req_id, finished_sending)
                else:
                    self._finished_reqs_waiting_for_cache.add(req_id)

        return finished_sending, finished_recving

    def take_secure_store_metadata(self) -> SecureInferSecureStoreMetadata | None:
        if not self._secure_store_ready_notifications:
            return None
        metadata = SecureInferSecureStoreMetadata(
            secure_store_ready_counts={
                req_id: 1 for req_id in self._secure_store_ready_notifications
            }
        )
        self._secure_store_ready_notifications = set()
        return metadata


class SecureInferOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    """Scheduler side of the security-aware two-completion protocol."""

    def __init__(self, spec):
        super().__init__(spec)
        # Older vLLM offloading schedulers do not track this optional set.
        if not hasattr(self, "_blocks_being_loaded"):
            self._blocks_being_loaded = None
        self._secure_ready_expected_workers = max(
            1, int(spec.vllm_config.parallel_config.world_size)
        )
        self._secure_ready_worker_counts: dict[str, int] = defaultdict(int)
        self._secure_admission_enabled = (
            os.getenv("PIPE_D2H_ADMISSION", "1") != "0"
        )
        try:
            self._secure_pending_max_bytes = int(
                os.getenv("PIPE_D2H_PENDING_MAX_BYTES", str(4 << 30))
            )
        except ValueError:
            self._secure_pending_max_bytes = 4 << 30
        try:
            self._secure_pending_max_jobs = int(
                os.getenv("PIPE_D2H_PENDING_MAX_JOBS", "16")
            )
        except ValueError:
            self._secure_pending_max_jobs = 16
        self._secure_pending_max_bytes = max(1, self._secure_pending_max_bytes)
        self._secure_pending_max_jobs = max(1, self._secure_pending_max_jobs)

        page_sizes = {
            group.kv_cache_spec.page_size_bytes
            for group in spec.kv_cache_config.kv_cache_groups
        }
        if len(page_sizes) != 1:
            raise ValueError(
                "SecureInfer secure admission requires a uniform KV page size"
            )
        block_size_factor = getattr(spec, "block_size_factor", None)
        if block_size_factor is None:
            gpu_block_size = int(getattr(spec, "gpu_block_size"))
            offloaded_block_size = int(getattr(spec, "offloaded_block_size"))
            block_size_factor = max(
                1, offloaded_block_size // max(1, gpu_block_size)
            )
        self._kv_bytes_per_offloaded_block = (
            page_sizes.pop()
            * len(spec.kv_cache_config.kv_cache_tensors)
            * int(spec.vllm_config.parallel_config.world_size)
            * int(block_size_factor)
        )
        self._secure_pending_bytes = 0
        self._secure_pending_jobs = 0
        self._secure_pending_by_req: dict[str, int] = defaultdict(int)
        self._secure_pending_jobs_by_req: dict[str, int] = defaultdict(int)
        self._secure_store_rejected_reqs: set[str] = set()
        self._secure_admitted_jobs = 0
        self._secure_admission_drops = 0
        # These requests run while their H2D transaction is still in flight.
        # Their allocated blocks are deliberately *not* placed in vLLM's
        # global prefix cache until the worker reports request-level GCM
        # completion.  The scheduler consults this set when handling the
        # usual ``finished_recving`` notification.
        self._layerwise_load_reqs: set[str] = set()
        self._eager_commit_reqs: set[str] = set()
        self._eager_output_checked: set[str] = set()
        logger.info(
            "SecureInfer secure-store admission: enabled=%s max_jobs=%d "
            "max_bytes=%d bytes_per_block=%d",
            self._secure_admission_enabled,
            self._secure_pending_max_jobs,
            self._secure_pending_max_bytes,
            self._kv_bytes_per_offloaded_block,
        )

    def _get_reqs_to_store(self, scheduler_output):
        """Admit optional cache stores before worker jobs are constructed.

        Dropping a store does not affect the completed request or its model
        output.  It only forgoes a future external-cache hit.  Calling
        complete_store(success=False) rolls back CPU-cache allocation, while
        the already advanced stored-block cursor prevents retry storms.
        """
        # Do this before the base connector asks its CPU-cache manager to
        # reserve blocks.  Reserving and then rolling back a setup request can
        # still evict the experiment's target entry from the CPU LRU, turning
        # the intended external reload into a full prefill.
        skip_marker = os.getenv("PIPE_SKIP_SECURE_STORE_MARKER_PATH", "")
        if skip_marker and os.path.exists(skip_marker):
            return {}
        previous_hashes = {
            req_id: set(block_hashes)
            for req_id, block_hashes in self._reqs_being_stored.items()
        }
        reqs_to_store = super()._get_reqs_to_store(scheduler_output)
        if not self._secure_admission_enabled:
            return reqs_to_store

        for req_id in list(reqs_to_store):
            all_hashes = self._reqs_being_stored[req_id]
            new_hashes = all_hashes - previous_hashes.get(req_id, set())
            job_bytes = len(new_hashes) * self._kv_bytes_per_offloaded_block
            over_jobs = self._secure_pending_jobs + 1 > self._secure_pending_max_jobs
            over_bytes = (
                self._secure_pending_bytes + job_bytes
                > self._secure_pending_max_bytes
            )
            reject = (
                req_id in self._secure_store_rejected_reqs
                or over_jobs
                or over_bytes
            )
            if reject:
                if req_id in self._secure_store_rejected_reqs:
                    reject_reason = "sticky"
                elif over_jobs:
                    reject_reason = "jobs"
                else:
                    reject_reason = "bytes"
                if new_hashes:
                    self.manager.complete_store(new_hashes, success=False)
                    all_hashes.difference_update(new_hashes)
                if not all_hashes:
                    self._reqs_being_stored.pop(req_id, None)
                reqs_to_store.pop(req_id, None)
                self._secure_store_rejected_reqs.add(req_id)
                self._secure_admission_drops += 1
                logger.warning(
                    "SecureInfer secure-store admission rejected request=%s "
                    "blocks=%d bytes=%d pending_jobs=%d pending_bytes=%d "
                    "reason=%s",
                    req_id,
                    len(new_hashes),
                    job_bytes,
                    self._secure_pending_jobs,
                    self._secure_pending_bytes,
                    reject_reason,
                )
                continue

            self._secure_pending_bytes += job_bytes
            self._secure_pending_jobs += 1
            self._secure_pending_by_req[req_id] += job_bytes
            self._secure_pending_jobs_by_req[req_id] += 1
            self._secure_admitted_jobs += 1
            logger.info(
                "SecureInfer secure-store admitted request=%s blocks=%d bytes=%d "
                "pending_jobs=%d pending_bytes=%d",
                req_id,
                len(new_hashes),
                job_bytes,
                self._secure_pending_jobs,
                self._secure_pending_bytes,
            )

        return reqs_to_store

    def _release_secure_admission(self, req_id: str) -> None:
        released_bytes = self._secure_pending_by_req.pop(req_id, 0)
        released_jobs = self._secure_pending_jobs_by_req.pop(req_id, 0)
        self._secure_pending_bytes = max(
            0, self._secure_pending_bytes - released_bytes
        )
        self._secure_pending_jobs = max(
            0, self._secure_pending_jobs - released_jobs
        )
        logger.info(
            "SecureInfer secure-store admission released request=%s bytes=%d "
            "jobs=%d pending_jobs=%d pending_bytes=%d admitted=%d dropped=%d",
            req_id,
            released_bytes,
            released_jobs,
            self._secure_pending_jobs,
            self._secure_pending_bytes,
            self._secure_admitted_jobs,
            self._secure_admission_drops,
        )

    def request_finished(self, request, block_ids):
        result = super().request_finished(request, block_ids)
        self._secure_store_rejected_reqs.discard(request.request_id)
        # Do not discard an in-flight layerwise receive here.  A model output
        # can become host-visible before the worker's final H2D event is
        # polled; dropping this marker would let the scheduler free the
        # destination blocks before the authenticated stream has retired.
        # ``complete_layerwise_load`` owns the matching cleanup.  Report the
        # outstanding receive as an asynchronous dependency so the core keeps
        # its destination blocks pinned if generation happens to finish before
        # the final layer's authenticated event is observed.
        return (
            bool(result[0] or request.request_id in self._layerwise_load_reqs),
            result[1],
        )

    def activate_layerwise_load(self, req_id: str, external_tokens: int) -> None:
        """Mark an already-allocated request for verified-layer execution.

        The core scheduler makes this decision only when the external hit
        leaves at least one prompt token to execute.  A full external hit has
        to follow vLLM's normal request-atomic path so it can recompute its
        final token.  Keeping the decision in the scheduler, rather than
        inferring it here from allocation state, prevents the two paths from
        disagreeing about cache publication.
        """
        if not _layerwise_h2d_enabled():
            raise RuntimeError("layerwise H2D activated while the feature is disabled")
        if req_id in self._layerwise_load_reqs:
            raise RuntimeError(f"duplicate layerwise H2D activation for {req_id}")
        self._layerwise_load_reqs.add(req_id)
        if os.getenv("PIPE_GCM_COMMIT_DIR"):
            self._eager_commit_reqs.add(req_id)
        logger.info(
            "SecureInfer layerwise H2D admitted request=%s external_tokens=%d; "
            "prefix-cache publication deferred until final GCM event",
            req_id,
            external_tokens,
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        """Register a layerwise load without claiming its residual block.

        The stock offloading connector is intentionally request-atomic: when
        it records an external hit, every newly allocated GPU block must be a
        remote block.  A verified-layer request is different: it loads the
        external full blocks *and* computes a residual prompt suffix in the
        same scheduler step.  The latter may allocate one additional partial
        GPU block, which is not part of the external transfer.  Copy the
        small load-registration portion of the base implementation so its
        destination spec contains exactly the external blocks.

        No cryptographic state is relaxed here.  The worker still performs
        normal AES-GCM verification for every block, and the layer events
        gate model access to those destination blocks.
        """
        if request.request_id not in self._layerwise_load_reqs:
            return super().update_state_after_alloc(
                request, blocks, num_external_tokens
            )

        self._requests[request.request_id] = request
        self._request_block_ids[request.request_id] = []
        if num_external_tokens == 0:
            raise RuntimeError(
                "layerwise H2D request has no external blocks to verify"
            )
        if num_external_tokens % self.gpu_block_size:
            raise RuntimeError(
                "layerwise H2D requires a whole-number of GPU KV blocks"
            )

        block_ids = blocks.get_block_ids()[0]
        num_local_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_external_gpu_blocks = num_external_tokens // self.gpu_block_size
        end_gpu_block = num_local_gpu_blocks + num_external_gpu_blocks
        if len(block_ids) < end_gpu_block:
            raise RuntimeError(
                "layerwise H2D allocation is shorter than its external KV "
                f"range: allocated={len(block_ids)} expected_at_least={end_gpu_block}"
            )

        num_local_tokens = num_local_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_local_tokens + num_external_tokens
        if full_block_tokens % self.offloaded_block_size:
            raise RuntimeError(
                "layerwise H2D external range is not aligned to the CPU KV "
                "block size"
            )
        num_offloaded_blocks = full_block_tokens // self.offloaded_block_size
        if len(request.block_hashes) // self.block_size_factor < num_offloaded_blocks:
            raise RuntimeError(
                "layerwise H2D request does not own all external block hashes"
            )

        block_hashes = self._get_block_hashes(
            request,
            start_idx=num_local_tokens // self.offloaded_block_size,
            end_idx=num_offloaded_blocks,
        )
        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(
            block_ids[num_local_gpu_blocks:end_gpu_block]
        )
        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)

        registered_hashes = set(
            self._get_block_hashes(
                request,
                start_idx=num_local_tokens // self.offloaded_block_size,
                end_idx=num_offloaded_blocks,
            )
        )
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(registered_hashes)
        self._next_stored_block_idx[request.request_id] = num_offloaded_blocks
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)
        logger.debug(
            "SecureInfer layerwise H2D registered request=%s external_gpu_blocks=%d "
            "residual_allocated_blocks=%d",
            request.request_id,
            num_external_gpu_blocks,
            len(block_ids) - end_gpu_block,
        )
        return

    def is_layerwise_load_request(self, req_id: str) -> bool:
        return req_id in self._layerwise_load_reqs

    def wait_for_layerwise_commit(self, req_id: str, phase: str) -> bool:
        if req_id not in self._eager_commit_reqs:
            return True
        if phase == "output" and req_id in self._eager_output_checked:
            return True
        path = os.path.join(os.environ["PIPE_GCM_COMMIT_DIR"], f"{req_id}.json")
        deadline = time.monotonic() + 60.0
        began_ns = time.time_ns()
        while not os.path.exists(path):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out awaiting H2D authentication: {req_id}")
            time.sleep(0.001)
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream)
        status = int(result["status"])
        event_path = os.getenv("PIPE_GCM_FAULT_EVENT_PATH", "")
        if event_path:
            with open(event_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "phase": f"{phase}_commit_checked",
                    "realtime_ns": time.time_ns(),
                    "wait_ms": (time.time_ns() - began_ns) / 1e6,
                    "request_id": req_id,
                    "status": status,
                }) + "\n")
        if status == 1:
            if phase == "output":
                self._eager_output_checked.add(req_id)
            return True
        return False

    def _clear_layerwise_load_marker(self, req_id: str) -> None:
        if req_id not in self._layerwise_load_reqs:
            raise RuntimeError(
                f"unexpected layerwise H2D completion for request {req_id}"
            )
        self._layerwise_load_reqs.remove(req_id)
        logger.info(
            "SecureInfer layerwise H2D complete request=%s; authenticated KV may "
            "now enter the shared prefix cache",
            req_id,
        )

    def build_connector_meta(self, scheduler_output):
        load_req_ids = set(self._reqs_to_load)
        metadata = SecureInferOffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
            layerwise_load_req_ids=(
                load_req_ids.intersection(self._layerwise_load_reqs)
            ),
        )
        self._reqs_to_load = {}
        # The stock connector marks preempted stores complete here because it
        # lacks a second readiness notification.  SecureInfer must wait for the
        # actual secure-store metadata instead.
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        worker_metadata = getattr(
            connector_output, "kv_connector_worker_meta", None
        )
        if worker_metadata is not None:
            if not isinstance(worker_metadata, SecureInferSecureStoreMetadata):
                raise TypeError(
                    "unexpected worker metadata for SecureInfer connector: "
                    f"{type(worker_metadata)!r}"
                )
            ready_reqs: list[str] = []
            for req_id, count in worker_metadata.secure_store_ready_counts.items():
                self._secure_ready_worker_counts[req_id] += count
                if (
                    self._secure_ready_worker_counts[req_id]
                    >= self._secure_ready_expected_workers
                ):
                    ready_reqs.append(req_id)

            for req_id in ready_reqs:
                del self._secure_ready_worker_counts[req_id]
                block_hashes = self._reqs_being_stored.pop(req_id, None)
                safe_to_publish = self.wait_for_layerwise_commit(
                    req_id, "host_kv_publish"
                )
                if block_hashes:
                    self.manager.complete_store(
                        block_hashes, success=safe_to_publish
                    )
                self._release_secure_admission(req_id)
                logger.info(
                    "SecureInfer secure CPU KV store request=%s blocks=%d published=%s",
                    req_id,
                    len(block_hashes) if block_hashes else 0,
                    safe_to_publish,
                )

        # This notification only says that the worker's request-level H2D
        # event is complete.  For the verified-layer path the core scheduler
        # owns the final global-prefix-cache publication after it sees the
        # same notification, so retain connector bookkeeping until then.
        for req_id in connector_output.finished_recving or []:
            if req_id in self._layerwise_load_reqs:
                continue
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)
                self.manager.complete_load(block_hashes)

    def complete_layerwise_load(self, req_id: str) -> bool:
        """Retire connector state and report whether KV may be cached.

        The fault-recovery experiment publishes an authentication failure
        through a process-shared flag.  A failed speculative request may keep
        private blocks until vLLM aborts it, but those blocks must not enter
        the global prefix cache.
        """
        fault_flag = os.getenv("PIPE_GCM_FAULT_FLAG_PATH", "")
        safe_to_publish = self.wait_for_layerwise_commit(
            req_id, "device_kv_publish"
        ) and not (
            not os.getenv("PIPE_GCM_COMMIT_DIR")
            and fault_flag and os.path.exists(fault_flag)
        )
        self._clear_layerwise_load_marker(req_id)
        block_hashes = self._reqs_being_loaded.pop(req_id, None)
        if block_hashes:
            if self._blocks_being_loaded:
                self._blocks_being_loaded.difference_update(block_hashes)
            self.manager.complete_load(block_hashes)
        if not safe_to_publish:
            logger.warning(
                "SealFlow quarantined failed H2D KV from prefix cache: request=%s",
                req_id,
            )
        return safe_to_publish


class SecureInferOffloadingConnector(OffloadingConnector):
    """Dynamically loaded, source-unmodified vLLM connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # Reproduce OffloadingConnector's small constructor so it instantiates
        # our scheduler/worker variants rather than the stock single-phase
        # implementations.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert kv_cache_config is not None
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: SecureInferOffloadingConnectorScheduler | None = None
        self.connector_worker: SecureInferOffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SecureInferOffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = SecureInferOffloadingConnectorWorker(spec)

        logger.info("SecureInfer security-aware two-phase connector role=%s", role)

    def can_execute_layerwise_load(self) -> bool:
        """Whether vLLM may run a request before its full H2D completes.

        The paired scheduler change still delays cache publication; actual KV
        use is protected by the per-layer event installed at attention entry.
        This is kept as an explicit capability rather than inferred from a
        connector type so every other connector retains stock behavior.
        """
        return _layerwise_h2d_enabled()

    def should_execute_layerwise_load(
        self, external_tokens: int, residual_tokens: int
    ) -> bool:
        """Use layerwise release only when useful overlap exists.

        Very small reloads or a nearly complete external hit expose too little
        authentication/prefill work to amortize the background submitter and
        per-layer events.  Falling back to the stock request-atomic path in
        that region makes the optimization non-regressive while preserving
        identical GCM security semantics.
        """
        if not _layerwise_h2d_enabled():
            return False
        try:
            min_external = max(
                1, int(os.getenv("PIPE_GCM_LAYERWISE_MIN_EXTERNAL_TOKENS", "1024"))
            )
            min_residual = max(
                1, int(os.getenv("PIPE_GCM_LAYERWISE_MIN_RESIDUAL_TOKENS", "128"))
            )
        except ValueError:
            min_external = 1024
            min_residual = 128
        return external_tokens >= min_external and residual_tokens >= min_residual

    def is_layerwise_load_request(self, req_id: str) -> bool:
        if self.connector_scheduler is None:
            return False
        return self.connector_scheduler.is_layerwise_load_request(req_id)

    def activate_layerwise_load(self, req_id: str, external_tokens: int) -> None:
        if self.connector_scheduler is None:
            raise RuntimeError("layerwise H2D activation reached a worker connector")
        self.connector_scheduler.activate_layerwise_load(req_id, external_tokens)

    def complete_layerwise_load(self, req_id: str) -> bool:
        if self.connector_scheduler is None:
            raise RuntimeError("layerwise completion reached a worker connector")
        return self.connector_scheduler.complete_layerwise_load(req_id)

    def wait_for_layerwise_commit(self, req_id: str, phase: str) -> bool:
        if self.connector_scheduler is None:
            raise RuntimeError("output commit reached a worker connector")
        return self.connector_scheduler.wait_for_layerwise_commit(req_id, phase)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not _layerwise_h2d_enabled():
            return
        if self.connector_worker is None:
            raise RuntimeError("layerwise H2D wait reached a scheduler connector")
        self.connector_worker.wait_for_layer_load(layer_name)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.take_secure_store_metadata()
