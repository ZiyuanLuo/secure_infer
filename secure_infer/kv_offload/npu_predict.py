"""SecureInfer-aware NPU KV offload module.

This module is intentionally outside vllm-ascend.  vLLM loads it through
kv_transfer_config.extra_config["spec_module_path"], so the framework source can
stay pristine while SecureInfer observes the host/device copies issued by the
runtime.  Cryptographic preparation is driven by the native history predictor;
this module does not submit an exact future block list to the crypto queues.
"""

from __future__ import annotations

from collections.abc import Iterator
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec
from vllm.config import VllmConfig

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler, expand_block_ids
from vllm_ascend.kv_offload.npu import NPUOffloadingSpec as _AscendNPUOffloadingSpec


_SECURE_INFER_REGISTER_HOST_RANGE = None
_SECURE_INFER_REGISTER_HOST_RANGE_EX = None
_SECURE_INFER_GET_TRANSFER_PREFETCH_STATS = None
_SECURE_INFER_GET_NPU_KEYSTREAM_STATS = None
_SECURE_INFER_GET_GCM_KEYSTREAM_STATS = None
_SECURE_INFER_GET_NPU_RESOURCE_STATS = None
_SECURE_INFER_RETIRE_TRANSFER_PREFETCHES = None

_SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES = None
_SECURE_INFER_D2H_BATCH_BEGIN = None
_SECURE_INFER_D2H_GROUP_BEGIN = None
_SECURE_INFER_D2H_GROUP_END = None
_SECURE_INFER_D2H_DEVICE_BARRIER = None
_SECURE_INFER_D2H_BATCH_END = None
_SECURE_INFER_FLUSH_H2D_PROFILE = None
_SECURE_INFER_DRAIN_H2D_PROFILE = None
_SECURE_INFER_SYMBOL_LOOKUP_DONE = False
_SECURE_INFER_LAST_TRANSFER_STATS = None
_SECURE_INFER_LAST_NPU_STATS = None
_SECURE_INFER_LAST_GCM_KEYSTREAM_STATS = None
_SECURE_INFER_LAST_RESOURCE_STATS = None
_TRANSFER_GEOMETRY_LOCK = threading.Lock()


def _d2h_write_behind_enabled() -> bool:
    return os.getenv("PIPE_D2H_WRITE_BEHIND", "1") != "0"


def _d2h_two_phase_enabled() -> bool:
    return os.getenv("PIPE_D2H_TWO_PHASE", "1") != "0"


def _d2h_admission_enabled() -> bool:
    return os.getenv("PIPE_D2H_ADMISSION", "1") != "0"


def _d2h_pending_job_limit() -> int:
    try:
        value = int(os.getenv("PIPE_D2H_PENDING_MAX_JOBS", "16"))
    except ValueError:
        value = 16
    return max(1, value)


def _d2h_pending_byte_limit() -> int:
    try:
        value = int(os.getenv("PIPE_D2H_PENDING_MAX_BYTES", str(4 << 30)))
    except ValueError:
        value = 4 << 30
    return max(1, value)


def _d2h_group_blocks() -> int:
    """Bound one encrypt-and-pack group without changing logical semantics."""
    try:
        value = int(os.getenv("PIPE_D2H_GROUP_BLOCKS", "16"))
    except ValueError:
        value = 16
    return max(1, min(value, 128))


@dataclass
class _D2HSubmission:
    job_id: int
    spec: TransferSpec
    bytes: int
    queued_ns: int = field(default_factory=time.monotonic_ns)
    submitted: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    device_done: bool = False
    device_done_ns: int = 0


def _transfer_geometry_path() -> Path | None:
    raw = os.getenv("PIPE_TRANSFER_GEOMETRY_PATH", "")
    return Path(raw) if raw else None


def _append_transfer_geometry(record: dict[str, object]) -> None:
    """Persist transfer geometry for diagnostics before a memcpy burst.

    The ACL hook remains the ground truth for observed copy sizes.  This
    metadata is diagnostic only; it is not submitted to the crypto predictor
    and does not create future tickets.
    """
    path = _transfer_geometry_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRANSFER_GEOMETRY_LOCK:
        with path.open("a", encoding="utf-8") as geometry_file:
            geometry_file.write(json.dumps(record, sort_keys=True) + "\n")


def _gcm_h2d_slice_streams(block_bytes: int = 0) -> int:
    """Return the opt-in intra-request GCM H2D stream width.

    A single external-KV reload has many source/destination-disjoint physical
    KV slices.  Splitting those slices across independent streams lets Ascend
    schedule several AICPU GHASH tasks concurrently, while a join event keeps
    the framework-visible load completion atomic.  ``auto`` bases its width
    on the physical transfer geometry already available during connector
    initialization, rather than on a model name or a hard-coded layer count.
    """
    raw = os.getenv("PIPE_GCM_H2D_SLICE_STREAMS", "1").strip().lower()
    if raw == "auto":
        # 256-KiB GQA slices are host-submission dominated once three streams
        # are active, so extra streams do not shorten end-to-end reload.
        # MiB-scale MHA slices instead leave each AICPU GHASH invocation long
        # enough to benefit from six independent chains.  This policy was
        # cross-checked with strict per-slice tag validation; it never changes
        # the IV/tag mapping or request-level completion ordering.
        if block_bytes >= (1 << 20):
            return 6
        return 3 if block_bytes > 0 else 1
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(value, 8))


def _lookup_secure_infer_symbols() -> None:
    global _SECURE_INFER_REGISTER_HOST_RANGE
    global _SECURE_INFER_REGISTER_HOST_RANGE_EX
    global _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS
    global _SECURE_INFER_GET_NPU_KEYSTREAM_STATS
    global _SECURE_INFER_GET_GCM_KEYSTREAM_STATS
    global _SECURE_INFER_GET_NPU_RESOURCE_STATS
    global _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES
    global _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES
    global _SECURE_INFER_D2H_BATCH_BEGIN
    global _SECURE_INFER_D2H_GROUP_BEGIN
    global _SECURE_INFER_D2H_GROUP_END
    global _SECURE_INFER_D2H_DEVICE_BARRIER
    global _SECURE_INFER_D2H_BATCH_END
    global _SECURE_INFER_FLUSH_H2D_PROFILE, _SECURE_INFER_DRAIN_H2D_PROFILE
    global _SECURE_INFER_SYMBOL_LOOKUP_DONE

    if _SECURE_INFER_SYMBOL_LOOKUP_DONE:
        return
    _SECURE_INFER_SYMBOL_LOOKUP_DONE = True

    try:
        process = ctypes.CDLL(None)
    except OSError:
        return

    def bind(name, argtypes, restype=None):
        function = getattr(process, name, None)
        if function is None:
            return None
        try:
            function.argtypes = argtypes
            function.restype = restype
        except AttributeError:
            return None
        return function

    _SECURE_INFER_REGISTER_HOST_RANGE = bind(
        "secure_infer_register_host_range",
        [ctypes.c_void_p, ctypes.c_size_t],
    )
    _SECURE_INFER_REGISTER_HOST_RANGE_EX = bind(
        "secure_infer_register_host_range_ex",
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_size_t,
        ],
    )
    _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS = bind(
        "secure_infer_get_transfer_prefetch_stats",
        [ctypes.POINTER(ctypes.c_uint64)] * 3,
    )
    _SECURE_INFER_GET_NPU_KEYSTREAM_STATS = bind(
        "secure_infer_get_npu_keystream_stats",
        [ctypes.POINTER(ctypes.c_uint64)] * 3,
    )
    _SECURE_INFER_GET_GCM_KEYSTREAM_STATS = bind(
        "secure_infer_get_gcm_keystream_stats",
        [ctypes.POINTER(ctypes.c_uint64)] * 7,
    )
    _SECURE_INFER_GET_NPU_RESOURCE_STATS = bind(
        "secure_infer_get_npu_resource_stats",
        [
            *([ctypes.POINTER(ctypes.c_int32)] * 5),
            *([ctypes.POINTER(ctypes.c_uint64)] * 7),
        ],
    )
    _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES = bind(
        "secure_infer_invalidate_transfer_prefetches", []
    )
    _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES = bind(
        "secure_infer_retire_transfer_prefetches", []
    )
    _SECURE_INFER_D2H_BATCH_BEGIN = bind(
        "secure_infer_d2h_batch_begin", [ctypes.c_size_t], ctypes.c_int
    )
    _SECURE_INFER_D2H_GROUP_BEGIN = bind(
        "secure_infer_d2h_group_begin", [ctypes.c_size_t], ctypes.c_int
    )
    _SECURE_INFER_D2H_GROUP_END = bind(
        "secure_infer_d2h_group_end", [], ctypes.c_int
    )
    _SECURE_INFER_D2H_DEVICE_BARRIER = bind(
        "secure_infer_d2h_batch_device_barrier", [], ctypes.c_int
    )
    _SECURE_INFER_D2H_BATCH_END = bind(
        "secure_infer_d2h_batch_end", [], ctypes.c_int
    )
    _SECURE_INFER_FLUSH_H2D_PROFILE = bind(
        "secure_infer_flush_completed_h2d_profile", []
    )
    _SECURE_INFER_DRAIN_H2D_PROFILE = bind(
        "secure_infer_drain_h2d_profile_after_job", []
    )


def _log_transfer_prefetch_stats() -> None:
    global _SECURE_INFER_LAST_TRANSFER_STATS

    _lookup_secure_infer_symbols()
    if _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS is None:
        return
    ready = ctypes.c_uint64()
    busy = ctypes.c_uint64()
    missing = ctypes.c_uint64()
    _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS(
        ctypes.byref(ready), ctypes.byref(busy), ctypes.byref(missing)
    )
    stats = (ready.value, busy.value, missing.value)
    if stats == _SECURE_INFER_LAST_TRANSFER_STATS:
        return
    _SECURE_INFER_LAST_TRANSFER_STATS = stats
    logger.info(
        "SecureInfer CPU pre-encryption stats: ready=%d busy_fallback=%d "
        "missing_fallback=%d",
        *stats,
    )

def _log_npu_keystream_stats() -> None:
    global _SECURE_INFER_LAST_NPU_STATS

    _lookup_secure_infer_symbols()
    if _SECURE_INFER_GET_NPU_KEYSTREAM_STATS is None:
        return
    ready = ctypes.c_uint64()
    busy = ctypes.c_uint64()
    missing = ctypes.c_uint64()
    _SECURE_INFER_GET_NPU_KEYSTREAM_STATS(
        ctypes.byref(ready), ctypes.byref(busy), ctypes.byref(missing)
    )
    stats = (ready.value, busy.value, missing.value)
    if stats == _SECURE_INFER_LAST_NPU_STATS:
        return
    _SECURE_INFER_LAST_NPU_STATS = stats
    logger.info(
        "SecureInfer NPU keystream stats: ready=%d busy_fallback=%d missing_fallback=%d",
        *stats,
    )


def _log_gcm_keystream_stats() -> None:
    """Log the process-local ticket-pool counters after a secure store.

    ctypes.CDLL(None) resolves symbols in the EngineCore process, so unlike a
    standalone client this reports the actual pool that served H2D reloads.
    """
    global _SECURE_INFER_LAST_GCM_KEYSTREAM_STATS

    _lookup_secure_infer_symbols()
    if _SECURE_INFER_GET_GCM_KEYSTREAM_STATS is None:
        return
    values = [ctypes.c_uint64() for _ in range(7)]
    _SECURE_INFER_GET_GCM_KEYSTREAM_STATS(
        *(ctypes.byref(value) for value in values)
    )
    stats = tuple(value.value for value in values)
    if stats == _SECURE_INFER_LAST_GCM_KEYSTREAM_STATS:
        return
    _SECURE_INFER_LAST_GCM_KEYSTREAM_STATS = stats
    logger.info(
        "SecureInfer GCM ticket keystream: ready=%d busy=%d missing=%d "
        "aicpu_submitted=%d aicore_submitted=%d dropped=%d switches=%d",
        *stats,
    )


def _log_npu_resource_stats() -> None:
    global _SECURE_INFER_LAST_RESOURCE_STATS

    _lookup_secure_infer_symbols()
    if _SECURE_INFER_GET_NPU_RESOURCE_STATS is None:
        return

    int_values = [ctypes.c_int32() for _ in range(5)]
    count_values = [ctypes.c_uint64() for _ in range(7)]
    _SECURE_INFER_GET_NPU_RESOURCE_STATS(
        *(ctypes.byref(value) for value in (*int_values, *count_values))
    )
    names = (
        "cube_util",
        "vector_util",
        "aicpu_util",
        "memory_util",
        "pressure",
        "samples",
        "sample_errors",
        "pressure_samples",
        "submitted",
        "throttled",
        "deferred_pressure",
        "deferred_capacity",
    )
    values = tuple(value.value for value in (*int_values, *count_values))
    if values == _SECURE_INFER_LAST_RESOURCE_STATS:
        return
    _SECURE_INFER_LAST_RESOURCE_STATS = values
    stats = dict(zip(names, values))
    logger.info(
        "SecureInfer NPU resource stats: cube=%d vector=%d aicpu=%d memory=%d "
        "pressure=%d submitted=%d throttled=%d deferred_pressure=%d "
        "deferred_capacity=%d samples=%d errors=%d",
        stats["cube_util"],
        stats["vector_util"],
        stats["aicpu_util"],
        stats["memory_util"],
        stats["pressure"],
        stats["submitted"],
        stats["throttled"],
        stats["deferred_pressure"],
        stats["deferred_capacity"],
        stats["samples"],
        stats["sample_errors"],
    )

    stats_path = os.getenv("PIPE_NPU_RESOURCE_STATS_PATH")
    if stats_path:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats["timestamp_ns"] = time.time_ns()
        with path.open("a", encoding="utf-8") as stats_file:
            stats_file.write(json.dumps(stats, sort_keys=True) + "\n")


def _register_host_range(
    tensor: torch.Tensor,
    layer_id: int | None = None,
    tensor_kind: int | None = None,
) -> None:
    _lookup_secure_infer_symbols()
    if _SECURE_INFER_REGISTER_HOST_RANGE_EX is not None and layer_id is not None:
        block_bytes = (
            int(tensor[0].numel() * tensor.element_size())
            if tensor.ndim > 0 and tensor.shape[0] > 0 else 0
        )
        _SECURE_INFER_REGISTER_HOST_RANGE_EX(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel() * tensor.element_size()),
            ctypes.c_uint32(layer_id),
            ctypes.c_uint32(0 if tensor_kind is None else tensor_kind),
            ctypes.c_size_t(block_bytes),
        )
        return
    if _SECURE_INFER_REGISTER_HOST_RANGE is None:
        return
    _SECURE_INFER_REGISTER_HOST_RANGE(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(tensor.numel() * tensor.element_size()),
    )


def _begin_d2h_job_batch(expected_tasks: int) -> bool:
    _lookup_secure_infer_symbols()
    if _SECURE_INFER_D2H_BATCH_BEGIN is None or _SECURE_INFER_D2H_BATCH_END is None:
        return False
    return _SECURE_INFER_D2H_BATCH_BEGIN(ctypes.c_size_t(expected_tasks)) > 0


def _begin_d2h_tensor_group(expected_tasks: int) -> None:
    if _SECURE_INFER_D2H_GROUP_BEGIN is None or _SECURE_INFER_D2H_GROUP_END is None:
        raise RuntimeError("SecureInfer D2H tensor-group symbols are unavailable")
    result = _SECURE_INFER_D2H_GROUP_BEGIN(ctypes.c_size_t(expected_tasks))
    if result <= 0:
        raise RuntimeError(f"SecureInfer D2H tensor-group begin failed: {result}")


def _end_d2h_tensor_group() -> None:
    if _SECURE_INFER_D2H_GROUP_END is None:
        raise RuntimeError("SecureInfer D2H tensor-group end symbol is unavailable")
    result = _SECURE_INFER_D2H_GROUP_END()
    if result != 0:
        raise RuntimeError(f"SecureInfer D2H tensor-group completion failed: {result}")


def _join_d2h_device_pipeline() -> None:
    if _SECURE_INFER_D2H_DEVICE_BARRIER is None:
        raise RuntimeError("SecureInfer D2H device-barrier symbol is unavailable")
    result = _SECURE_INFER_D2H_DEVICE_BARRIER()
    if result != 0:
        raise RuntimeError(f"SecureInfer D2H device barrier failed: {result}")


def _end_d2h_job_batch() -> None:
    if _SECURE_INFER_D2H_BATCH_END is None:
        return
    result = _SECURE_INFER_D2H_BATCH_END()
    if result != 0:
        raise RuntimeError(f"SecureInfer D2H batch completion failed: {result}")


def _retire_h2d_prediction_scope() -> None:
    """Retire speculative tickets without discarding repetitive history.

    A completed logical H2D job must release unused ticket buffers.  The
    repetitive predictor, however, learns across jobs, so its observation
    history is reset only by the full invalidation path (for example, after a
    prediction mismatch or an explicit new predictor session).  Keep the
    legacy invalidation fallback for older native libraries and for predictor
    modes that still use job-scoped state.
    """
    if (_SECURE_INFER_RETIRE_TRANSFER_PREFETCHES is not None and
            os.getenv("PIPE_SECURE_INFER_REPETITIVE_PREDICTOR", "0") != "0"):
        _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES()
    elif _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES is not None:
        _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES()


class SecureInferCpuNpuOffloadingHandler(CpuNpuOffloadingHandler):
    """CpuNpuOffloadingHandler with history-driven secure preparation."""

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        super().__init__(
            gpu_block_size=gpu_block_size,
            cpu_block_size=cpu_block_size,
            num_cpu_blocks=num_cpu_blocks,
            gpu_caches=gpu_caches,
            attn_backends=attn_backends,
        )

        for layer_index, (key_cache, value_cache) in enumerate(self.cpu_tensors):
            _register_host_range(key_cache, layer_index, 0)
            _register_host_range(value_cache, layer_index, 1)

        # Learn the cache geometry from the tensors actually constructed by
        # vLLM-Ascend.  Model configuration alone is insufficient under TP/PP
        # because KV heads and layers may be sharded or replicated per rank.
        self._kv_tensor_block_bytes: list[int] = []
        for key_cache, value_cache in self.cpu_tensors:
            for cache in (key_cache, value_cache):
                block_bytes = (
                    int(cache[0].numel() * cache.element_size())
                    if cache.shape[0] > 0
                    else 0
                )
                self._kv_tensor_block_bytes.append(block_bytes)
        size_histogram: dict[str, int] = {}
        for block_bytes in self._kv_tensor_block_bytes:
            size_histogram[str(block_bytes)] = (
                size_histogram.get(str(block_bytes), 0) + 1
            )
        self._rows_per_logical_block = (
            int(self.block_size_factor) * len(self._kv_tensor_block_bytes)
        )
        self._bytes_per_logical_block = int(self.block_size_factor) * sum(
            self._kv_tensor_block_bytes
        )
        logger.info(
            "SecureInfer learned KV transfer geometry: tensor_copies=%d "
            "block_size_factor=%d rows_per_logical_block=%d "
            "bytes_per_logical_block=%d size_histogram=%s",
            len(self._kv_tensor_block_bytes),
            self.block_size_factor,
            self._rows_per_logical_block,
            self._bytes_per_logical_block,
            size_histogram,
        )
        _append_transfer_geometry(
            {
                "version": 1,
                "event": "handler_geometry",
                "timestamp_ns": time.time_ns(),
                "pid": os.getpid(),
                "tensor_copies": len(self._kv_tensor_block_bytes),
                "block_size_factor": int(self.block_size_factor),
                "rows_per_logical_block": self._rows_per_logical_block,
                "bytes_per_logical_block": self._bytes_per_logical_block,
                "copy_size_histogram": size_histogram,
            }
        )

        # vLLM already keeps the source NPU blocks pinned until get_finished()
        # reports the store job.  What used to block the engine thread was the
        # *submission* of thousands of D2H slices in transfer_async().  Move
        # only that submission work to a bounded background thread; the NPU
        # event recorded below remains the cache-ready completion point.
        self._transfer_lock = threading.RLock()
        self._d2h_write_behind = _d2h_write_behind_enabled()
        self._d2h_two_phase = _d2h_two_phase_enabled()
        self._d2h_admission = _d2h_admission_enabled()
        self._d2h_max_jobs = _d2h_pending_job_limit()
        self._d2h_max_bytes = _d2h_pending_byte_limit()
        self._d2h_pending_bytes = 0
        self._d2h_submissions: dict[int, _D2HSubmission] = {}
        # The scheduler-side secure-store admission controller bounds the
        # actual KV bytes before a worker job is created.  Keep this submission
        # queue unbounded so queue.Full can never push submission back onto the
        # EngineCore thread.  In compatibility mode the legacy limits below
        # are still enforced.
        self._d2h_queue: queue.Queue[_D2HSubmission] = queue.Queue()
        self._d2h_queued_jobs = 0
        self._d2h_completed_jobs = 0
        self._d2h_sync_fallbacks = 0
        # D2H has two security-relevant completion points.  The device event
        # is recorded after encrypted DMA reaches host memory but before CPU
        # plaintext restoration.  transfer_events remains the cache-ready
        # event recorded after the CPU HostFunc.
        self.device_transfer_events: dict[int, torch.npu.Event] = {}
        # swap_blocks may copy its CPU block-map tensor to the device
        # asynchronously.  Keeping only a Python local is unsafe once the
        # next background job reuses the CPU allocator before DEVICE_COPY_DONE;
        # the device then observes arbitrary 64-bit block IDs.  Retain every
        # block-map tensor through the stronger cache-ready event.
        self._transfer_host_refs: dict[int, list[object]] = {}
        # These are deliberately separate from inter-request submit streams.
        # They stripe one source/destination-disjoint logical reload, then the
        # caller stream joins all of them before advertising the load ready.
        max_kv_tensor_block_bytes = max(self._kv_tensor_block_bytes, default=0)
        self._h2d_slice_stream_count = _gcm_h2d_slice_streams(
            max_kv_tensor_block_bytes
        )
        self._h2d_slice_streams: list[torch.npu.Stream] = [self.h2d_stream]
        if self._h2d_slice_stream_count > 1:
            try:
                self._h2d_slice_streams.extend(
                    torch.npu.Stream()
                    for _ in range(self._h2d_slice_stream_count - 1)
                )
            except BaseException:
                logger.exception(
                    "SecureInfer could not create intra-request H2D slice streams; "
                    "falling back to one stream"
                )
                self._h2d_slice_stream_count = 1
                self._h2d_slice_streams = [self.h2d_stream]

        # Wall-clock telemetry for request-atomic transfers.  Per-slice event
        # sums double-count concurrent streams; these job boundaries expose the
        # actual submission and completion time seen by the framework.
        self._h2d_job_timing_ns: dict[int, tuple[int, int]] = {}

        if self._d2h_write_behind:
            self._d2h_submitter = threading.Thread(
                target=self._d2h_submit_loop,
                name="secure_infer-d2h-submit",
                daemon=True,
            )
            self._d2h_submitter.start()
        else:
            self._d2h_submitter = None

        logger.info(
            "SecureInfer bounded D2H write-behind: enabled=%s two_phase=%s "
            "admission=%s max_jobs=%d max_bytes=%d",
            self._d2h_write_behind,
            self._d2h_two_phase,
            self._d2h_admission,
            self._d2h_max_jobs,
            self._d2h_max_bytes,
        )
        logger.info(
            "SecureInfer H2D preparation: native history predictor only; "
            "exact descriptor admission disabled"
        )
        logger.info(
            "SecureInfer authenticated H2D slice streams: %d "
            "(one request is joined before publication)",
            self._h2d_slice_stream_count,
        )

    def _estimate_d2h_bytes(self, spec: TransferSpec) -> int:
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, CPULoadStoreSpec)

        bytes_per_gpu_block = 0
        for key_cache, value_cache in self.npu_tensors:
            if key_cache.shape[0] == 0:
                continue
            bytes_per_gpu_block += key_cache[0].numel() * key_cache.element_size()
            bytes_per_gpu_block += value_cache[0].numel() * value_cache.element_size()
        return int(src_spec.block_ids.size) * bytes_per_gpu_block

    def _record_h2d_geometry(
        self,
        job_id: int,
        logical_block_count: int,
        physical_block_count: int,
    ) -> None:
        size_histogram: dict[str, int] = {}
        for block_bytes in self._kv_tensor_block_bytes:
            key = str(block_bytes)
            size_histogram[key] = (
                size_histogram.get(key, 0) + physical_block_count
            )
        _append_transfer_geometry(
            {
                "version": 1,
                "event": "h2d_job",
                "timestamp_ns": time.time_ns(),
                "pid": os.getpid(),
                "job_id": int(job_id),
                "logical_blocks": int(logical_block_count),
                "physical_blocks": int(physical_block_count),
                "tensor_copies": len(self._kv_tensor_block_bytes),
                "expected_rows": int(
                    physical_block_count * len(self._kv_tensor_block_bytes)
                ),
                "expected_bytes": int(
                    physical_block_count * sum(self._kv_tensor_block_bytes)
                ),
                "copy_size_histogram": size_histogram,
            }
        )

    def _queue_d2h_submission(self, job_id: int, spec: TransferSpec) -> bool:
        pending = _D2HSubmission(
            job_id=job_id,
            spec=spec,
            bytes=self._estimate_d2h_bytes(spec),
        )
        with self._transfer_lock:
            over_jobs = len(self._d2h_submissions) >= self._d2h_max_jobs
            over_bytes = self._d2h_pending_bytes + pending.bytes > self._d2h_max_bytes
            if (over_jobs or over_bytes) and not self._d2h_admission:
                self._d2h_sync_fallbacks += 1
                logger.warning(
                    "SecureInfer D2H write-behind pressure: job=%d bytes=%d "
                    "pending_jobs=%d pending_bytes=%d; using synchronous "
                    "submission for backpressure",
                    job_id,
                    pending.bytes,
                    len(self._d2h_submissions),
                    self._d2h_pending_bytes,
                )
                return False

            if over_jobs or over_bytes:
                # This is a diagnostic mismatch, not a reason to block the
                # request thread.  The scheduler has already admitted the
                # store; finish it asynchronously to preserve cache state.
                logger.warning(
                    "SecureInfer worker observed admission-accounting overflow: "
                    "job=%d bytes=%d pending_jobs=%d pending_bytes=%d; "
                    "keeping write-behind asynchronous",
                    job_id,
                    pending.bytes,
                    len(self._d2h_submissions),
                    self._d2h_pending_bytes,
                )

            self._d2h_submissions[job_id] = pending
            self._d2h_pending_bytes += pending.bytes
            self._d2h_queued_jobs += 1
            self._d2h_queue.put_nowait(pending)

        logger.info(
            "SecureInfer queued secure D2H store: job=%d bytes=%d pending_jobs=%d "
            "pending_bytes=%d",
            job_id,
            pending.bytes,
            len(self._d2h_submissions),
            self._d2h_pending_bytes,
        )
        return True

    def _d2h_submit_loop(self) -> None:
        while True:
            pending = self._d2h_queue.get()
            started_ns = time.monotonic_ns()
            try:
                self._submit_transfer(pending.job_id, pending.spec)
            except BaseException as exc:
                pending.error = exc
                logger.exception(
                    "SecureInfer background D2H submission failed for job=%d",
                    pending.job_id,
                )
            finally:
                finished_ns = time.monotonic_ns()
                logger.info(
                    "SecureInfer D2H background submission timing: job=%d "
                    "queue_delay_ms=%.3f submit_ms=%.3f",
                    pending.job_id,
                    (started_ns - pending.queued_ns) / 1e6,
                    (finished_ns - started_ns) / 1e6,
                )
                # wait() may be called by vLLM preemption before an NPU event
                # exists.  Publish either the event or the error first, then
                # release that waiter.
                pending.submitted.set()
                self._d2h_queue.task_done()

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        is_d2h = isinstance(src_spec, GPULoadStoreSpec)
        if is_d2h:
            assert isinstance(dst_spec, CPULoadStoreSpec)
            if self._d2h_write_behind and self._queue_d2h_submission(job_id, spec):
                return True
        # H2D demand loads must be submitted immediately.  D2H also reaches
        # this path when write-behind is disabled or its bounded queue applies
        # backpressure.
        return self._submit_transfer(job_id, spec)

    def _submit_transfer(
        self,
        job_id: int,
        spec: TransferSpec,
    ) -> bool:
        """Submit one ordinary request-atomic H2D or two-phase D2H job."""
        submit_started_ns = time.monotonic_ns()
        logger.info("start transfer submission: job=%d", job_id)
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.npu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
            is_h2d = True
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream = self.d2h_stream
            src_tensors = self.npu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor
            is_h2d = False

        if is_h2d:
            with self._transfer_lock:
                self._h2d_job_timing_ns.setdefault(
                    job_id, (submit_started_ns, 0)
                )

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1
        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor
        assert (
            src_sub_block_count
            == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip
        )

        src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])
        expand_block_ids(
            dst_blocks,
            dst_block_size_factor,
            src_to_dst[:, 1],
            skip_count=dst_sub_blocks_to_skip,
        )
        if is_h2d:
            self._record_h2d_geometry(
                job_id=job_id,
                logical_block_count=int(src_blocks.size),
                physical_block_count=int(src_to_dst.shape[0]),
            )
        src_to_dst_tensor = torch.from_numpy(src_to_dst)
        transfer_host_refs: list[object] = [src_to_dst_tensor]
        h2d_slice_streams = (
            self._h2d_slice_streams
            if is_h2d
            and self._h2d_slice_stream_count > 1
            else [stream]
        )

        with self._transfer_lock:
            event = self.events_pool.pop() if self.events_pool else torch.npu.Event()
            device_event = None
            if not is_h2d and self._d2h_two_phase:
                device_event = (
                    self.events_pool.pop()
                    if self.events_pool
                    else torch.npu.Event()
                )

        try:
            with torch.npu.stream(stream):
                if is_h2d:
                    for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
                        for src_cache, dst_cache in (
                            (src_tensor[0], dst_tensor[0]),
                            (src_tensor[1], dst_tensor[1]),
                        ):
                            if len(h2d_slice_streams) == 1:
                                torch.ops._C_ascend.swap_blocks(
                                    src_cache, dst_cache, src_to_dst_tensor
                                )
                                continue
                            total_rows = int(src_to_dst_tensor.shape[0])
                            stripe_count = min(
                                len(h2d_slice_streams), total_rows
                            )
                            for stripe_index in range(stripe_count):
                                begin = total_rows * stripe_index // stripe_count
                                end = total_rows * (stripe_index + 1) // stripe_count
                                mapping = src_to_dst_tensor[begin:end]
                                transfer_host_refs.append(mapping)
                                with torch.npu.stream(
                                    h2d_slice_streams[stripe_index]
                                ):
                                    torch.ops._C_ascend.swap_blocks(
                                        src_cache, dst_cache, mapping
                                    )
                    if len(h2d_slice_streams) > 1:
                        for slice_stream in h2d_slice_streams:
                            if slice_stream == stream:
                                continue
                            slice_done = torch.npu.Event()
                            slice_done.record(slice_stream)
                            transfer_host_refs.append(slice_done)
                            stream.wait_event(slice_done)
                else:
                    expected_tasks = (
                        int(src_to_dst.shape[0]) * len(src_tensors) * 2
                    )
                    d2h_batch_started = _begin_d2h_job_batch(expected_tasks)

                    def store_tensor_in_bounded_groups(
                        src_cache: torch.Tensor, dst_cache: torch.Tensor
                    ) -> None:
                        if not d2h_batch_started:
                            torch.ops._C_ascend.swap_blocks(
                                src_cache, dst_cache, src_to_dst_tensor
                            )
                            return
                        group_blocks = _d2h_group_blocks()
                        total_blocks = int(src_to_dst.shape[0])
                        for begin in range(0, total_blocks, group_blocks):
                            end = min(begin + group_blocks, total_blocks)
                            mapping = src_to_dst_tensor[begin:end]
                            _begin_d2h_tensor_group(end - begin)
                            try:
                                torch.ops._C_ascend.swap_blocks(
                                    src_cache, dst_cache, mapping
                                )
                            finally:
                                _end_d2h_tensor_group()

                    try:
                        for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
                            store_tensor_in_bounded_groups(
                                src_tensor[0], dst_tensor[0]
                            )
                            store_tensor_in_bounded_groups(
                                src_tensor[1], dst_tensor[1]
                            )
                    finally:
                        if d2h_batch_started:
                            _join_d2h_device_pipeline()
                            if device_event is not None:
                                device_event.record(stream)
                            _end_d2h_job_batch()
                        elif device_event is not None:
                            device_event.record(stream)

                event.record(stream)
        except BaseException:
            with self._transfer_lock:
                self.events_pool.append(event)
                if device_event is not None:
                    self.events_pool.append(device_event)
            raise

        if is_h2d:
            _retire_h2d_prediction_scope()

        with self._transfer_lock:
            self.transfer_events[job_id] = event
            self._transfer_host_refs[job_id] = transfer_host_refs
            if device_event is not None:
                self.device_transfer_events[job_id] = device_event
            if is_h2d:
                started_ns, _ = self._h2d_job_timing_ns.get(
                    job_id, (submit_started_ns, 0)
                )
                submitted_ns = time.monotonic_ns()
                self._h2d_job_timing_ns[job_id] = (
                    started_ns,
                    submitted_ns,
                )
                logger.info(
                    "SecureInfer H2D host submission: job=%d streams=%d "
                    "submit_ms=%.3f",
                    job_id,
                    len(h2d_slice_streams),
                    (submitted_ns - started_ns) / 1e6,
                )
        return True

    def get_device_finished(self) -> list[int]:
        """Return D2H jobs whose encrypted DMA no longer reads NPU KV.

        These jobs are source-safe, but their CPU cache entries may still be
        ciphertext.  Only the custom two-phase connector consumes this
        notification; get_finished() below remains the cache-ready boundary.
        """
        with self._transfer_lock:
            events = list(self.device_transfer_events.items())

        finished = [(job_id, event) for job_id, event in events if event.query()]
        job_ids: list[int] = []
        if not finished:
            return job_ids

        now_ns = time.monotonic_ns()
        with self._transfer_lock:
            for job_id, event in finished:
                if self.device_transfer_events.get(job_id) is not event:
                    continue
                del self.device_transfer_events[job_id]
                self.events_pool.append(event)
                pending = self._d2h_submissions.get(job_id)
                if pending is not None and not pending.device_done:
                    pending.device_done = True
                    pending.device_done_ns = now_ns
                    self._d2h_pending_bytes -= pending.bytes
                    logger.info(
                        "SecureInfer DEVICE_COPY_DONE: job=%d bytes=%d "
                        "observed_ms=%.3f pending_bytes=%d",
                        job_id,
                        pending.bytes,
                        (now_ns - pending.queued_ns) / 1e6,
                        self._d2h_pending_bytes,
                    )
                job_ids.append(job_id)
        return job_ids

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_h2d_job = False
        with self._transfer_lock:
            failed = [
                pending
                for pending in self._d2h_submissions.values()
                if pending.submitted.is_set() and pending.error is not None
            ]
            events = list(self.transfer_events.items())

        # vLLM's current offloading connector asserts that every accepted job
        # succeeds.  Fail fast here instead of silently leaking pinned blocks
        # or advertising an incomplete CPU cache entry.
        if failed:
            pending = failed[0]
            raise RuntimeError(
                f"SecureInfer secure transfer job {pending.job_id} failed during "
                "background submission"
            ) from pending.error

        finished_events = [(job_id, event) for job_id, event in events if event.query()]
        if finished_events:
            with self._transfer_lock:
                for job_id, event in finished_events:
                    # Ignore a stale snapshot if a future implementation ever
                    # reuses a job ID with a different event.
                    if self.transfer_events.get(job_id) is not event:
                        continue
                    del self.transfer_events[job_id]
                    self._transfer_host_refs.pop(job_id, None)
                    self.events_pool.append(event)
                    pending = self._d2h_submissions.pop(job_id, None)
                    if pending is not None:
                        if not pending.device_done:
                            self._d2h_pending_bytes -= pending.bytes
                        self._d2h_completed_jobs += 1
                        logger.info(
                            "SecureInfer SECURE_STORE_READY: job=%d "
                            "observed_ms=%.3f post_device_ms=%.3f",
                            job_id,
                            (time.monotonic_ns() - pending.queued_ns) / 1e6,
                            (
                                (time.monotonic_ns() - pending.device_done_ns) / 1e6
                                if pending.device_done_ns
                                else 0.0
                            ),
                        )
                    h2d_timing = self._h2d_job_timing_ns.pop(job_id, None)
                    if h2d_timing is not None:
                        finished_h2d_job = True
                        started_ns, submitted_ns = h2d_timing
                        now_ns = time.monotonic_ns()
                        logger.info(
                            "SecureInfer H2D job completion: job=%d "
                            "submit_ms=%.3f completion_ms=%.3f "
                            "post_submit_ms=%.3f",
                            job_id,
                            (
                                (submitted_ns - started_ns) / 1e6
                                if submitted_ns else 0.0
                            ),
                            (now_ns - started_ns) / 1e6,
                            (
                                (now_ns - submitted_ns) / 1e6
                                if submitted_ns else 0.0
                            ),
                        )
                    try:
                        result = TransferResult(
                            job_id=job_id,
                            success=True,
                            transfer_size=None,
                            transfer_time=None,
                            transfer_type=None,
                        )
                    except TypeError:
                        # vLLM 0.14 exposes TransferResult as tuple[int, bool].
                        result = (job_id, True)
                    results.append(result)

        if results:
            # event.query() above proves the framework-visible load/store
            # event complete.  Retire any final GCM ring slots now so exact
            # H2D accounting does not depend on a later request reusing them.
            if _SECURE_INFER_FLUSH_H2D_PROFILE is not None:
                _SECURE_INFER_FLUSH_H2D_PROFILE()
            if (
                finished_h2d_job
                and not self._h2d_job_timing_ns
                and _SECURE_INFER_DRAIN_H2D_PROFILE is not None
            ):
                _SECURE_INFER_DRAIN_H2D_PROFILE()
            _log_transfer_prefetch_stats()
            _log_npu_keystream_stats()
            _log_gcm_keystream_stats()
            _log_npu_resource_stats()
            logger.info(
                "SecureInfer D2H write-behind stats: queued=%d completed=%d "
                "sync_fallbacks=%d pending_jobs=%d pending_bytes=%d",
                self._d2h_queued_jobs,
                self._d2h_completed_jobs,
                self._d2h_sync_fallbacks,
                len(self._d2h_submissions),
                self._d2h_pending_bytes,
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Wait for queued submission and its cache-ready NPU event.

        Preemption can call wait() immediately after transfer_async().  A
        background D2H job may not have created its event yet, so first wait
        for event publication rather than treating the missing event as done.
        """
        for job_id in job_ids:
            with self._transfer_lock:
                pending = self._d2h_submissions.get(job_id)
            if pending is not None:
                pending.submitted.wait()
                if pending.error is not None:
                    raise RuntimeError(
                        f"SecureInfer secure transfer job {job_id} failed during "
                        "background submission"
                    ) from pending.error

            with self._transfer_lock:
                event = self.transfer_events.get(job_id)
            if event is not None:
                event.synchronize()


class NPUOffloadingSpec(_AscendNPUOffloadingSpec):
    """Drop-in replacement for vllm_ascend.kv_offload.npu.NPUOffloadingSpec."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, kv_cache_config)

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            configured_block_size = self.gpu_block_size
            if isinstance(configured_block_size, (list, tuple)):
                if len(configured_block_size) != 1:
                    raise ValueError(
                        "NPU offloading requires one GPU block size, "
                        f"got {configured_block_size!r}"
                    )
                gpu_block_size = configured_block_size[0]
            else:
                # vLLM 0.14/vllm-ascend may expose a scalar here.
                gpu_block_size = configured_block_size
            cpu_block_size = getattr(self, "offloaded_block_size", None)
            if cpu_block_size is None:
                cpu_block_size = gpu_block_size * getattr(
                    self, "block_size_factor", 1
                )
            self._handler = SecureInferCpuNpuOffloadingHandler(
                attn_backends=attn_backends,
                gpu_block_size=gpu_block_size,
                cpu_block_size=cpu_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
