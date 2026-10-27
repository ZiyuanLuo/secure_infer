"""SecureInfer-aware NPU KV offload module.

This module is intentionally outside vllm-ascend.  vLLM loads it through
kv_transfer_config.extra_config["spec_module_path"], so the framework source can
stay pristine while SecureInfer observes the host/device copies issued by the
runtime.  Cryptographic preparation is driven by the native history predictor;
this module does not submit an exact future block list to the crypto queues.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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
_SECURE_INFER_COUNT_READY_GCM_TICKETS = None
_SECURE_INFER_GET_NPU_RESOURCE_STATS = None
_SECURE_INFER_RETIRE_TRANSFER_PREFETCHES = None

# Opt-in event trace used by the layer-wise verification study.  The hot path
# is unchanged when the variable is unset; the file is append-only JSONL so a
# run can be analyzed even if the server exits after a failed sample.
_LAYER_TRACE_PATH = os.environ.get("PIPE_LAYERWISE_TRACE_PATH", "")
_LAYER_TRACE_DETAIL_ENABLED = (
    os.environ.get("PIPE_LAYERWISE_TRACE_DETAIL", "1") == "1"
)
_LAYER_TRACE_LOCK = threading.Lock()


def _layerwise_device_timing_enabled() -> bool:
    """Enable device-event timing only for the layerwise evaluation trace."""
    return os.getenv("PIPE_LAYERWISE_DEVICE_TIMING", "0") == "1"


def _layer_trace(kind: str, **fields: object) -> None:
    if not _LAYER_TRACE_PATH:
        return
    # Device-event summaries are sufficient for latency attribution.  Avoid
    # opening the trace file at every attention and submission boundary when
    # detailed host timestamps are not requested, since that instrumentation
    # can measurably perturb short inference runs.
    if not _LAYER_TRACE_DETAIL_ENABLED and kind != "device_layer_window":
        return
    record = {
        "kind": kind,
        "ts_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    try:
        with _LAYER_TRACE_LOCK:
            path = Path(_LAYER_TRACE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        # Instrumentation must never change serving semantics.
        return
_SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES = None
_SECURE_INFER_D2H_BATCH_BEGIN = None
_SECURE_INFER_D2H_GROUP_BEGIN = None
_SECURE_INFER_D2H_GROUP_END = None
_SECURE_INFER_D2H_DEVICE_BARRIER = None
_SECURE_INFER_D2H_BATCH_END = None
_SECURE_INFER_H2D_LAYER_GROUP_BEGIN = None
_SECURE_INFER_H2D_LAYER_GROUP_END = None
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


@dataclass
class _H2DSubmission:
    """A real H2D job tracked until its device event is complete."""

    job_id: int
    spec: TransferSpec
    queued_ns: int = field(default_factory=time.monotonic_ns)
    submitted: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


def _history_predict_enabled() -> bool:
    return (
        os.getenv("PIPE_PERIODIC_ADDRESS_PREDICTOR", "0") != "0"
        and os.getenv("PIPE_TRANSFER_PREFETCH", "1") != "0"
    )


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


def _gcm_h2d_layerwise_enabled() -> bool:
    """Enable the experimental verified-layer H2D pipeline.

    A normal offloading load is an atomic request-level operation: vLLM waits
    for every layer before dispatching the model.  This opt-in mode keeps the
    *cache publication* atomic, but exposes a device event after each
    layer's K and V tensors have completed the existing GCM
    verify-before-release path.  The attention hook can then make layer ``i``
    wait only for layer ``i`` while H2D authenticates later layers.

    Intra-layer striping is also safe: every stripe owns disjoint KV rows and
    the layer event joins all stripe streams after both K and V have passed
    GCM verification.  Thus attention can never observe a partially verified
    layer, while independent GHASH chains can use otherwise idle AICPU cores.
    """
    return os.getenv("PIPE_GCM_H2D_LAYERWISE", "0") == "1"


def _gcm_h2d_layerwise_initial_admission_layers() -> int:
    """Return the initial number of layer events to submit before layer 0.

    This is a host-command admission horizon, *not* an authentication
    horizon: layer 0 still waits for its own GCM-complete event, and every
    later layer retains the same verify-before-use event dependency.  The
    small initial lead prevents the foreground Python thread from immediately
    overtaking the background event submitter.
    """
    try:
        value = int(os.getenv(
            "PIPE_GCM_H2D_LAYERWISE_INITIAL_ADMISSION_LAYERS", "0"))
    except ValueError:
        value = 0
    return max(0, min(value, 64))


def _gcm_h2d_layer_lookahead() -> int:
    """Limit how far the layerwise submitter may run ahead of consumption.

    Zero preserves the original behavior (submit the complete logical H2D
    job).  A positive value keeps only that many future layers admitted past
    the foreground consumption point, creating a bounded preparation window.
    """
    try:
        value = int(os.getenv("PIPE_GCM_H2D_LAYER_LOOKAHEAD", "0"))
    except ValueError:
        value = 0
    # This is a command-admission horizon, not a security relaxation.  Every
    # layer maps to a GCM-complete event (or a conservative grouped event), and
    # attention waits for that event before first use.  Capping the horizon at eight forced a
    # 40-layer reload into several host-driven waves, leaving the authenticated
    # H2D streams under-filled for large reloads.  Permit a whole-model horizon
    # while retaining a defensive bound for malformed environment values.
    return max(0, min(value, 256))


def _gcm_h2d_layer_event_group_size() -> int:
    """Coarsen only the completion notification, not authentication.

    Layer 0 always keeps its own event so inference can start as soon as its
    KV is verified.  Later adjacent layers may share an event recorded after
    the whole group has completed GCM.  Waiting on that shared event is
    conservative: it can delay an already verified layer, but can never expose
    unverified KV.  The default of one preserves the original implementation.
    """
    try:
        value = int(os.getenv("PIPE_GCM_H2D_LAYER_EVENT_GROUP_SIZE", "1"))
    except ValueError:
        value = 1
    return max(1, min(value, 64))


def _layer_event_group_bounds(
    layer_index: int, layer_count: int, group_size: int
) -> tuple[int, int]:
    """Return the inclusive completion-event group for one layer."""
    if group_size <= 1 or layer_index == 0:
        return layer_index, layer_index
    begin = 1 + ((layer_index - 1) // group_size) * group_size
    return begin, min(layer_count - 1, begin + group_size - 1)


def _gcm_h2d_layerwise_eager_submission_enabled() -> bool:
    """Allow a bounded layerwise horizon to be submitted before layer 0.

    Normally the submitter pauses after recording the layer-0 event until the
    first attention hook has dynamically proved that it installed a
    verify-before-use dependency.  That is a useful fail-closed compatibility
    check, but it also means the foreground Python thread can outrun event
    admission for the next layers.  In the checked SealFlow connector path,
    the hook is part of the enabled execution stack, so this opt-in lets the
    submitter fill ``PIPE_GCM_H2D_LAYER_LOOKAHEAD`` before layer 0 runs.

    This changes *when commands are admitted*, not their security dependency:
    every layer still has a GCM-complete event (possibly shared with a later
    conservative group), and the attention stream still waits for that event
    before first use.  Keep it off by default for
    compatibility with connectors that have not installed the layer hook.
    """
    return os.getenv("PIPE_GCM_H2D_LAYERWISE_EAGER_SUBMISSION", "0") == "1"


def _gcm_h2d_layerwise_submit_workers() -> int:
    """Return independent host/ACL submit lanes for future layer events.

    This is intentionally separate from intra-layer striping.  Striping one
    layer improves GCM throughput, whereas these lanes make the *events for
    different future layers* visible quickly enough that foreground Python
    does not overtake the admission thread.  The layer tensors use disjoint
    CPU and NPU storage, so their authenticated H2D commands carry independent
    tickets and may be admitted on independent streams.  A layer still joins
    its K/V pair before its public release event is recorded.
    """
    try:
        value = int(os.getenv("PIPE_GCM_H2D_LAYERWISE_SUBMIT_WORKERS", "1"))
    except ValueError:
        value = 1
    return max(1, min(value, 4))


def _gcm_h2d_layer_round_robin_enabled() -> bool:
    """Share H2D command admission among active requests one layer at a time.

    The gate covers one layer's K/V pair, not its device execution.  It keeps
    commands for a layer contiguous on the shared ACL stream while preventing
    one request from admitting every layer before another request can admit
    layer 0.  The feature is opt-in until the multi-request evaluation is
    complete.
    """
    return os.getenv("PIPE_GCM_H2D_LAYER_ROUND_ROBIN", "0") == "1"


def _gcm_h2d_layer_round_robin_quantum() -> int:
    """Return the number of adjacent layers admitted per request turn."""
    try:
        value = int(os.getenv("PIPE_GCM_H2D_LAYER_ROUND_ROBIN_QUANTUM", "1"))
    except ValueError:
        value = 1
    return max(1, min(value, 16))


def _gcm_h2d_layer_urgency_enabled() -> bool:
    """Prioritize layer admission by proximity to foreground consumption."""
    return os.getenv("PIPE_GCM_H2D_LAYER_URGENCY", "0") == "1"


def _gcm_h2d_batch_layer_wave_enabled() -> bool:
    """Admit the same layer for every request before advancing the layer."""
    return os.getenv("PIPE_GCM_H2D_BATCH_LAYER_WAVE", "0") == "1"


def _gcm_h2d_max_submitted_layers() -> int:
    """Bound layer tasks already submitted to, but unfinished on, the NPU."""
    try:
        value = int(os.getenv("PIPE_GCM_H2D_MAX_SUBMITTED_LAYERS", "4"))
    except ValueError:
        value = 4
    return max(1, min(value, 64))


def _gcm_h2d_layerwise_host_admission_horizon() -> int:
    """Return the number of future layer events visible to the host.

    The model's Python front end can enqueue several Transformer layers much
    faster than the authenticated H2D submitter can record their events.  A
    small host-side admission horizon prevents that producer/consumer race.
    It waits only for command/event *recording*, never for GCM completion;
    the device event at each layer remains the sole verify-before-use gate.
    """
    try:
        value = int(os.getenv(
            "PIPE_GCM_H2D_LAYERWISE_HOST_ADMISSION_HORIZON", "0"))
    except ValueError:
        value = 0
    return max(0, min(value, 8))


def _gcm_h2d_slice_streams(block_bytes: int = 0) -> int:
    """Return the opt-in intra-request GCM H2D stream width.

    A single external-KV reload has many source/destination-disjoint physical
    KV slices.  Splitting those slices across independent streams lets Ascend
    schedule several AICPU GHASH tasks concurrently, while a join event keeps
    the framework-visible load completion atomic.  ``auto`` bases its width
    on the physical transfer geometry already available during connector
    initialization, rather than on a model name or a hard-coded layer count.
    """
    # Layer-wise verified release is the only path that can consume a layer
    # before the whole logical reload is finished.  It therefore benefits
    # directly from exposing independent physical KV slices to AICPU in
    # parallel.  Keep the legacy request-atomic path conservative, but make
    # the verified-layer path choose the already validated geometry-aware
    # width unless the deployment explicitly overrides it.
    default_width = "auto" if _gcm_h2d_layerwise_enabled() else "1"
    raw = os.getenv("PIPE_GCM_H2D_SLICE_STREAMS", default_width).strip().lower()
    if raw == "auto":
        # 256-KiB GQA slices are host-submission dominated once three streams
        # are active, so extra streams do not shorten end-to-end reload.
        # MiB-scale MHA slices instead leave each AICPU GHASH invocation long
        # enough to benefit from six independent chains.  This policy was
        # cross-checked with strict per-slice tag validation; it never changes
        # the IV/tag mapping or joins a layer before all its stripes finish.
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
    global _SECURE_INFER_COUNT_READY_GCM_TICKETS
    global _SECURE_INFER_GET_NPU_RESOURCE_STATS
    global _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES
    global _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES
    global _SECURE_INFER_D2H_BATCH_BEGIN
    global _SECURE_INFER_D2H_GROUP_BEGIN
    global _SECURE_INFER_D2H_GROUP_END
    global _SECURE_INFER_D2H_DEVICE_BARRIER
    global _SECURE_INFER_D2H_BATCH_END
    global _SECURE_INFER_H2D_LAYER_GROUP_BEGIN
    global _SECURE_INFER_H2D_LAYER_GROUP_END
    global _SECURE_INFER_FLUSH_H2D_PROFILE, _SECURE_INFER_DRAIN_H2D_PROFILE
    global _SECURE_INFER_SYMBOL_LOOKUP_DONE

    if _SECURE_INFER_SYMBOL_LOOKUP_DONE:
        return
    _SECURE_INFER_SYMBOL_LOOKUP_DONE = True

    try:
        process = ctypes.CDLL(None)
    except OSError:
        return

    try:
        register = process.secure_infer_register_host_range
        register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        register.restype = None
        _SECURE_INFER_REGISTER_HOST_RANGE = register
    except AttributeError:
        _SECURE_INFER_REGISTER_HOST_RANGE = None

    try:
        register_ex = process.secure_infer_register_host_range_ex
        register_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_size_t,
        ]
        register_ex.restype = None
        _SECURE_INFER_REGISTER_HOST_RANGE_EX = register_ex
    except AttributeError:
        _SECURE_INFER_REGISTER_HOST_RANGE_EX = None

    try:
        get_transfer_stats = process.secure_infer_get_transfer_prefetch_stats
        get_transfer_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        get_transfer_stats.restype = None
        _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS = get_transfer_stats
    except AttributeError:
        _SECURE_INFER_GET_TRANSFER_PREFETCH_STATS = None

    try:
        get_stats = process.secure_infer_get_npu_keystream_stats
        get_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        get_stats.restype = None
        _SECURE_INFER_GET_NPU_KEYSTREAM_STATS = get_stats
    except AttributeError:
        _SECURE_INFER_GET_NPU_KEYSTREAM_STATS = None

    try:
        get_gcm_stats = process.secure_infer_get_gcm_keystream_stats
        get_gcm_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 7
        get_gcm_stats.restype = None
        _SECURE_INFER_GET_GCM_KEYSTREAM_STATS = get_gcm_stats
    except AttributeError:
        _SECURE_INFER_GET_GCM_KEYSTREAM_STATS = None

    try:
        count_ready = process.secure_infer_count_ready_unconsumed_gcm_tickets
        count_ready.argtypes = []
        count_ready.restype = ctypes.c_uint64
        _SECURE_INFER_COUNT_READY_GCM_TICKETS = count_ready
    except AttributeError:
        _SECURE_INFER_COUNT_READY_GCM_TICKETS = None

    try:
        get_resource_stats = process.secure_infer_get_npu_resource_stats
        get_resource_stats.argtypes = [
            *([ctypes.POINTER(ctypes.c_int32)] * 5),
            *([ctypes.POINTER(ctypes.c_uint64)] * 7),
        ]
        get_resource_stats.restype = None
        _SECURE_INFER_GET_NPU_RESOURCE_STATS = get_resource_stats
    except AttributeError:
        _SECURE_INFER_GET_NPU_RESOURCE_STATS = None

    try:
        invalidate = process.secure_infer_invalidate_transfer_prefetches
        invalidate.argtypes = []
        invalidate.restype = None
        _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES = invalidate
    except AttributeError:
        _SECURE_INFER_INVALIDATE_TRANSFER_PREFETCHES = None

    try:
        retire = process.secure_infer_retire_transfer_prefetches
        retire.argtypes = []
        retire.restype = None
        _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES = retire
    except AttributeError:
        _SECURE_INFER_RETIRE_TRANSFER_PREFETCHES = None

    try:
        batch_begin = process.secure_infer_d2h_batch_begin
        batch_begin.argtypes = [ctypes.c_size_t]
        batch_begin.restype = ctypes.c_int
        _SECURE_INFER_D2H_BATCH_BEGIN = batch_begin
    except AttributeError:
        _SECURE_INFER_D2H_BATCH_BEGIN = None

    try:
        group_begin = process.secure_infer_d2h_group_begin
        group_begin.argtypes = [ctypes.c_size_t]
        group_begin.restype = ctypes.c_int
        _SECURE_INFER_D2H_GROUP_BEGIN = group_begin
    except AttributeError:
        _SECURE_INFER_D2H_GROUP_BEGIN = None

    try:
        group_end = process.secure_infer_d2h_group_end
        group_end.argtypes = []
        group_end.restype = ctypes.c_int
        _SECURE_INFER_D2H_GROUP_END = group_end
    except AttributeError:
        _SECURE_INFER_D2H_GROUP_END = None

    try:
        device_barrier = process.secure_infer_d2h_batch_device_barrier
        device_barrier.argtypes = []
        device_barrier.restype = ctypes.c_int
        _SECURE_INFER_D2H_DEVICE_BARRIER = device_barrier
    except AttributeError:
        _SECURE_INFER_D2H_DEVICE_BARRIER = None

    try:
        batch_end = process.secure_infer_d2h_batch_end
        batch_end.argtypes = []
        batch_end.restype = ctypes.c_int
        _SECURE_INFER_D2H_BATCH_END = batch_end
    except AttributeError:
        _SECURE_INFER_D2H_BATCH_END = None

    try:
        h2d_layer_group_begin = process.secure_infer_h2d_layer_group_begin
        h2d_layer_group_begin.argtypes = [ctypes.c_size_t]
        h2d_layer_group_begin.restype = ctypes.c_int
        _SECURE_INFER_H2D_LAYER_GROUP_BEGIN = h2d_layer_group_begin
    except AttributeError:
        _SECURE_INFER_H2D_LAYER_GROUP_BEGIN = None

    try:
        h2d_layer_group_end = process.secure_infer_h2d_layer_group_end
        h2d_layer_group_end.argtypes = []
        h2d_layer_group_end.restype = ctypes.c_int
        _SECURE_INFER_H2D_LAYER_GROUP_END = h2d_layer_group_end
    except AttributeError:
        _SECURE_INFER_H2D_LAYER_GROUP_END = None

    try:
        flush_h2d_profile = process.secure_infer_flush_completed_h2d_profile
        flush_h2d_profile.argtypes = []
        flush_h2d_profile.restype = None
        _SECURE_INFER_FLUSH_H2D_PROFILE = flush_h2d_profile
    except AttributeError:
        _SECURE_INFER_FLUSH_H2D_PROFILE = None

    try:
        drain_h2d_profile = process.secure_infer_drain_h2d_profile_after_job
        drain_h2d_profile.argtypes = []
        drain_h2d_profile.restype = None
        _SECURE_INFER_DRAIN_H2D_PROFILE = drain_h2d_profile
    except AttributeError:
        _SECURE_INFER_DRAIN_H2D_PROFILE = None


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


def _begin_h2d_layer_group(expected_tasks: int) -> bool:
    """Open one native authentication batch at a K/V layer boundary."""
    _lookup_secure_infer_symbols()
    if (_SECURE_INFER_H2D_LAYER_GROUP_BEGIN is None or
            _SECURE_INFER_H2D_LAYER_GROUP_END is None):
        return False
    result = _SECURE_INFER_H2D_LAYER_GROUP_BEGIN(ctypes.c_size_t(expected_tasks))
    if result != 0:
        raise RuntimeError(f"SecureInfer H2D layer-group begin failed: {result}")
    return True


def _end_h2d_layer_group(started: bool) -> None:
    """Flush authentication work before the layer release event is recorded."""
    if not started:
        return
    assert _SECURE_INFER_H2D_LAYER_GROUP_END is not None
    result = _SECURE_INFER_H2D_LAYER_GROUP_END()
    if result != 0:
        raise RuntimeError(
            f"SecureInfer H2D layer-group completion failed: {result}"
        )


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

        # The request-atomic baseline intentionally receives no layer-name or
        # execution-order mapping.  It only traverses the raw KV tensors
        # needed to submit the whole H2D transfer.  Retain framework layer
        # names *only* for the verified-layer path, where the attention hook
        # must bind a concrete layer to its authenticated-release event.
        # This keeps the serial baseline from using the information that
        # enables SealFlow's per-consumption ordering.
        self._h2d_layerwise = _gcm_h2d_layerwise_enabled()
        self._layer_names: tuple[str, ...] = ()
        if self._h2d_layerwise:
            self._layer_names = tuple(gpu_caches)
            if len(self._layer_names) != len(self.cpu_tensors):
                raise RuntimeError(
                    "SecureInfer layerwise H2D requires one CPU KV tensor pair per "
                    "registered attention layer"
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
        # job id -> (layer name -> event after K/V GCM verification and D2D
        # publication).  These events are intentionally not pooled: an
        # attention stream can retain an asynchronous wait even after the
        # host has observed the request-level completion event.
        self._layer_h2d_events: dict[int, dict[str, torch.npu.Event]] = {}
        # The foreground may reach a layer before the background submitter
        # has recorded that layer's event.  This condition waits only for
        # event *submission*; NPU execution remains asynchronous and is
        # gated by the ensuing stream event.
        self._layer_h2d_condition = threading.Condition(self._transfer_lock)
        self._layer_h2d_active_jobs: set[int] = set()
        # Lock-free read-mostly hint for the attention hook.  Secure loads are
        # activated at a scheduler boundary before model execution, and this
        # flag is set under the authoritative condition lock before the job is
        # published.  It avoids acquiring a Python condition 40 times for
        # every decode token after the one reload has already completed.
        self._layer_h2d_active_fast = False
        self._layer_h2d_recorded_layers: dict[int, set[str]] = {}
        # Runtime coverage for the verify-before-use hook.  A missing hook is
        # a security error, not merely a lost overlap opportunity, so the
        # submitter performs a one-time handshake after publishing layer 0.
        self._layer_h2d_waited_layers: dict[int, set[str]] = {}
        self._layer_h2d_consumed_events: dict[int, dict[str, torch.npu.Event]] = {}
        self._layer_h2d_threads: dict[int, threading.Thread] = {}
        # Layerwise jobs share the same ACL stream pool.  Keep each layer's
        # K/V command pair contiguous, but rotate command admission across
        # active requests at layer boundaries.  This removes request-sized
        # head-of-line blocking without adding a device synchronization.
        self._layer_h2d_job_submit_lock = threading.Lock()
        self._layer_h2d_layer_submit_lock = threading.Lock()
        self._layer_h2d_round_robin = _gcm_h2d_layer_round_robin_enabled()
        self._layer_h2d_round_robin_quantum = (
            _gcm_h2d_layer_round_robin_quantum()
        )
        self._layer_h2d_urgency = _gcm_h2d_layer_urgency_enabled()
        self._layer_h2d_batch_layer_wave = (
            _gcm_h2d_batch_layer_wave_enabled()
        )
        if sum((
            self._layer_h2d_round_robin,
            self._layer_h2d_urgency,
            self._layer_h2d_batch_layer_wave,
        )) > 1:
            raise ValueError(
                "round-robin, urgency, and batch-layer-wave H2D admission "
                "are mutually exclusive"
            )
        self._layer_h2d_max_submitted_layers = (
            _gcm_h2d_max_submitted_layers()
        )
        self._layer_h2d_admission_queue: list[int] = []
        self._layer_h2d_admission_owner: int | None = None
        self._layer_h2d_admission_owner_layer: int | None = None
        self._layer_h2d_admission_turn_layers = 0
        # Urgency scheduling deliberately leaves work in this host queue until
        # a bounded device-submission slot is available.  Once a layer's K/V
        # commands have been admitted to an ACL stream they cannot be
        # reordered, so bounding that irreversible prefix is what makes the
        # priority policy effective.
        self._layer_h2d_current_layer: dict[int, int] = {}
        self._layer_h2d_urgency_waiters: dict[tuple[int, int], int] = {}
        self._layer_h2d_urgency_sequence = 0
        self._layer_h2d_inflight: list[
            tuple[int, int, torch.npu.Event]
        ] = []
        # A worker metadata object describes the requests participating in
        # one model execution step.  Batch-layer-wave admission records that
        # membership before any submitter thread starts and advances all jobs
        # through layer i before admitting layer i+1.  The per-job submitters
        # remain unchanged in this first implementation; only their command
        # admission order is coordinated.
        self._layer_h2d_batch_sequence = 0
        self._layer_h2d_job_batch: dict[int, int] = {}
        self._layer_h2d_batch_jobs: dict[int, list[int]] = {}
        self._layer_h2d_batch_next_layer: dict[int, int] = {}
        # Restrict attention waits to the layerwise reload jobs belonging to
        # the current model execution step, rather than every globally active
        # background reload.
        self._layer_h2d_wait_jobs: set[int] = set()
        # These timing events are deliberately opt-in.  Host monotonic clocks
        # only measure Python submission, whereas this experiment needs the
        # device-time interval from a layer's verified-use gate to the next
        # one.  A record is retained until all four events have completed.
        self._layer_device_timing = _layerwise_device_timing_enabled()
        self._layer_device_windows: dict[int, dict[str, dict[str, torch.npu.Event]]] = {}
        self._h2d_submission_count = 0
        # Real H2D copies are submitted by the normal path below; the native
        # interceptor observes those copies and performs history-based
        # preparation.  There is no descriptor-admission worker.
        self._h2d_submissions: dict[int, _H2DSubmission] = {}
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

        # A bounded set of independent streams used only for layerwise H2D
        # event admission.  They are distinct from the regular per-layer
        # stripe pool: a layer is assigned to exactly one submit lane, so its
        # K/V pair has one ordered authenticated-release event.  Keeping the
        # parent h2d_stream free lets the coordinator join all lane events
        # into the request-level completion event without host synchronizing.
        self._layer_h2d_submit_worker_count = (
            _gcm_h2d_layerwise_submit_workers()
            if self._h2d_layerwise else 1
        )
        self._layer_h2d_event_group_size = (
            _gcm_h2d_layer_event_group_size() if self._h2d_layerwise else 1
        )
        if self._layer_h2d_urgency and self._layer_h2d_event_group_size != 1:
            raise ValueError(
                "urgency admission requires one completion event per layer"
            )
        if (self._layer_h2d_event_group_size > 1
                and self._layer_h2d_submit_worker_count > 1):
            raise ValueError(
                "grouped layer completion events require one ordered "
                "layerwise submitter"
            )
        self._layer_h2d_submit_streams: list[torch.npu.Stream] = []
        self._layer_h2d_submit_stream_locks: list[threading.Lock] = []
        if self._h2d_layerwise and self._layer_h2d_submit_worker_count > 1:
            try:
                self._layer_h2d_submit_streams = [
                    torch.npu.Stream()
                    for _ in range(self._layer_h2d_submit_worker_count)
                ]
                self._layer_h2d_submit_stream_locks = [
                    threading.Lock()
                    for _ in self._layer_h2d_submit_streams
                ]
            except BaseException:
                logger.exception(
                    "SecureInfer could not create independent layerwise submit "
                    "streams; falling back to one lane"
                )
                self._layer_h2d_submit_worker_count = 1
                self._layer_h2d_submit_stream_locks = []

        # Per-job policy arrives from scheduler metadata.  The global flag is
        # only a capability/configuration gate; using it directly for every
        # H2D job deadlocks atomic fallbacks because no attention hook is
        # expected while the request waits for request-level completion.
        self._layerwise_h2d_job_ids: set[int] = set()
        if self._h2d_layerwise:
            if not _history_predict_enabled():
                raise ValueError(
                    "PIPE_GCM_H2D_LAYERWISE=1 requires the native history "
                    "address predictor (PIPE_PERIODIC_ADDRESS_PREDICTOR=1)"
                )
            if not self._layer_names:
                raise ValueError(
                    "PIPE_GCM_H2D_LAYERWISE=1 requires attention KV caches"
                )
        self._h2d_queued_jobs = 0
        self._h2d_completed_jobs = 0
        # Wall-clock telemetry for both the stock request-atomic path and the
        # verified-layer path.  Per-slice event sums double-count concurrent
        # streams; these job boundaries expose the actual submission and
        # completion time seen by the framework.
        self._h2d_job_timing_ns: dict[int, tuple[int, int]] = {}
        self._h2d_commit_requests: dict[int, str] = {}

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
        logger.info(
            "SecureInfer verified-layer H2D pipeline: enabled=%s layers=%d "
            "(cache publication remains request-atomic)",
            self._h2d_layerwise,
            len(self._layer_names),
        )
        if self._h2d_layerwise:
            logger.info(
                "SecureInfer registered KV execution order: %s",
                ",".join(self._layer_names),
            )
            logger.info(
                "SecureInfer verified-layer admission: lookahead=%d eager=%s "
                "submit_lanes=%d round_robin=%s urgency=%s batch_wave=%s "
                "max_submitted_layers=%d",
                _gcm_h2d_layer_lookahead(),
                _gcm_h2d_layerwise_eager_submission_enabled(),
                self._layer_h2d_submit_worker_count,
                self._layer_h2d_round_robin,
                self._layer_h2d_urgency,
                self._layer_h2d_batch_layer_wave,
                self._layer_h2d_max_submitted_layers,
            )

    def set_h2d_job_layerwise(self, job_id: int, enabled: bool) -> None:
        """Install the scheduler's per-request H2D execution policy."""
        with self._transfer_lock:
            if enabled:
                if not self._h2d_layerwise:
                    raise RuntimeError(
                        "layerwise H2D job selected while the worker capability "
                        "is disabled"
                    )
                self._layerwise_h2d_job_ids.add(job_id)
            else:
                self._layerwise_h2d_job_ids.discard(job_id)

    def set_h2d_job_request(self, job_id: int, req_id: str) -> None:
        if os.getenv("PIPE_GCM_COMMIT_DIR"):
            self._h2d_commit_requests[job_id] = req_id

    def _write_commit_result(self, job_id: int, status: int) -> None:
        commit_dir = os.getenv("PIPE_GCM_COMMIT_DIR")
        req_id = self._h2d_commit_requests.pop(job_id, None)
        if not commit_dir or req_id is None:
            return
        directory = Path(commit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / f"{req_id}.json"
        temporary = directory / f"{req_id}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({
            "request_id": req_id,
            "job_id": job_id,
            "status": status,
            "realtime_ns": time.time_ns(),
        }), encoding="utf-8")
        os.replace(temporary, result)

    def set_h2d_batch_layerwise_jobs(self, job_ids: list[int]) -> None:
        """Set layerwise jobs consumed by the next model execution step."""
        with self._layer_h2d_condition:
            self._layer_h2d_wait_jobs = set(job_ids)
            if not self._layer_h2d_batch_layer_wave or not job_ids:
                return
            self._layer_h2d_batch_sequence += 1
            batch_id = self._layer_h2d_batch_sequence
            ordered_jobs = list(job_ids)
            self._layer_h2d_batch_jobs[batch_id] = ordered_jobs
            for job_id in ordered_jobs:
                if job_id in self._layer_h2d_job_batch:
                    raise RuntimeError(
                        f"layerwise H2D job {job_id} already belongs to a batch"
                    )
                self._layer_h2d_job_batch[job_id] = batch_id
                self._layer_h2d_batch_next_layer[job_id] = 0
            logger.info(
                "SecureInfer batch-layer wave registered: batch=%d jobs=%s",
                batch_id,
                ordered_jobs,
            )
            self._layer_h2d_condition.notify_all()

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
        elif job_id in self._layerwise_h2d_job_ids:
            # The stock request-atomic path submits every layer on the
            # EngineCore thread before model execution.  That host-side loop
            # would turn per-layer events into an effective global barrier.
            # Submit this one real reload in the background instead: the
            # first verified K/V pair releases layer 0, while later pairs are
            # enqueued concurrently with earlier foreground layers.
            assert isinstance(src_spec, CPULoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            pending = _H2DSubmission(job_id=job_id, spec=spec)
            with self._layer_h2d_condition:
                if job_id in self._h2d_submissions:
                    raise RuntimeError(f"duplicate SecureInfer layerwise H2D job {job_id}")
                self._h2d_submissions[job_id] = pending
                self._layer_h2d_active_jobs.add(job_id)
                if (self._layer_h2d_round_robin or self._layer_h2d_urgency
                        or self._layer_h2d_batch_layer_wave):
                    self._layer_h2d_admission_queue.append(job_id)
                    self._layer_h2d_current_layer[job_id] = -1
                    self._layer_h2d_condition.notify_all()
                self._layer_h2d_active_fast = True
                from .attention_plugin import mark_layerwise_wait_active
                mark_layerwise_wait_active(self, True)
                submitter = threading.Thread(
                    target=self._layerwise_h2d_submit_loop,
                    args=(pending,),
                    name=f"secure_infer-layerwise-h2d-{job_id}",
                    daemon=True,
                )
                self._layer_h2d_threads[job_id] = submitter
                submitter.start()
            logger.info(
                "SecureInfer layerwise H2D submission started in background: job=%d",
                job_id,
            )
            return True
        # H2D demand loads must be submitted immediately.  D2H also reaches
        # this path when write-behind is disabled or its bounded queue applies
        # backpressure.
        return self._submit_transfer(job_id, spec)

    def transfer_layerwise_batch_async(
        self, jobs: list[tuple[int, TransferSpec]]
    ) -> bool:
        """Submit one batch of reloads as layer ordered authentication waves.

        The CPU and device KV pools are shared tensors.  Consequently, block
        mappings from requests in the same model execution step can be joined
        and passed to one swap_blocks call per K/V tensor and layer.  This
        preserves each request's source and destination rows while replacing
        per-request launches and completion events with one event per layer.
        """
        if not self._layer_h2d_batch_layer_wave or len(jobs) < 2:
            return False
        if os.getenv("PIPE_GCM_COMMIT_DIR"):
            raise RuntimeError("eager commit experiment requires individual H2D jobs")
        pendings: list[_H2DSubmission] = []
        with self._layer_h2d_condition:
            for job_id, spec in jobs:
                if job_id not in self._layerwise_h2d_job_ids:
                    raise RuntimeError(
                        f"batch-layer wave received atomic H2D job {job_id}"
                    )
                if job_id in self._h2d_submissions:
                    raise RuntimeError(
                        f"duplicate SecureInfer layerwise H2D job {job_id}"
                    )
                pending = _H2DSubmission(job_id=job_id, spec=spec)
                self._h2d_submissions[job_id] = pending
                self._layer_h2d_active_jobs.add(job_id)
                self._layer_h2d_admission_queue.append(job_id)
                self._layer_h2d_current_layer[job_id] = -1
                pendings.append(pending)
            self._layer_h2d_active_fast = True
            from .attention_plugin import mark_layerwise_wait_active
            mark_layerwise_wait_active(self, True)
            coordinator = threading.Thread(
                target=self._layerwise_h2d_batch_submit_loop,
                args=(pendings,),
                name="secure_infer-layerwise-h2d-batch",
                daemon=True,
            )
            for pending in pendings:
                self._layer_h2d_threads[pending.job_id] = coordinator
            coordinator.start()
            self._layer_h2d_condition.notify_all()
        logger.info(
            "SecureInfer layerwise H2D batch submission started: jobs=%s",
            [pending.job_id for pending in pendings],
        )
        return True

    def _layerwise_h2d_batch_submit_loop(
        self, pendings: list[_H2DSubmission]
    ) -> None:
        started_ns = time.monotonic_ns()
        try:
            self._submit_layerwise_h2d_batch(pendings)
        except BaseException as exc:
            for pending in pendings:
                pending.error = exc
            logger.exception(
                "SecureInfer layerwise H2D batch submission failed for jobs=%s",
                [pending.job_id for pending in pendings],
            )
        finally:
            for pending in pendings:
                self._finish_layerwise_h2d_admission(pending.job_id)
                pending.submitted.set()
            with self._layer_h2d_condition:
                self._layer_h2d_condition.notify_all()
            logger.info(
                "SecureInfer layerwise H2D batch submission finished: jobs=%s "
                "submit_ms=%.3f",
                [pending.job_id for pending in pendings],
                (time.monotonic_ns() - started_ns) / 1e6,
            )

    def _submit_layerwise_h2d_batch(
        self, pendings: list[_H2DSubmission]
    ) -> None:
        """Join request mappings and authenticate one complete layer wave."""
        submit_started_ns = time.monotonic_ns()
        mappings: list[torch.Tensor] = []
        logical_counts: dict[int, int] = {}
        for pending in pendings:
            src_spec, dst_spec = pending.spec
            if not isinstance(src_spec, CPULoadStoreSpec):
                raise RuntimeError("batch-layer wave requires CPU H2D sources")
            if not isinstance(dst_spec, GPULoadStoreSpec):
                raise RuntimeError("batch-layer wave requires device destinations")
            src_blocks = src_spec.block_ids
            dst_blocks = dst_spec.block_ids
            src_count = src_blocks.size * self.block_size_factor
            if src_count != dst_blocks.size:
                raise RuntimeError("invalid batch-layer H2D block mapping")
            mapping_np = np.empty((src_count, 2), dtype=np.int64)
            expand_block_ids(src_blocks, self.block_size_factor, mapping_np[:, 0])
            expand_block_ids(dst_blocks, 1, mapping_np[:, 1])
            mapping = torch.from_numpy(mapping_np)
            mappings.append(mapping)
            logical_counts[pending.job_id] = int(src_blocks.size)
            self._record_h2d_geometry(
                job_id=pending.job_id,
                logical_block_count=int(src_blocks.size),
                physical_block_count=int(mapping.shape[0]),
            )
        combined_mapping = torch.cat(mappings, dim=0)

        stream = self.h2d_stream
        with self._transfer_lock:
            final_events = {
                pending.job_id: (
                    self.events_pool.pop()
                    if self.events_pool else torch.npu.Event()
                )
                for pending in pendings
            }
            for pending in pendings:
                self._h2d_job_timing_ns.setdefault(
                    pending.job_id, (submit_started_ns, 0)
                )

        shared_layer_events = {
            layer_name: torch.npu.Event(enable_timing=self._layer_device_timing)
            for layer_name in self._layer_names
        }
        with self._layer_h2d_condition:
            for pending in pendings:
                job_id = pending.job_id
                self._layer_h2d_events[job_id] = shared_layer_events
                self._layer_h2d_recorded_layers[job_id] = set()
                self._layer_h2d_waited_layers.setdefault(job_id, set())
                if self._layer_device_timing:
                    self._layer_device_windows[job_id] = {
                        layer_name: {"verify_ready": event}
                        for layer_name, event in shared_layer_events.items()
                    }
            self._layer_h2d_condition.notify_all()

        with torch.npu.stream(stream):
            for layer_index, (
                layer_name, (src_cache, dst_cache)
            ) in enumerate(zip(self._layer_names, zip(self.cpu_tensors,
                                                       self.npu_tensors))):
                if self._layer_device_timing:
                    verify_start = torch.npu.Event(enable_timing=True)
                    verify_start.record(stream)
                    with self._layer_h2d_condition:
                        for pending in pendings:
                            self._layer_device_windows[pending.job_id][layer_name][
                                "verify_start"
                            ] = verify_start
                _layer_trace(
                    "batch_verification_begin",
                    layer_name=layer_name,
                    layer_index=layer_index,
                    jobs=len(pendings),
                )
                h2d_group_started = _begin_h2d_layer_group(
                    2 * int(combined_mapping.shape[0])
                )
                try:
                    torch.ops._C_ascend.swap_blocks(
                        src_cache[0], dst_cache[0], combined_mapping
                    )
                    torch.ops._C_ascend.swap_blocks(
                        src_cache[1], dst_cache[1], combined_mapping
                    )
                finally:
                    # The native batch is flushed on this same stream before
                    # the public layer event below.  Authentication therefore
                    # remains a strict predecessor of attention.
                    _end_h2d_layer_group(h2d_group_started)
                shared_layer_events[layer_name].record(stream)
                with self._layer_h2d_condition:
                    for pending in pendings:
                        self._layer_h2d_recorded_layers[pending.job_id].add(
                            layer_name
                        )
                    self._layer_h2d_condition.notify_all()
                _layer_trace(
                    "batch_verification_ready",
                    layer_name=layer_name,
                    layer_index=layer_index,
                    jobs=len(pendings),
                )
                if layer_index == 0 and not (
                    _gcm_h2d_layerwise_eager_submission_enabled()
                ):
                    for pending in pendings:
                        self._wait_for_layerwise_hook_handshake(pending.job_id)
            for final_event in final_events.values():
                final_event.record(stream)

        _retire_h2d_prediction_scope()

        submitted_ns = time.monotonic_ns()
        retained: list[object] = [combined_mapping, *mappings]
        retained.extend(shared_layer_events.values())
        with self._transfer_lock:
            for pending in pendings:
                job_id = pending.job_id
                self.transfer_events[job_id] = final_events[job_id]
                self._transfer_host_refs[job_id] = retained
                started_ns, _ = self._h2d_job_timing_ns[job_id]
                self._h2d_job_timing_ns[job_id] = (started_ns, submitted_ns)
                logger.info(
                    "SecureInfer H2D batch host submission: job=%d batch_jobs=%d "
                    "logical_blocks=%d submit_ms=%.3f",
                    job_id,
                    len(pendings),
                    logical_counts[job_id],
                    (submitted_ns - started_ns) / 1e6,
                )

    def _layerwise_h2d_submit_loop(self, pending: _H2DSubmission) -> None:
        """Submit one real layerwise H2D request without a host barrier."""
        started_ns = time.monotonic_ns()
        commit_mode = bool(os.getenv("PIPE_GCM_COMMIT_DIR"))
        native = ctypes.CDLL(None) if commit_mode else None
        if native is not None:
            native.secure_infer_recovery_set_job_id.argtypes = [ctypes.c_uint64]
            native.secure_infer_recovery_set_job_id(pending.job_id)
        try:
            if (self._layer_h2d_round_robin or self._layer_h2d_urgency
                    or self._layer_h2d_batch_layer_wave):
                self._submit_transfer(pending.job_id, pending.spec)
            else:
                # Preserve the established request-at-a-time admission path
                # as the controlled baseline for this experiment.
                with self._layer_h2d_job_submit_lock:
                    self._submit_transfer(pending.job_id, pending.spec)
            if native is not None:
                wait_auth = native.secure_infer_recovery_wait_job_authentication
                wait_auth.argtypes = [ctypes.c_uint64]
                wait_auth.restype = ctypes.c_int
                status = wait_auth(pending.job_id)
                self._write_commit_result(pending.job_id, status)
                if status < 0:
                    raise RuntimeError("cannot observe eager H2D authentication")
        except BaseException as exc:
            if commit_mode and pending.job_id in self._h2d_commit_requests:
                self._write_commit_result(pending.job_id, -1)
            pending.error = exc
            logger.exception(
                "SecureInfer layerwise H2D submission failed for job=%d",
                pending.job_id,
            )
        finally:
            if native is not None:
                native.secure_infer_recovery_set_job_id(0)
            self._finish_layerwise_h2d_admission(pending.job_id)
            pending.submitted.set()
            with self._layer_h2d_condition:
                self._layer_h2d_condition.notify_all()
            logger.info(
                "SecureInfer layerwise H2D submission finished: job=%d submit_ms=%.3f",
                pending.job_id,
                (time.monotonic_ns() - started_ns) / 1e6,
            )

    def _reap_layerwise_h2d_inflight_locked(self) -> None:
        """Drop device-submission slots whose layer completion event fired."""
        self._layer_h2d_inflight = [
            item for item in self._layer_h2d_inflight if not item[2].query()
        ]

    def _begin_layerwise_h2d_layer(
        self, job_id: int, layer_index: int
    ) -> None:
        """Acquire one command-admission turn for a layer K/V pair."""
        if (not self._layer_h2d_round_robin
                and not self._layer_h2d_urgency
                and not self._layer_h2d_batch_layer_wave):
            return
        with self._layer_h2d_condition:
            waiter = (job_id, layer_index)
            if self._layer_h2d_urgency:
                self._layer_h2d_urgency_sequence += 1
                self._layer_h2d_urgency_waiters[waiter] = (
                    self._layer_h2d_urgency_sequence
                )
            while True:
                if job_id not in self._layer_h2d_admission_queue:
                    self._layer_h2d_urgency_waiters.pop(waiter, None)
                    raise RuntimeError(
                        f"missing layerwise H2D admission job {job_id}"
                    )
                can_admit = False
                if self._layer_h2d_batch_layer_wave:
                    batch_id = self._layer_h2d_job_batch.get(job_id)
                    batch_jobs = self._layer_h2d_batch_jobs.get(batch_id, [])
                    candidates = [
                        queued_job
                        for queued_job in batch_jobs
                        if queued_job in self._layer_h2d_admission_queue
                    ]
                    selected = min(
                        candidates,
                        key=lambda queued_job: (
                            self._layer_h2d_batch_next_layer.get(
                                queued_job, len(self._layer_names)
                            ),
                            batch_jobs.index(queued_job),
                        ),
                        default=None,
                    )
                    can_admit = (
                        self._layer_h2d_admission_owner is None
                        and selected == job_id
                        and self._layer_h2d_batch_next_layer.get(job_id)
                        == layer_index
                    )
                elif self._layer_h2d_urgency:
                    self._reap_layerwise_h2d_inflight_locked()

                    def priority(item: tuple[int, int]) -> tuple[int, int]:
                        queued_job, queued_layer = item
                        current = self._layer_h2d_current_layer.get(
                            queued_job, -1
                        )
                        distance = max(0, queued_layer - current)
                        return (
                            distance,
                            self._layer_h2d_urgency_waiters[item],
                        )

                    best = min(
                        self._layer_h2d_urgency_waiters,
                        key=priority,
                        default=None,
                    )
                    can_admit = (
                        self._layer_h2d_admission_owner is None
                        and best == waiter
                        and len(self._layer_h2d_inflight)
                        < self._layer_h2d_max_submitted_layers
                    )
                else:
                    can_admit = (
                        self._layer_h2d_admission_owner is None
                        and self._layer_h2d_admission_queue[0] == job_id
                    )
                if can_admit:
                    self._layer_h2d_admission_owner = job_id
                    self._layer_h2d_admission_owner_layer = layer_index
                    break
                # Device completion does not notify a Python condition, so an
                # urgency waiter polls at a short interval to reclaim slots.
                self._layer_h2d_condition.wait(
                    0.0002 if self._layer_h2d_urgency else None
                )
        self._layer_h2d_layer_submit_lock.acquire()

    def _end_layerwise_h2d_layer(
        self,
        job_id: int,
        layer_index: int,
        completion_event: torch.npu.Event | None = None,
    ) -> None:
        """Release a layer turn after its completion boundary is recorded."""
        if (not self._layer_h2d_round_robin
                and not self._layer_h2d_urgency
                and not self._layer_h2d_batch_layer_wave):
            return
        self._layer_h2d_layer_submit_lock.release()
        with self._layer_h2d_condition:
            if (
                self._layer_h2d_admission_owner != job_id
                or self._layer_h2d_admission_owner_layer != layer_index
            ):
                raise RuntimeError(
                    "layerwise H2D admission owner mismatch for "
                    f"job={job_id} layer={layer_index}"
                )
            self._layer_h2d_admission_owner = None
            self._layer_h2d_admission_owner_layer = None
            if self._layer_h2d_batch_layer_wave:
                expected_layer = self._layer_h2d_batch_next_layer.get(job_id)
                if expected_layer != layer_index:
                    raise RuntimeError(
                        "batch-layer wave advanced out of order for "
                        f"job={job_id}: expected={expected_layer} "
                        f"completed={layer_index}"
                    )
                self._layer_h2d_batch_next_layer[job_id] = layer_index + 1
                _layer_trace(
                    "batch_wave_admit",
                    job_id=job_id,
                    layer_index=layer_index,
                )
            elif self._layer_h2d_urgency:
                self._layer_h2d_urgency_waiters.pop(
                    (job_id, layer_index), None
                )
                if completion_event is not None:
                    self._layer_h2d_inflight.append(
                        (job_id, layer_index, completion_event)
                    )
                _layer_trace(
                    "urgency_admit",
                    job_id=job_id,
                    layer_index=layer_index,
                    current_layer=self._layer_h2d_current_layer.get(job_id, -1),
                    inflight=len(self._layer_h2d_inflight),
                )
            elif self._layer_h2d_admission_queue:
                if self._layer_h2d_admission_queue[0] != job_id:
                    raise RuntimeError(
                        f"layerwise H2D admission queue mismatch for job {job_id}"
                    )
                self._layer_h2d_admission_turn_layers += 1
                if (
                    self._layer_h2d_admission_turn_layers
                    >= self._layer_h2d_round_robin_quantum
                ):
                    self._layer_h2d_admission_queue.append(
                        self._layer_h2d_admission_queue.pop(0)
                    )
                    self._layer_h2d_admission_turn_layers = 0
            self._layer_h2d_condition.notify_all()

    def _finish_layerwise_h2d_admission(self, job_id: int) -> None:
        """Remove a job after all of its layer commands have been admitted."""
        if (not self._layer_h2d_round_robin
                and not self._layer_h2d_urgency
                and not self._layer_h2d_batch_layer_wave):
            return
        with self._layer_h2d_condition:
            if self._layer_h2d_admission_owner == job_id:
                # An exception should normally release the layer in the local
                # finally block.  Keep this fail-safe so one failed request
                # cannot strand all other requests behind its queue entry.
                self._layer_h2d_admission_owner = None
                self._layer_h2d_admission_owner_layer = None
                if self._layer_h2d_layer_submit_lock.locked():
                    self._layer_h2d_layer_submit_lock.release()
            self._layer_h2d_urgency_waiters = {
                key: value
                for key, value in self._layer_h2d_urgency_waiters.items()
                if key[0] != job_id
            }
            was_front = bool(
                self._layer_h2d_admission_queue
                and self._layer_h2d_admission_queue[0] == job_id
            )
            try:
                self._layer_h2d_admission_queue.remove(job_id)
            except ValueError:
                pass
            batch_id = self._layer_h2d_job_batch.pop(job_id, None)
            self._layer_h2d_batch_next_layer.pop(job_id, None)
            if batch_id is not None:
                batch_jobs = self._layer_h2d_batch_jobs.get(batch_id, [])
                if not any(
                    queued_job in self._layer_h2d_job_batch
                    for queued_job in batch_jobs
                ):
                    self._layer_h2d_batch_jobs.pop(batch_id, None)
            if was_front or not self._layer_h2d_admission_queue:
                self._layer_h2d_admission_turn_layers = 0
            self._layer_h2d_condition.notify_all()

    def _submit_one_layerwise_h2d_pair(
        self,
        *,
        job_id: int,
        layer_index: int,
        src_key_cache: torch.Tensor,
        src_value_cache: torch.Tensor,
        dst_key_cache: torch.Tensor,
        dst_value_cache: torch.Tensor,
        src_to_dst_tensor: torch.Tensor,
        layer_event: torch.npu.Event,
        stream: torch.npu.Stream,
        stream_lock: threading.Lock,
    ) -> None:
        """Admit one layer's K/V pair and publish its verified-release event.

        Every task owns a distinct layer's cache tensors and a distinct ACL
        stream.  This only parallelizes host command admission for independent
        future layers; authentication, the per-ticket IV/tag mapping, and the
        layer's final event dependency are unchanged.
        """
        layer_name = self._layer_names[layer_index]
        # ThreadPoolExecutor schedules tasks by completion order, not lane
        # index.  Serialize host calls *within* one ACL stream while retaining
        # concurrent submission across distinct streams.  Across requests,
        # admit one complete K/V pair per turn.
        self._begin_layerwise_h2d_layer(job_id, layer_index)
        admission_completion_event: torch.npu.Event | None = None
        try:
            with stream_lock:
                with torch.npu.stream(stream):
                    if self._layer_device_timing:
                        verify_start = torch.npu.Event(enable_timing=True)
                        verify_start.record(stream)
                        with self._layer_h2d_condition:
                            self._layer_device_windows[job_id][layer_name][
                                "verify_start"
                            ] = verify_start
                    _layer_trace(
                        "verification_begin",
                        job_id=job_id,
                        layer_name=layer_name,
                        layer_index=layer_index,
                    )
                    h2d_group_started = _begin_h2d_layer_group(
                        2 * int(src_to_dst_tensor.shape[0])
                    )
                    commit_native = (
                        ctypes.CDLL(None) if os.getenv("PIPE_GCM_COMMIT_DIR")
                        else None
                    )
                    if commit_native is not None:
                        commit_native.secure_infer_recovery_set_job_id.argtypes = [
                            ctypes.c_uint64
                        ]
                        commit_native.secure_infer_recovery_set_job_id(job_id)
                    try:
                        torch.ops._C_ascend.swap_blocks(
                            src_key_cache, dst_key_cache, src_to_dst_tensor
                        )
                        torch.ops._C_ascend.swap_blocks(
                            src_value_cache, dst_value_cache, src_to_dst_tensor
                        )
                    finally:
                        if commit_native is not None:
                            commit_native.secure_infer_recovery_set_job_id(0)
                        _end_h2d_layer_group(h2d_group_started)
                    # This event is after both K/V authenticated publication
                    # calls on the same lane.  A consumer can therefore never
                    # observe a half-verified layer.
                    layer_event.record(stream)
                    admission_completion_event = layer_event
        finally:
            self._end_layerwise_h2d_layer(
                job_id, layer_index, admission_completion_event
            )
        _layer_trace(
            "verification_ready",
            job_id=job_id,
            layer_name=layer_name,
            layer_index=layer_index,
        )
        with self._layer_h2d_condition:
            self._layer_h2d_recorded_layers[job_id].add(layer_name)
            self._layer_h2d_condition.notify_all()

    def _wait_for_layerwise_hook_handshake(self, job_id: int) -> None:
        """Preserve the default fail-closed hook check before future release."""
        deadline = time.monotonic() + 10.0
        with self._layer_h2d_condition:
            while not self._layer_h2d_waited_layers[job_id]:
                pending = self._h2d_submissions.get(job_id)
                if pending is not None and pending.error is not None:
                    raise RuntimeError(
                        "SecureInfer verified-layer H2D submission failed before "
                        "the attention-layer hook handshake"
                    ) from pending.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "SecureInfer verified-layer H2D did not observe an "
                        "attention-layer wait hook within 10 seconds"
                    )
                self._layer_h2d_condition.wait(remaining)

    def _wait_for_layerwise_lookahead_slot(
        self, job_id: int, predecessor_index: int
    ) -> None:
        """Keep a bounded number of future layer events admitted."""
        predecessor = self._layer_names[predecessor_index]
        while True:
            with self._layer_h2d_condition:
                consume_event = self._layer_h2d_consumed_events.get(
                    job_id, {}
                ).get(predecessor)
                pending = self._h2d_submissions.get(job_id)
                if pending is not None and pending.error is not None:
                    raise RuntimeError(
                        "SecureInfer bounded layerwise H2D submission failed "
                        f"before layer {predecessor_index + 1}"
                    ) from pending.error
            if consume_event is not None and consume_event.query():
                return
            time.sleep(0.0002)

    def _submit_layerwise_h2d_pairs_parallel(
        self,
        *,
        job_id: int,
        flat_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        src_to_dst_tensor: torch.Tensor,
        layer_events: dict[str, torch.npu.Event],
        parent_stream: torch.npu.Stream,
    ) -> None:
        """Fill and maintain a future-layer authenticated-H2D horizon.

        Several independent layer pairs are submitted concurrently, then the
        coordinator replenishes the horizon only after the foreground has
        crossed its oldest layer gate.  Each real copy is observed by the
        native history predictor; no future block list is registered here.
        """
        layer_count = len(self._layer_names)
        if len(flat_pairs) != 2 * layer_count:
            raise RuntimeError("invalid layerwise K/V pair count")
        horizon = _gcm_h2d_layer_lookahead()
        if horizon <= 0:
            horizon = layer_count
        horizon = min(horizon, layer_count)
        lanes = self._layer_h2d_submit_streams
        if len(lanes) <= 1:
            raise RuntimeError("parallel layerwise submission requested without lanes")

        def submit(executor: ThreadPoolExecutor, layer_index: int):
            src_key, dst_key = flat_pairs[2 * layer_index]
            src_value, dst_value = flat_pairs[2 * layer_index + 1]
            lane_index = layer_index % len(lanes)
            return executor.submit(
                self._submit_one_layerwise_h2d_pair,
                job_id=job_id,
                layer_index=layer_index,
                src_key_cache=src_key,
                src_value_cache=src_value,
                dst_key_cache=dst_key,
                dst_value_cache=dst_value,
                src_to_dst_tensor=src_to_dst_tensor,
                layer_event=layer_events[self._layer_names[layer_index]],
                stream=lanes[lane_index],
                stream_lock=self._layer_h2d_submit_stream_locks[lane_index],
            )

        futures = []
        with ThreadPoolExecutor(
            max_workers=len(lanes), thread_name_prefix="secure_infer-layer-h2d"
        ) as executor:
            for layer_index in range(layer_count):
                if layer_index == 1 and not _gcm_h2d_layerwise_eager_submission_enabled():
                    # Layer 0's worker has published its event before the
                    # compatibility handshake is checked.  No later layer is
                    # submitted until the real attention hook is observed.
                    futures[0].result()
                    self._wait_for_layerwise_hook_handshake(job_id)
                if layer_index >= horizon:
                    self._wait_for_layerwise_lookahead_slot(
                        job_id, layer_index - horizon
                    )
                futures.append(submit(executor, layer_index))
            for future in futures:
                future.result()

        # The framework still owns one request-completion event.  Join every
        # independently submitted layer event onto its original H2D stream;
        # this is asynchronous and retains request-atomic cache publication.
        with torch.npu.stream(parent_stream):
            for layer_name in self._layer_names:
                parent_stream.wait_event(layer_events[layer_name])

    def _submit_transfer(
        self,
        job_id: int,
        spec: TransferSpec,
        h2d_stream_override: torch.npu.Stream | None = None,
        completion_event: torch.npu.Event | None = None,
    ) -> bool:
        submit_started_ns = time.monotonic_ns()
        logger.info("start transfer submission: job=%d", job_id)
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = h2d_stream_override or self.h2d_stream
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
        job_layerwise = is_h2d and job_id in self._layerwise_h2d_job_ids

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor

        assert src_sub_block_count == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip

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
        # This list also retains non-default-stream join events below.  Ascend
        # may consume an event/mapping asynchronously after this Python
        # frame returns, so do not allow either object to be collected before
        # the framework-visible transfer event has completed.
        transfer_host_refs: list[object] = [src_to_dst_tensor]

        # All real H2D source addresses are observed at the intercepted
        # swap_blocks call.  Keep the source-id view only to select the H2D
        # submission branch; no Python code registers these IDs as future
        # crypto work.
        h2d_source_block_ids = (
            np.ascontiguousarray(src_to_dst[:, 0], dtype=np.int64)
            if is_h2d else None
        )
        if is_h2d:
            with self._transfer_lock:
                self._h2d_submission_count += 1
                h2d_submission_ordinal = self._h2d_submission_count
        else:
            h2d_submission_ordinal = 0

        with self._transfer_lock:
            event = self.events_pool.pop() if self.events_pool else torch.npu.Event()
            device_event = None
            if not is_h2d and self._d2h_two_phase:
                device_event = (
                    self.events_pool.pop() if self.events_pool else torch.npu.Event()
                )
        layer_events: dict[str, torch.npu.Event] = {}
        if job_layerwise:
            if len(src_tensors) != len(self._layer_names):
                with self._transfer_lock:
                    self.events_pool.append(event)
                raise RuntimeError(
                    "SecureInfer layer/tensor registration changed after handler "
                    "initialization; refusing an unsafe layerwise H2D load"
                )
            # Layer 0 has a private event.  Later adjacent layers may share a
            # conservative group-completion event, reducing per-layer stream
            # joins while preserving verify-before-use for every member.
            group_events: dict[int, torch.npu.Event] = {}
            layer_events = {}
            for layer_index, layer_name in enumerate(self._layer_names):
                _, group_end = _layer_event_group_bounds(
                    layer_index,
                    len(self._layer_names),
                    self._layer_h2d_event_group_size,
                )
                group_event = group_events.get(group_end)
                if group_event is None:
                    group_event = torch.npu.Event(
                        enable_timing=self._layer_device_timing
                    )
                    group_events[group_end] = group_event
                layer_events[layer_name] = group_event
            transfer_host_refs.extend(group_events.values())
            # The foreground waits until a layer event is recorded instead of
            # ever attaching a dependency to an unrecorded event.
            with self._layer_h2d_condition:
                self._layer_h2d_events[job_id] = layer_events
                self._layer_h2d_recorded_layers[job_id] = set()
                # The foreground hook can run after transfer_async() marks
                # the job active but before this background submitter obtains
                # the cross-job stream lock.  Preserve that early fail-closed
                # handshake instead of erasing it during late event-table
                # initialization.
                self._layer_h2d_waited_layers.setdefault(job_id, set())
                if self._layer_device_timing:
                    self._layer_device_windows[job_id] = {
                        layer_name: {"verify_ready": event}
                        for layer_name, event in layer_events.items()
                    }
                self._layer_h2d_condition.notify_all()
        # Intra-request striping is safe only on the regular per-handler H2D
        # stream.  An inter-request worker already owns a private stream and
        # has a separate disjointness protocol; mixing its stream with this
        # shared stripe pool would create an unnecessary cross-job dependency.
        parallel_layer_admission = (
            job_layerwise and self._layer_h2d_submit_worker_count > 1
        )
        h2d_slice_streams: list[torch.npu.Stream] = (
            [stream]
            if parallel_layer_admission
            else
            self._h2d_slice_streams
            if is_h2d and h2d_stream_override is None
            and self._h2d_slice_stream_count > 1
            else [stream]
        )
        layer_admission_held = False
        layer_admission_index = -1
        h2d_group_started = False
        try:
            with torch.npu.stream(stream):
                if h2d_source_block_ids is not None:
                    flat_pairs = []
                    for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
                        flat_pairs.append((src_tensor[0], dst_tensor[0]))
                        flat_pairs.append((src_tensor[1], dst_tensor[1]))
                    if job_layerwise and len(flat_pairs) != 2 * len(
                        self._layer_names
                    ):
                        raise RuntimeError(
                            "SecureInfer layerwise H2D expected exactly one K/V "
                            "pair per attention layer"
                        )

                    if parallel_layer_admission:
                        # Submit independently ordered layer pairs on their
                        # own streams.  The native interceptor observes each
                        # real copy and performs history-based preparation;
                        # there is no exact future-address admission here.
                        self._submit_layerwise_h2d_pairs_parallel(
                            job_id=job_id,
                            flat_pairs=flat_pairs,
                            src_to_dst_tensor=src_to_dst_tensor,
                            layer_events=layer_events,
                            parent_stream=stream,
                        )
                    else:
                        layer_lookahead = _gcm_h2d_layer_lookahead()
                        for op_idx, (src_cache, dst_cache) in enumerate(flat_pairs):
                            if (job_layerwise and layer_lookahead > 0
                                    and op_idx % 2 == 0):
                                layer_index = op_idx // 2
                                predecessor_index = layer_index - layer_lookahead
                                if predecessor_index >= 0:
                                    predecessor = self._layer_names[predecessor_index]
                                # Do not submit arbitrarily far-ahead GCM
                                # work.  First wait until the foreground has
                                # installed the predecessor dependency, then
                                # wait for that dependency to complete on the
                                # device.  The latter is essential: a host
                                # ``wait_event`` call only enqueues a command
                                # and does not mean that predecessor execution
                                # has started yet.
                                    self._wait_for_layerwise_lookahead_slot(
                                        job_id, predecessor_index
                                    )
                            if job_layerwise and op_idx % 2 == 0:
                                layer_admission_index = op_idx // 2
                                self._begin_layerwise_h2d_layer(
                                    job_id, layer_admission_index
                                )
                                layer_admission_held = True
                                if self._layer_device_timing:
                                    layer_name = self._layer_names[op_idx // 2]
                                    verify_start = torch.npu.Event(enable_timing=True)
                                    verify_start.record(stream)
                                    with self._layer_h2d_condition:
                                        self._layer_device_windows[job_id][layer_name][
                                            "verify_start"
                                        ] = verify_start
                                _layer_trace(
                                    "verification_begin",
                                    job_id=job_id,
                                    layer_name=self._layer_names[op_idx // 2],
                                    layer_index=op_idx // 2,
                                )
                                h2d_group_started = _begin_h2d_layer_group(
                                    2 * int(src_to_dst_tensor.shape[0])
                                )
                            if len(h2d_slice_streams) == 1:
                                torch.ops._C_ascend.swap_blocks(
                                    src_cache,
                                    dst_cache,
                                    src_to_dst_tensor,
                                )
                            else:
                                # Every mapping row names one physical CPU source
                            # and one NPU destination.  Stripes are contiguous,
                            # disjoint row ranges, so each authenticated GCM
                            # task retains its source/ticket/IV mapping.
                            # The only semantic change is that independent
                            # verified releases can run on separate AICPU
                            # streams; the join below still makes the logical
                            # request visible atomically.
                                total_rows = int(src_to_dst_tensor.shape[0])
                                stripe_count = min(len(h2d_slice_streams), total_rows)
                                for stripe_index in range(stripe_count):
                                    begin = total_rows * stripe_index // stripe_count
                                    end = total_rows * (stripe_index + 1) // stripe_count
                                    mapping = src_to_dst_tensor[begin:end]
                                # Keep the view object and its CPU mapping
                                # storage alive through the final load event.
                                    transfer_host_refs.append(mapping)
                                    with torch.npu.stream(h2d_slice_streams[stripe_index]):
                                        torch.ops._C_ascend.swap_blocks(
                                            src_cache, dst_cache, mapping
                                        )
                            if job_layerwise and op_idx % 2 == 1:
                                _end_h2d_layer_group(h2d_group_started)
                                h2d_group_started = False
                                layer_index = op_idx // 2
                                group_begin, group_end = _layer_event_group_bounds(
                                    layer_index,
                                    len(self._layer_names),
                                    self._layer_h2d_event_group_size,
                                )
                                # Non-final members are authenticated now but
                                # conservatively released with the group's
                                # final member.  This avoids an expensive
                                # three-stream join at every Transformer layer.
                                if layer_index != group_end:
                                    self._end_layerwise_h2d_layer(
                                        job_id,
                                        layer_index,
                                        layer_events[
                                            self._layer_names[layer_index]
                                        ],
                                    )
                                    layer_admission_held = False
                                    layer_admission_index = -1
                                    continue
                            # The preceding two operations are this layer's K
                            # and V tensors.  With striping, join every private
                            # stream at this exact layer boundary before
                            # recording the public layer event.  Each stripe
                            # owns disjoint mapping rows, and swap_blocks only
                            # publishes its destination after a successful GCM
                            # gate, so attention cannot observe a partially
                            # authenticated layer.
                                if len(h2d_slice_streams) > 1:
                                    for slice_stream in h2d_slice_streams:
                                        if slice_stream == stream:
                                            continue
                                        stripe_done = torch.npu.Event()
                                        stripe_done.record(slice_stream)
                                        transfer_host_refs.append(stripe_done)
                                        stream.wait_event(stripe_done)
                                layer_name = self._layer_names[layer_index]
                                layer_events[layer_name].record(stream)
                                with self._layer_h2d_condition:
                                    for released_index in range(
                                        group_begin, group_end + 1
                                    ):
                                        released_name = self._layer_names[
                                            released_index
                                        ]
                                        self._layer_h2d_recorded_layers[job_id].add(
                                            released_name
                                        )
                                        _layer_trace(
                                            "verification_ready",
                                            job_id=job_id,
                                            layer_name=released_name,
                                            layer_index=released_index,
                                        )
                                    self._layer_h2d_condition.notify_all()
                                    # Rotate to the next active request as soon
                                    # as this layer's K/V event is visible.  No
                                    # device synchronization is introduced.
                                    self._end_layerwise_h2d_layer(
                                        job_id,
                                        layer_index,
                                        layer_events[layer_name],
                                    )
                                    layer_admission_held = False
                                    layer_admission_index = -1
                                    if (op_idx == 1 and not
                                            _gcm_h2d_layerwise_eager_submission_enabled()):
                                        # Prove that the active attention backend
                                    # actually installed a dependency before
                                    # allowing later layers to be submitted.
                                    # This turns an unsupported backend from a
                                    # silent verify-before-use violation into
                                    # a fail-closed experiment.
                                        deadline = time.monotonic() + 10.0
                                        while not self._layer_h2d_waited_layers[
                                            job_id
                                        ]:
                                            remaining = deadline - time.monotonic()
                                            if remaining <= 0:
                                                raise RuntimeError(
                                                    "SecureInfer verified-layer H2D did "
                                                    "not observe an attention-layer "
                                                    "wait hook within 10 seconds"
                                                )
                                            self._layer_h2d_condition.wait(remaining)
                                    elif op_idx == 1:
                                        # The connector deployment has already
                                    # selected the checked verified-layer
                                    # execution path.  Fill the bounded
                                    # horizon while layer 0 waits on its own
                                    # device event, rather than letting host
                                    # event admission lag behind model launch.
                                        logger.info(
                                            "SecureInfer eager layerwise admission: "
                                            "job=%d horizon=%d",
                                            job_id,
                                            _gcm_h2d_layer_lookahead(),
                                        )
                else:
                    # Bracket the complete logical store job.  event.record()
                    # stays after the final CPU decrypt HostFunc, so publishing
                    # the event never makes ciphertext visible as a cache hit.
                    expected_tasks = int(src_to_dst.shape[0]) * len(src_tensors) * 2
                    d2h_batch_started = _begin_d2h_job_batch(expected_tasks)

                    def store_tensor_in_bounded_groups(
                        src_cache: torch.Tensor, dst_cache: torch.Tensor
                    ) -> None:
                        if not d2h_batch_started:
                            torch.ops._C_ascend.swap_blocks(
                                src_cache, dst_cache, src_to_dst_tensor
                            )
                            return

                        # Keep each device staging slot bounded (default:
                        # 16 x 256 KiB ~= 4 MiB) while preserving the exact
                        # descriptor order.  Multiple groups can occupy the
                        # NPU, PCIe, and CPU stages concurrently.
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
                            src_key_cache, src_value_cache = src_tensor[0], src_tensor[1]
                            dst_key_cache, dst_value_cache = dst_tensor[0], dst_tensor[1]
                            store_tensor_in_bounded_groups(
                                src_key_cache, dst_key_cache
                            )
                            store_tensor_in_bounded_groups(
                                src_value_cache, dst_value_cache
                            )
                    finally:
                        if d2h_batch_started:
                            # Join the independent PCIe stream first.  The
                            # device event is source-safe but CPU decrypt may
                            # still be running; batch_end adds the stronger
                            # plaintext-ready completion point.
                            _join_d2h_device_pipeline()
                            if device_event is not None:
                                device_event.record(stream)
                            _end_d2h_job_batch()
                        elif device_event is not None:
                            # Compatibility path: per-slice HostFuncs were
                            # inserted inside swap_blocks, so this event is
                            # conservative (device release == cache ready).
                            device_event.record(stream)

                if (is_h2d and len(h2d_slice_streams) > 1
                        and not job_layerwise):
                    # `event` is recorded on the framework's original H2D
                    # stream.  Make it wait for each private stripe before
                    # publishing completion, so callers never observe a
                    # partial KV load and a failed tag stays fail-closed.
                    for slice_stream in h2d_slice_streams:
                        if slice_stream == stream:
                            continue
                        slice_done = torch.npu.Event()
                        slice_done.record(slice_stream)
                        transfer_host_refs.append(slice_done)
                        stream.wait_event(slice_done)
                event.record(stream)
                # A parallel submit worker must retain an independent
                # completion boundary.  The public event can be consumed and
                # returned to events_pool by get_finished() immediately after
                # it fires, whereas this private event is never pooled and is
                # used solely to release the source/destination safety guard.
                if completion_event is not None:
                    completion_event.record(stream)
        except BaseException:
            if h2d_group_started:
                _end_h2d_layer_group(True)
                h2d_group_started = False
            if layer_admission_held:
                self._end_layerwise_h2d_layer(
                    job_id, layer_admission_index
                )
                layer_admission_held = False
            with self._transfer_lock:
                self.events_pool.append(event)
                if device_event is not None:
                    self.events_pool.append(device_event)
            raise

        # swap_blocks has already passed every real H2D slice through the
        # native hook before returning. Retire predictions left unused by this
        # completed logical transfer scope; their ticket slabs are reclaimed
        # only after the NPU ready event reports completion. Otherwise one
        # unused slice can pin an entire tensor slab indefinitely.
        if is_h2d:
            _retire_h2d_prediction_scope()

        with self._transfer_lock:
            self.transfer_events[job_id] = event
            self._transfer_host_refs[job_id] = transfer_host_refs
            if layer_events:
                self._layer_h2d_events[job_id] = layer_events
            if device_event is not None:
                self.device_transfer_events[job_id] = device_event
            if is_h2d:
                started_ns, _ = self._h2d_job_timing_ns.get(
                    job_id, (submit_started_ns, 0)
                )
                submitted_ns = time.monotonic_ns()
                self._h2d_job_timing_ns[job_id] = (
                    started_ns, submitted_ns
                )
                logger.info(
                    "SecureInfer H2D host submission: job=%d layerwise=%s "
                    "streams=%d submit_ms=%.3f",
                    job_id,
                    job_layerwise,
                    len(h2d_slice_streams),
                    (submitted_ns - started_ns) / 1e6,
                )
        return True

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Insert authenticated-H2D dependencies for one attention layer.

        This method never synchronizes the host.  It adds event waits to the
        foreground stream, allowing H2D authentication for later layers to
        run concurrently with earlier-layer attention.  In a batch with
        several external reloads, the layer must wait for every matching KV
        tensor before use.
        """
        if not self._h2d_layerwise:
            return
        if not self._layer_h2d_active_fast:
            return

        events: list[tuple[int, torch.npu.Event]] = []
        try:
            current_layer_index = self._layer_names.index(layer_name)
        except ValueError:
            current_layer_index = -1
        with self._layer_h2d_condition:
            active_jobs = sorted(
                self._layer_h2d_active_jobs.intersection(
                    self._layer_h2d_wait_jobs
                )
            )
            # Once the reload has completed, later decode iterations must not
            # pay per-layer tracing, event lookup, or admission bookkeeping.
            # The check stays under the same lock used to publish/remove jobs,
            # so it cannot race a newly activated secure load.
            if not active_jobs:
                return
            if current_layer_index >= 0:
                for job_id in active_jobs:
                    self._layer_h2d_current_layer[job_id] = max(
                        current_layer_index,
                        self._layer_h2d_current_layer.get(job_id, -1),
                    )
            _layer_trace("layer_wait_begin", layer_name=layer_name)
            # Register this real attention hook for the whole serving batch
            # before waiting for any one job's event.  H2D command admission
            # is serialized across requests because the jobs share a small
            # stream pool.  Waiting for an arbitrary job here before marking
            # the jobs ahead of it can otherwise form a host-side cycle:
            # the foreground waits for job B, while job A holds the submit
            # lock waiting for its hook handshake.  This first pass only
            # releases the fail-closed handshake; no KV can be consumed until
            # the second pass finds and enqueues every authenticated event.
            first_wait_jobs: list[int] = []
            for job_id in active_jobs:
                layer_events = self._layer_h2d_events.get(job_id)
                if layer_events is not None and layer_name not in layer_events:
                    raise RuntimeError(
                        "SecureInfer attention/cache layer-name mismatch: "
                        f"attention={layer_name!r} registered="
                        f"{tuple(layer_events)!r}"
                    )
                waited_layers = self._layer_h2d_waited_layers.setdefault(
                    job_id, set()
                )
                first_wait = not waited_layers
                waited_layers.add(layer_name)
                if first_wait:
                    first_wait_jobs.append(job_id)
                    try:
                        execution_index = self._layer_names.index(layer_name)
                    except ValueError:
                        execution_index = -1
                    logger.info(
                        "SecureInfer first verified-layer wait: job=%d layer=%s "
                        "registered_index=%d",
                        job_id,
                        layer_name,
                        execution_index,
                    )
            self._layer_h2d_condition.notify_all()

            initial_admission = _gcm_h2d_layerwise_initial_admission_layers()
            if initial_admission > 1:
                for job_id in first_wait_jobs:
                    # The first attention hook is the only safe place to wait
                    # for a bounded submission lead.  All jobs in this batch
                    # have already released their backend handshakes above,
                    # so this wait cannot block an earlier submitter.
                    # Authentication completion is still enforced by the
                    # exact per-job event wait below.
                    deadline = time.monotonic() + 10.0
                    while (
                        len(self._layer_h2d_recorded_layers.get(job_id, ()))
                        < initial_admission
                    ):
                        pending = self._h2d_submissions.get(job_id)
                        if pending is not None and pending.error is not None:
                            raise RuntimeError(
                                "SecureInfer layerwise initial admission failed "
                                f"for job={job_id}"
                            ) from pending.error
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError(
                                "SecureInfer timed out waiting for initial "
                                f"layer-event admission (job={job_id}, "
                                f"layers={initial_admission})"
                            )
                        self._layer_h2d_condition.wait(remaining)
                    logger.info(
                        "SecureInfer layerwise initial event admission: "
                        "job=%d layers=%d",
                        job_id,
                        initial_admission,
                    )

            for job_id in active_jobs:
                while True:
                    pending = self._h2d_submissions.get(job_id)
                    if pending is not None and pending.error is not None:
                        raise RuntimeError(
                            "SecureInfer layerwise H2D submission failed before "
                            f"layer {layer_name!r} could be released"
                        ) from pending.error
                    layer_events = self._layer_h2d_events.get(job_id)
                    recorded_layers = self._layer_h2d_recorded_layers.get(job_id)
                    if (
                        layer_events is not None
                        and recorded_layers is not None
                        and layer_name in recorded_layers
                    ):
                        events.append((job_id, layer_events[layer_name]))
                        _layer_trace(
                            "layer_event_admitted",
                            job_id=job_id,
                            layer_name=layer_name,
                        )
                        break
                    if pending is None:
                        raise RuntimeError(
                            "SecureInfer lost an active layerwise H2D job before "
                            f"layer {layer_name!r} was verified"
                        )
                    # This releases the GIL for the submitter.  It waits only
                    # for event recording, never for GCM completion itself.
                    self._layer_h2d_condition.wait()

        # Keep a bounded command lead for future layer events.  Crucially this
        # is after the
        # current layer event has been found, and it waits only for future
        # event *recording*.  While the host is paused, the prior foreground
        # device work and the current layer's authenticated H2D can progress
        # on their own streams.  It therefore removes host admission bubbles
        # without turning the horizon into a global authentication barrier.
        admission_horizon = _gcm_h2d_layerwise_host_admission_horizon()
        submit_lookahead = _gcm_h2d_layer_lookahead()
        if submit_lookahead > 0:
            # The submitter intentionally waits for layer i's device gate
            # before admitting layer i+lookahead.  At layer i itself that
            # gate has not yet been enqueued, so admitting that exact boundary
            # here would form a host-side cycle.  Keep one slot below it.
            admission_horizon = min(
                admission_horizon, max(0, submit_lookahead - 1)
            )
        if admission_horizon > 0 and current_layer_index >= 0:
            target_index = min(
                len(self._layer_names) - 1,
                current_layer_index + admission_horizon,
            )
            target_layer = self._layer_names[target_index]
            with self._layer_h2d_condition:
                for job_id, _ in events:
                    deadline = time.monotonic() + 10.0
                    while target_layer not in self._layer_h2d_recorded_layers.get(
                        job_id, set()
                    ):
                        pending = self._h2d_submissions.get(job_id)
                        if pending is not None and pending.error is not None:
                            raise RuntimeError(
                                "SecureInfer layerwise future-event admission "
                                f"failed before layer {target_layer!r}"
                            ) from pending.error
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError(
                                "SecureInfer timed out waiting for layerwise "
                                f"admission horizon={admission_horizon}"
                            )
                        self._layer_h2d_condition.wait(remaining)
            _layer_trace(
                "layer_admission_horizon_ready",
                layer_name=layer_name,
                layer_index=current_layer_index,
                horizon=admission_horizon,
            )

        if not events:
            return

        foreground_stream = torch.npu.current_stream()
        # Record the deadline *before* inserting the authenticated-H2D wait.
        # On the foreground stream this event is ordered after all work from
        # the previous decoder layer, so it is the first-use deadline for this
        # layer's KV.  The post-wait event is the actual safe consumption
        # point.  Their difference is exactly the exposed verification stall.
        deadline_events: dict[int, torch.npu.Event] = {}
        if self._layer_device_timing:
            for job_id, _ in events:
                deadline = torch.npu.Event(enable_timing=True)
                deadline.record(foreground_stream)
                deadline_events[job_id] = deadline
        # A batch-layer wave deliberately shares one authenticated completion
        # event across all requests in the same serving batch.  Preserve every
        # request's bookkeeping below, but enqueue the identical device
        # dependency only once on the foreground stream.
        unique_events: list[torch.npu.Event] = []
        seen_event_ids: set[int] = set()
        for _, event in events:
            event_id = id(event)
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            unique_events.append(event)
        for event in unique_events:
            foreground_stream.wait_event(event)
        if os.getenv("PIPE_GCM_FAULT_EVENT_PATH") and not (
            os.getenv("PIPE_GCM_FASTTRACK_EAGER", "0") == "1"
        ):
            # Only the recovery experiment observes host-side verifier status.
            # The failed record/layer comes from that status, never the injector.
            library = ctypes.CDLL(None)
            wait_auth = library.secure_infer_recovery_wait_layer_authentication
            wait_auth.argtypes = [ctypes.c_uint32]
            wait_auth.restype = ctypes.c_int
            if current_layer_index < 0 or wait_auth(current_layer_index) != 0:
                raise RuntimeError("authenticated KV recovery did not complete")
        # The event wait is now enqueued on the real foreground stream.  This
        # is the producer's bounded-lookahead release point even when device
        # timing is disabled.
        with self._layer_h2d_condition:
            needs_consume_event = self._layer_device_timing or submit_lookahead > 0
            shared_consume: torch.npu.Event | None = None
            if needs_consume_event:
                shared_consume = torch.npu.Event(
                    enable_timing=self._layer_device_timing
                )
                shared_consume.record(foreground_stream)
            for job_id, _ in events:
                if shared_consume is not None:
                    self._layer_h2d_consumed_events.setdefault(
                        job_id, {})[layer_name] = shared_consume
                if self._layer_device_timing:
                    timing = self._layer_device_windows.get(job_id, {}).get(layer_name)
                    if timing is not None:
                        timing["deadline"] = deadline_events[job_id]
                        assert shared_consume is not None
                        timing["consume"] = shared_consume
            self._layer_h2d_condition.notify_all()
        _layer_trace("layer_wait_enqueued", layer_name=layer_name)
        logger.debug(
            "SecureInfer layerwise H2D dependency: layer=%s active_loads=%d",
            layer_name,
            len(events),
        )

    def _emit_completed_layer_device_timings(self) -> None:
        """Emit device-time security windows without synchronizing serving."""
        if not self._layer_device_timing:
            return
        completed_jobs: list[int] = []
        with self._layer_h2d_condition:
            jobs = list(self._layer_device_windows.items())
        for job_id, layers in jobs:
            # A job can finish its H2D stream before the foreground has
            # consumed every layer.  Query rather than synchronize, and defer
            # emission to a later scheduler poll if any layer is still live.
            required = ("verify_start", "verify_ready", "deadline", "consume")
            if not layers or any(any(key not in item for key in required)
                                 for item in layers.values()):
                continue
            if not all(event.query() for item in layers.values()
                       for event in item.values()):
                continue
            previous_consume: torch.npu.Event | None = None
            first_deadline = layers.get(self._layer_names[0], {}).get("deadline")
            first_deadline_time = (
                first_deadline.recorded_time()
                if first_deadline is not None else None
            )
            for layer_index, layer_name in enumerate(self._layer_names):
                item = layers.get(layer_name)
                if item is None:
                    continue
                verify_ms = item["verify_start"].elapsed_time(item["verify_ready"])
                lead_ms = item["verify_start"].elapsed_time(item["deadline"])
                gate_stall_ms = item["deadline"].elapsed_time(item["consume"])
                prior_compute_ms = (
                    previous_consume.elapsed_time(item["deadline"])
                    if previous_consume is not None else None
                )
                _layer_trace(
                    "device_layer_window",
                    job_id=job_id,
                    layer_name=layer_name,
                    layer_index=layer_index,
                    verification_ms=verify_ms,
                    verification_lead_to_deadline_ms=lead_ms,
                    prior_layer_compute_window_ms=prior_compute_ms,
                    gate_stall_ms=gate_stall_ms,
                    # Absolute device timestamps put authentication and
                    # foreground consumption on one timeline.  The analysis
                    # normalizes these values to layer 0's first-consumption
                    # point; it does not compare device time with host time.
                    device_time_origin=first_deadline_time,
                    verification_start_device_time=(
                        item["verify_start"].recorded_time()
                    ),
                    verification_ready_device_time=(
                        item["verify_ready"].recorded_time()
                    ),
                    deadline_device_time=item["deadline"].recorded_time(),
                    consume_device_time=item["consume"].recorded_time(),
                )
                previous_consume = item["consume"]
            completed_jobs.append(job_id)
        if completed_jobs:
            with self._layer_h2d_condition:
                for job_id in completed_jobs:
                    self._layer_device_windows.pop(job_id, None)

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
                for pending in (
                    list(self._d2h_submissions.values())
                    + list(self._h2d_submissions.values())
                )
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
                    self._layer_h2d_events.pop(job_id, None)
                    self._layer_h2d_recorded_layers.pop(job_id, None)
                    waited_layers = self._layer_h2d_waited_layers.pop(job_id, set())
                    self._layer_h2d_consumed_events.pop(job_id, None)
                    self._layer_h2d_current_layer.pop(job_id, None)
                    self._layer_h2d_urgency_waiters = {
                        key: value
                        for key, value in self._layer_h2d_urgency_waiters.items()
                        if key[0] != job_id
                    }
                    self._layer_h2d_inflight = [
                        item
                        for item in self._layer_h2d_inflight
                        if item[0] != job_id
                    ]
                    self._layer_h2d_active_jobs.discard(job_id)
                    self._layer_h2d_active_fast = bool(
                        self._layer_h2d_active_jobs
                    )
                    from .attention_plugin import mark_layerwise_wait_active
                    mark_layerwise_wait_active(
                        self, self._layer_h2d_active_fast
                    )
                    self._layerwise_h2d_job_ids.discard(job_id)
                    self._layer_h2d_threads.pop(job_id, None)
                    self._layer_h2d_condition.notify_all()
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
                    h2d_pending = self._h2d_submissions.pop(job_id, None)
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
                    if h2d_pending is not None:
                        self._h2d_completed_jobs += 1
                        logger.info(
                            "SecureInfer SECURE_LOAD_READY: job=%d "
                            "observed_ms=%.3f layer_wait_coverage=%d/%d",
                            job_id,
                            (time.monotonic_ns() - h2d_pending.queued_ns) / 1e6,
                            len(waited_layers),
                            len(self._layer_names),
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
        # This is a nonblocking query-only collector.  It keeps measurement
        # out of the serving critical path and may emit on a later poll once
        # the final foreground layer has completed.
        self._emit_completed_layer_device_timings()
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
                if pending is None:
                    pending = self._h2d_submissions.get(job_id)
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
