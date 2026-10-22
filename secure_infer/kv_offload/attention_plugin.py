"""Runtime installation of the SealFlow attention dependency.

The vLLM connector interface already defines ``wait_for_layer_load``.  The
Ascend MLA and SFA implementations call it, while the generic MHA backends in
the evaluated version do not.  The dynamically loaded SealFlow connector
inserts the same call at their pre-compute dependency point without editing
the vLLM-Ascend package.  Exact source checks make an unsupported backend
version fail closed instead of silently weakening authentication-before-use.
"""

from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
import os
import textwrap
import threading
import time


_INSTALL_LOCK = threading.Lock()
_MARKER = "__sealflow_layer_wait_installed__"
_TRACE_ENABLED = bool(os.environ.get("PIPE_LAYERWISE_TRACE_PATH"))
_STATE_TRACE_PATH = os.environ.get("PIPE_ATTENTION_STATE_TRACE_PATH", "")
_STATE_TRACE_LOCK = threading.Lock()
_ACTIVE_WAIT_OWNERS: set[int] = set()
_ACTIVE_WAIT_LOCK = threading.Lock()
_FAULT_ABORT_REPORTED = False


def _raise_on_authenticated_transfer_failure(layer_name: str | None) -> None:
    """Report an opt-in speculative execution failure.

    The fault flag is used only by the recovery experiment.  In scheduler mode
    the already submitted forward pass is allowed to drain and the scheduler
    invalidates the request.  In fatal mode the exception stops execution.
    The experiment keeps response streaming disabled and prevents this
    request's KV from entering host cache, so no tentative output or KV is
    committed.
    """
    global _FAULT_ABORT_REPORTED
    flag_path = os.environ.get("PIPE_GCM_FAULT_FLAG_PATH", "")
    if not flag_path or not os.path.exists(flag_path):
        # A serving process can run more than one fault experiment.  Reset
        # this process-local guard after the previous arm is removed so each
        # injected failure gets one observation event.
        _FAULT_ABORT_REPORTED = False
        return
    if not _FAULT_ABORT_REPORTED:
        _FAULT_ABORT_REPORTED = True
        event_path = os.environ.get("PIPE_GCM_FAULT_EVENT_PATH", "")
        if event_path:
            record = {
                "phase": "authentication_failure_observed",
                "realtime_ns": time.time_ns(),
                "layer_name": layer_name,
                "abort_mode": os.environ.get("PIPE_GCM_FAULT_ABORT_MODE", "fatal"),
            }
            with open(event_path, "a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(record, sort_keys=True) + "\n")
    if os.environ.get("PIPE_GCM_FAULT_ABORT_MODE", "fatal") == "scheduler":
        # Raising inside a model kernel makes vLLM treat the failure as an
        # EngineCore failure.  The recovery experiment instead lets the
        # already-submitted forward pass drain, then the scheduler rejects its
        # output and frees only this request's private KV state.
        return
    raise RuntimeError(
        "authentication failed for speculative H2D KV; tentative request "
        "state must be discarded"
    )


def mark_layerwise_wait_active(owner: object, active: bool) -> None:
    """Publish whether an H2D job currently needs per-layer dependencies."""
    owner_id = id(owner)
    with _ACTIVE_WAIT_LOCK:
        if active:
            _ACTIVE_WAIT_OWNERS.add(owner_id)
        else:
            _ACTIVE_WAIT_OWNERS.discard(owner_id)


def layerwise_wait_active() -> bool:
    # Reading a set's truth value is atomic under the CPython GIL.  The write
    # side remains locked because multiple H2D completion threads may update
    # different owners concurrently.
    return bool(_ACTIVE_WAIT_OWNERS)


def _attention_trace(kind: str, layer_name: str | None) -> None:
    if not _TRACE_ENABLED or layer_name is None:
        return
    # Reuse the connector's append-only trace writer so all timestamps share
    # one lock and one format.  Import lazily to avoid a connector import cycle.
    from .npu_predict import _layer_trace

    _layer_trace(kind, layer_name=layer_name)


def _attention_state_trace(layer_name: str | None, output) -> None:
    """Record the final token's attention output for fault experiments."""
    if not _STATE_TRACE_PATH or layer_name is None or output is None:
        return
    try:
        # The attention output may be [tokens, heads, head_dim].  Select the
        # final token first and then retain every head; reshaping first would
        # inspect only the final head and can miss a localized perturbation.
        tail = output.detach()[-1].reshape(-1).float().cpu()
        raw = tail.numpy().tobytes()
        record = {
            "layer_name": layer_name,
            "elements": int(tail.numel()),
            "output_shape": list(output.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sum": float(tail.sum().item()),
            "l2": float(tail.square().sum().sqrt().item()),
            "max_abs": float(tail.abs().max().item()),
            "finite": bool(tail.isfinite().all().item()),
        }
        with _STATE_TRACE_LOCK:
            with open(_STATE_TRACE_PATH, "a", encoding="utf-8") as trace:
                trace.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as error:
        with _STATE_TRACE_LOCK:
            with open(_STATE_TRACE_PATH, "a", encoding="utf-8") as trace:
                trace.write(json.dumps({
                    "layer_name": layer_name,
                    "error": repr(error),
                }, sort_keys=True) + "\n")


def _kv_propagation_snapshot(layer_name, kind, tensor) -> None:
    """Save sampled new K/V and attention outputs for a diagnostic experiment.

    This copies the last 32 current-token rows, not reloaded prefix KV.
    CPU copies synchronize execution, so these runs are not latency samples.
    """
    root = os.environ.get("PIPE_KV_PROPAGATION_DIR", "")
    if not root or tensor is None or layer_name is None:
        return
    from pathlib import Path
    import numpy as np
    phase_file = Path(root) / "phase"
    if not phase_file.exists():
        return
    phase = phase_file.read_text().strip()
    if not phase:
        return
    destination = Path(root) / phase
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{layer_name}.{kind}.npy"
    if path.exists():
        raise RuntimeError(f"diagnostic expects one prefill per phase: {path}")
    sampled = tensor.detach()[-32:].float().cpu().numpy()
    np.save(path, sampled, allow_pickle=False)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SealFlow attention plugin expected one {label} site, found "
            f"{count}; the installed vLLM-Ascend backend is not supported"
        )
    return source.replace(old, new, 1)


def _patch_forward(cls, module_globals, marker, replacement, label) -> None:
    original = cls.forward
    if getattr(original, _MARKER, False):
        return

    source = textwrap.dedent(inspect.getsource(original))
    source = _replace_once(source, marker, replacement, label)
    namespace = dict(module_globals)
    exec(compile(source, f"<sealflow:{cls.__name__}.forward>", "exec"), namespace)
    patched = namespace["forward"]

    @wraps(patched)
    def traced(self, layer, *args, **kwargs):
        try:
            layer_name = getattr(layer, "layer_name", None)
            if os.environ.get("PIPE_KV_PROPAGATION_DIR"):
                key = kwargs.get("key", args[1] if len(args) > 1 else None)
                value = kwargs.get("value", args[2] if len(args) > 2 else None)
                _kv_propagation_snapshot(layer_name, "new_key", key)
                _kv_propagation_snapshot(layer_name, "new_value", value)
            result = patched(self, layer, *args, **kwargs)
            _kv_propagation_snapshot(layer_name, "attention_output", result)
            _attention_state_trace(getattr(layer, "layer_name", None), result)
            return result
        finally:
            _attention_trace("compute_end", getattr(layer, "layer_name", None))

    setattr(traced, _MARKER, True)
    cls.forward = traced


def install_attention_plugin() -> None:
    """Install the generic Ascend attention wait exactly once per process."""

    with _INSTALL_LOCK:
        from vllm_ascend.attention import attention_v1 as backend
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        try:
            from vllm_ascend.attention.attention_v1 import (
                AscendC8AttentionBackendImpl,
            )
        except ImportError:
            AscendC8AttentionBackendImpl = None
        from vllm_ascend.attention.utils import (
            wait_for_kv_layer_from_connector,
        )

        patch_globals = dict(vars(backend))
        patch_globals["wait_for_kv_layer_from_connector"] = (
            wait_for_kv_layer_from_connector
        )
        patch_globals["_sealflow_attention_trace"] = _attention_trace
        patch_globals["_sealflow_layerwise_wait_active"] = (
            layerwise_wait_active
        )
        patch_globals["_sealflow_raise_on_fault"] = (
            _raise_on_authenticated_transfer_failure
        )

        common_marker = """    if output_padded is not None:
"""
        common_replacement = """    layer_name = getattr(layer, "layer_name", None)
    _sealflow_attention_trace("compute_wait_begin", layer_name)
    _sealflow_raise_on_fault(layer_name)
    if (attn_metadata.attn_state != AscendAttentionState.DecodeOnly
            and _sealflow_layerwise_wait_active()):
        if layer_name is None:
            raise RuntimeError(
                "Ascend attention layer has no layer_name; SealFlow cannot "
                "enforce authentication before use"
            )
        wait_for_kv_layer_from_connector(layer_name)
    _sealflow_attention_trace("compute_start", layer_name)

    if output_padded is not None:
"""
        if common_marker in inspect.getsource(AscendAttentionBackendImpl.forward):
            _patch_forward(
                AscendAttentionBackendImpl,
                patch_globals,
                common_marker,
                common_replacement,
                "generic MHA dependency",
            )
        else:
            current_marker = (
                "    output = self.forward_impl(query, key, value, "
                "kv_cache, attn_metadata, output)\n"
            )
            current_replacement = (
                "    layer_name = getattr(layer, \"layer_name\", None)\n"
                "    _sealflow_attention_trace(\"compute_wait_begin\", layer_name)\n"
                "    _sealflow_raise_on_fault(layer_name)\n"
                "    if (attn_metadata.attn_state != AscendAttentionState.DecodeOnly\n"
                "            and _sealflow_layerwise_wait_active()):\n"
                "        if layer_name is None:\n"
                "            raise RuntimeError(\n"
                "                \"Ascend attention layer has no layer_name; SealFlow cannot \"\n"
                "                \"enforce authentication before use\"\n"
                "            )\n"
                "        wait_for_kv_layer_from_connector(layer_name)\n"
                "    _sealflow_attention_trace(\"compute_start\", layer_name)\n\n"
                "    output = self.forward_impl(query, key, value, kv_cache, "
                "attn_metadata, output)\n"
            )
            _patch_forward(
                AscendAttentionBackendImpl,
                patch_globals,
                current_marker,
                current_replacement,
                "generic MHA dependency",
            )

        c8_marker = """    self._prepare_c8_scales(layer, query.device)
    if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
"""
        c8_replacement = """    self._prepare_c8_scales(layer, query.device)
    layer_name = getattr(layer, "layer_name", None)
    _sealflow_attention_trace("compute_wait_begin", layer_name)
    _sealflow_raise_on_fault(layer_name)
    if (attn_metadata.attn_state != AscendAttentionState.DecodeOnly
            and _sealflow_layerwise_wait_active()):
        if layer_name is None:
            raise RuntimeError(
                "Ascend attention layer has no layer_name; SealFlow cannot "
                "enforce authentication before use"
            )
        wait_for_kv_layer_from_connector(layer_name)
    _sealflow_attention_trace("compute_start", layer_name)

    if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
"""
        if AscendC8AttentionBackendImpl is not None:
            _patch_forward(
                AscendC8AttentionBackendImpl,
                patch_globals,
                c8_marker,
                c8_replacement,
                "C8 attention dependency",
            )
