#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export MODEL_PATH="${MODEL_OVERRIDE:-${MODEL_PATH:-}}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_IDS:-0}"
export RUNTIME_RESULT_DIR="${RESULT_DIR:-runs/current}"
mkdir -p "$RUNTIME_RESULT_DIR"
export RUNTIME_COMPUTE_CHUNK="${COMPUTE_CHUNK:-1024}"
export RUNTIME_LOAD_CHUNK="${LOAD_CHUNK:-2048}"
export RUNTIME_NATIVE_MODE=native_secure
export PIPE_ENABLE_H2D="${ENABLE_H2D:-1}"
export PIPE_ENABLE_D2H="${ENABLE_D2H:-1}"
export PIPE_GHASH_AICPU_LIB="${GHASH_LIBRARY:-}"
export PIPE_GCM_AIV_KEYSTREAM_LIB="${AIV_KEYSTREAM_LIBRARY:-}"
if [[ "${SERVE_MODE:-runtime}" == runtime ]]; then
  export LD_PRELOAD="${ROOT_DIR}/libsecure_infer.so${PIPE_EXTRA_PRELOAD:+:${PIPE_EXTRA_PRELOAD}}"
elif [[ "${SERVE_MODE}" != plain ]]; then
  printf 'SERVE_MODE 必须是 runtime 或 plain\n' >&2
  exit 1
fi
exec "${PYTHON:-python3}" -m secure_infer.serving.entrypoint "$@"
