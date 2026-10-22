#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ -z "${ASCEND_ROOT:-}" ]]; then
  for candidate in /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do
    if [[ -d "$candidate/aarch64-linux/include" ]]; then
      export ASCEND_ROOT="$candidate"
      break
    fi
  done
fi
: "${ASCEND_ROOT:?请在 .env 中设置 ASCEND_ROOT}"
"${PYTHON:-python3}" -c 'import torch, torch_npu, vllm, vllm_ascend; print("推理依赖检查通过")'
make lib
