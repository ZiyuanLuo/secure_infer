#!/usr/bin/env bash
set -euo pipefail
# Both nodes expose the real DeepSeek-70B checkpoint at /checkpoint.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
pp_size=${PP_SIZE:-2}
container=paci-pp${pp_size}-tp8
docker exec "$container" mkdir -p /tmp/paci-runtime/secure_infer/serving
docker cp "$root/secure_infer/serving/entrypoint.py" \
    "$container":/tmp/paci-runtime/secure_infer/serving/entrypoint.py
docker exec \
    -e MODEL_PATH=/checkpoint -e MODEL_NAME="deepseek70b-pp${pp_size}" \
    -e SERVE_MODE=plain -e PP_SIZE="$pp_size" -e TP_SIZE=8 \
    -e DISTRIBUTED_EXECUTOR_BACKEND=ray \
    -e GPU_MEMORY_UTILIZATION=0.95 -e MAX_NUM_SEQS=1 \
    -e MAX_MODEL_LEN=8192 -e MAX_NUM_BATCHED_TOKENS=1024 \
    -e HOST=127.0.0.1 -e PORT=18080 \
    "$container" /bin/bash -lc '
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        exec python3 /tmp/paci-runtime/secure_infer/serving/entrypoint.py
    '
