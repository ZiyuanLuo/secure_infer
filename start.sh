#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
[[ -f .env ]] || cp .env.example .env
set -a
source .env
set +a
[[ -z "${1:-}" ]] || export MODEL_PATH="$1"
: "${MODEL_PATH:?请设置 .env 中的 MODEL_PATH，或执行 ./start.sh 模型目录}"
export MODEL_PATH=$(realpath -m -- "$MODEL_PATH")
docker info >/dev/null
IMAGE="${IMAGE:-secure-infer:local}"
BASE_IMAGE="${BASE_IMAGE:-quay.io/ascend/vllm-ascend:v0.14.0rc1}"
ID_SUFFIX=$(printf '%s' "${ROOT_DIR}:$(id -u)" | cksum | cut -d ' ' -f 1)
NAME="${CONTAINER_NAME:-secure-infer-${ID_SUFFIX}}"
if docker container inspect "$NAME" >/dev/null 2>&1; then
  printf '容器已存在: %s；执行 ./manage.sh status 或 ./manage.sh stop\n' "$NAME" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" .
fi
args=(run -d --name "$NAME" --network host --ipc host --privileged
  --label "secure_infer.project=$ROOT_DIR"
  --workdir "$ROOT_DIR"
  --mount "type=bind,source=$ROOT_DIR,target=$ROOT_DIR"
  --env "MODEL_PATH=$MODEL_PATH" --env "MODEL_OVERRIDE=$MODEL_PATH"
  --env "HOME=$ROOT_DIR/.cache"
  --env "USER=secure-infer" --env "LOGNAME=secure-infer"
  --env "ASCEND_GLOBAL_LOG_LEVEL=3" --env "ASCEND_SLOG_PRINT_TO_STDOUT=0"
  --env "PYTHONPYCACHEPREFIX=$ROOT_DIR/.cache/pycache")
if [[ "$MODEL_PATH" != "$ROOT_DIR/"* ]]; then
  args+=(--mount "type=bind,source=$MODEL_PATH,target=$MODEL_PATH,readonly")
fi
for system_path in /dev /usr/local/Ascend/driver; do
  [[ ! -e "$system_path" ]] || args+=(--mount "type=bind,source=$system_path,target=$system_path")
done
for system_path in /usr/local/bin/npu-smi /etc/ascend_install.info; do
  [[ ! -e "$system_path" ]] || args+=(--mount "type=bind,source=$system_path,target=$system_path,readonly")
done
args+=(--user 0:0 --env "HOST_UID=$(id -u)" --env "HOST_GID=$(id -g)")
for group_id in $(id -G); do args+=(--group-add "$group_id"); done
mkdir -p .cache runs
docker "${args[@]}" "$IMAGE" bash -c '
  set -e
  for setup in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann-*/set_env.sh; do
    if [[ -f "$setup" ]]; then source "$setup"; break; fi
  done
  umask 0022
  ./configure.sh
  chown "$HOST_UID:$HOST_GID" libsecure_infer.so
  exec ./run.sh
'
printf '容器: %s\n日志: ./manage.sh logs\n状态: ./manage.sh status\n' "$NAME"
