#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
ID_SUFFIX=$(printf '%s' "${ROOT_DIR}:$(id -u)" | cksum | cut -d ' ' -f 1)
NAME="${CONTAINER_NAME:-secure-infer-${ID_SUFFIX}}"
case "${1:-status}" in
  status) docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$NAME" ;;
  logs) docker logs --tail 80 "$NAME" ;;
  stop)
    docker exec --user 0 --workdir "$ROOT_DIR" "$NAME" bash -c 'chown -R "$HOST_UID:$HOST_GID" runs .cache libsecure_infer.so secure_infer/native' 2>/dev/null || true
    docker stop --time 30 "$NAME" >/dev/null; docker rm "$NAME" >/dev/null ;;
  test) docker exec --workdir "$ROOT_DIR" "$NAME" ./run_test.sh ;;
  query) shift; docker exec --workdir "$ROOT_DIR" "$NAME" ./query.sh "$@" ;;
  *) printf '用法: ./manage.sh {status|logs|stop|test|query [参数]}\n' >&2; exit 2 ;;
esac
