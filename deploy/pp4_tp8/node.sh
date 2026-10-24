#!/usr/bin/env bash
set -euo pipefail

# Run on each node. No host installation or other service is changed.
role=${1:?Usage: bash node.sh head|worker LOCAL_IP}
local_ip=${2:?Supply the local management IP}
head_ip=${HEAD_IP:-192.168.0.235}
ray_port=${RAY_PORT:-26479}
pp_size=${PP_SIZE:-3}
case $pp_size in
    2|3|4) ;;
    *) echo "PP_SIZE must be 2, 3, or 4 for this cluster" >&2; exit 1 ;;
esac
image=quay.io/ascend/vllm-ascend:v0.14.0rc1
expected_image=sha256:4154b3e344b4703e6c7c5bb86306ebdfa531ca3a826be861298d2a5594de3d9f
container=paci-pp${pp_size}-tp8
volume=${MODEL_VOLUME:-paci-qwen235b-models}

[[ $role == head || $role == worker ]] || { echo "Unknown role: $role" >&2; exit 1; }
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$expected_image" ]] || {
    echo "The installed image differs from the four-node reference image" >&2; exit 1;
}
for i in {0..7}; do
    health=$(npu-smi info -t health -i "$i" -c 0)
    grep -Eq 'Health Status[[:space:]]*:[[:space:]]*OK' <<< "$health" || {
        echo "NPU $i is not healthy: $health" >&2; exit 1;
    }
done
status=$(npu-smi info)
if grep -Eq '\|[[:space:]]*[0-7][[:space:]]+0[[:space:]]*\|[[:space:]]*[0-9]+' <<< "$status"; then
    echo "An NPU process is already running; this node will not be started" >&2
    exit 1
fi
devices=()
for i in {0..7}; do devices+=(--device "/dev/davinci$i"); done
model_subdir=${MODEL_SUBDIR:-}
if [[ -n "${MODEL_BIND_SOURCE:-}" ]]; then
    model_mounts=(--mount "type=bind,source=$MODEL_BIND_SOURCE,target=/checkpoint,readonly")
    model_subdir=
else
    docker volume inspect "$volume" >/dev/null
    model_mounts=(-v "$volume:/models:ro")
fi
ray_args=(--node-ip-address="$local_ip" --disable-usage-stats --block)
if [[ $role == head ]]; then
    ray_args+=(--head --port="$ray_port" --include-dashboard=false)
else
    ray_args+=(--address="$head_ip:$ray_port")
fi
docker run -d --name "$container" --network host --shm-size 16g \
    --privileged \
    "${devices[@]}" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    "${model_mounts[@]}" \
    -e PACI_MODEL_SUBDIR="$model_subdir" \
    -e HCCL_IF_IP="$local_ip" -e VLLM_HOST_IP="$local_ip" \
    -e GLOO_SOCKET_IFNAME=enp67s0f5 -e TP_SOCKET_IFNAME=enp67s0f5 \
    -e RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --entrypoint /bin/bash "$image" -lc \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh; if [[ -n "$PACI_MODEL_SUBDIR" ]]; then ln -s "/models/$PACI_MODEL_SUBDIR" /checkpoint; fi; exec ray start "$@"' bash "${ray_args[@]}"
