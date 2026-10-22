"""Start the configured vLLM server with the runtime connector."""
import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    model = os.environ.get("MODEL_PATH", "")
    if not model:
        raise SystemExit("MODEL_PATH is required")
    root = Path(__file__).resolve().parents[2]
    cmd = [
        "vllm", "serve", model,
        "--host", os.environ.get("HOST", "127.0.0.1"),
        "--port", os.environ.get("PORT", "8080"),
        "--served-model-name", os.environ.get("MODEL_NAME", "runtime-model"),
        "--tensor-parallel-size", os.environ.get("TP_SIZE", "1"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "1"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "8192"),
        "--max-num-batched-tokens", os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.61"),
        "--enable-chunked-prefill", "--enforce-eager", "--block-size", "128",
        "--scheduler-cls", "secure_infer.reuse.connector.RuntimeScheduler",
        "--kv-transfer-config",
        '{"kv_connector":"RuntimeConnector","kv_connector_module_path":"secure_infer.reuse.connector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":%s,"block_size":128,"spec_name":"NPUOffloadingSpec","spec_module_path":"secure_infer.kv_offload.npu_predict"}}'
        % os.environ.get("NUM_CPU_BLOCKS", "512"),
    ]
    mode = os.environ.get("SERVE_MODE", "runtime")
    if mode == "plain":
        del cmd[cmd.index("--scheduler-cls"):]
    elif mode != "runtime":
        raise SystemExit("SERVE_MODE must be runtime or plain")
    cmd.extend(sys.argv[1:])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
