# secure_infer

## 项目简介

secure_infer 是面向 Ascend NPU 的 LLM 推理运行时，基于 vLLM 和 vLLM-Ascend 实现主机 KV cache 卸载与复用。

- **安全传输**：使用 AES-GCM 保护主机与设备之间的 KV 数据，每层在解密、认证和写入完成后执行 attention。
- **加密预取**：根据 KV 传输地址预测后续传输，提前执行主机加密和设备密钥流生成。
- **前缀复用**：结合重算、逐层重载与后缀计算的耗时，在 prefill 前选择前缀切分点。
- **异步写回**：在后台将 KV 写回主机，认证完成后发布缓存条目。

主机与设备通过临时 X25519 密钥交换建立会话密钥。设备身份认证依赖外部可信配给，密钥交换本身不验证设备证书。

## 运行指南

### 1. 环境准备

需要 Linux、Ascend NPU 驱动、本地模型权重，以及相互兼容的 CANN、PyTorch、torch-npu、vLLM 和 vLLM-Ascend。可选择 Docker 部署或在已有推理环境中直接运行。

```bash
git clone https://github.com/ZiyuanLuo/secure_infer.git
cd secure_infer
cp .env.example .env
```

编辑 `.env`，设置模型路径和设备。以下为单卡配置示例，模型及 KV cache 需要能放入所选设备：

```dotenv
MODEL_PATH=/path/to/model
MODEL_NAME=secure-infer
DEVICE_IDS=0
TP_SIZE=1
PP_SIZE=1
HOST=127.0.0.1
PORT=8080
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=1
GPU_MEMORY_UTILIZATION=0.85
NUM_CPU_BLOCKS=128
COMPUTE_CHUNK=1024
LOAD_CHUNK=2048
SERVE_MODE=runtime
ENABLE_H2D=1
ENABLE_D2H=1
```

`NUM_CPU_BLOCKS` 决定主机 KV 池容量，每块保存 128 个 token 在所有层的 KV；内存开销随模型及 TP 分片变化。计算、加载粒度须为 128 的正整数倍。

### 2. Docker 启动

宿主机需安装 Docker，运行账号需有 Docker 使用权限。`.env` 中的 `BASE_IMAGE` 指定 Ascend 推理基础镜像，`IMAGE` 指定本项目镜像名称。

```bash
./start.sh
```

也可以在启动时指定模型目录：

```bash
./start.sh /path/to/model
```

脚本在本地镜像不存在时构建镜像，挂载项目、模型和驱动，编译传输库及 AICPU AES-GCM 算子，然后在后台启动服务。

```bash
./manage.sh status  # 查看容器状态
./manage.sh logs    # 查看最近的服务日志
```

容器运行不等于模型已经就绪。等待模型和主机 KV 池初始化完成，再检查：

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

停止并移除容器：

```bash
./manage.sh stop
```

修改 `.env` 后，停止并重新启动，使配置生效。重新构建镜像可执行：

```bash
docker build --build-arg BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1 \
  -t secure-infer:local .
```

镜像名称和基础镜像应与 `.env` 中的 `IMAGE`、`BASE_IMAGE` 一致。

### 3. 在已有环境中运行

先激活安装了推理依赖的 Python 环境，加载本机 CANN 环境；安装了 ATB 时也加载对应环境：

```bash
source /path/to/ascend-toolkit/set_env.sh
source /path/to/nnal/atb/set_env.sh
```

在 `.env` 中将 `ASCEND_ROOT` 设置为 CANN 安装目录，将 `PYTHON` 设置为该环境的解释器。编译还需 C++17 编译器、Make、CMake 和 OpenSSL 开发库。

```bash
./configure.sh
./run.sh
```

`configure.sh` 检查 Python 依赖，构建 `libsecure_infer.so`，并在 `runtime` 模式下构建 AICPU 认证算子。`run.sh` 加载配置并在前台启动服务，按 `Ctrl+C` 停止。

### 4. 张量并行

单机四卡 TP 示例，在 `.env` 中设置：

```dotenv
MODEL_PATH=/path/to/large-model
DEVICE_IDS=0,1,2,3
TP_SIZE=4
PP_SIZE=1
DISTRIBUTED_EXECUTOR_BACKEND=mp
MAX_NUM_SEQS=1
```

其余启动步骤相同。每个 rank 处理自己的 KV 分片和密码会话；主机池占用需按所有 rank 合计。

多机原生明文推理的 Ray/PP 部署步骤见 [多机部署指南](deploy/pp4_tp8/README.md)。

### 5. 发送请求

服务提供 OpenAI 兼容的 completions 接口，可直接发送普通文本：

```bash
curl --fail http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "secure-infer",
    "prompt": "Explain how a KV cache works.",
    "max_tokens": 64,
    "temperature": 0,
    "stream": false
  }'
```

也可以用仓库脚本指定共享前缀和后缀长度。Docker 部署使用：

```bash
./manage.sh query --prefix-tokens 512 --suffix-tokens 1024 --output-tokens 16
```

本地部署使用：

```bash
./query.sh --prefix-tokens 512 --suffix-tokens 1024 --output-tokens 16
```

脚本先生成共享前缀的 KV，等待异步写回，再清空设备前缀缓存、保留主机缓存，随后发送前缀加后缀请求。前缀长度建议为 128 的倍数，输入与输出 token 总数不能超过 `MAX_MODEL_LEN`。

常用选项：

| 选项 | 用途 |
|---|---|
| `--prefix-tokens` / `--suffix-tokens` | 指定共享前缀和新增后缀的 token 数 |
| `--output-tokens` | 指定生成 token 数 |
| `--repeat` | 重复发送目标请求 |
| `--store-wait` | 设置主机写回等待秒数 |
| `--no-device-reset` | 保留设备前缀缓存 |
| `--output` | 将请求结果保存为 JSON |
| `--url` / `--model` | 覆盖服务地址和模型名称 |

```bash
./query.sh --prefix-tokens 6144 --suffix-tokens 2048 \
  --output-tokens 16 --output runs/request.json
```

该示例需先将 `MAX_MODEL_LEN` 设为至少 8208。结果包含实际 token 数、TTFT、TPOT、总时延和响应文本；缓存重置是否成功可结合服务日志和实际重载记录确认。

### 6. 配置参考

| 参数 | 含义 |
|---|---|
| `MODEL_PATH` / `MODEL_NAME` | 本地模型目录与 API 模型名称 |
| `DEVICE_IDS` | 可见 NPU 编号，多个编号用逗号分隔 |
| `TP_SIZE` / `PP_SIZE` | 张量并行度与流水线并行度 |
| `DISTRIBUTED_EXECUTOR_BACKEND` | 执行后端，单机默认 `mp` |
| `HOST` / `PORT` | 服务监听地址与端口 |
| `MAX_MODEL_LEN` | 单请求输入与输出 token 总长度上限 |
| `MAX_NUM_BATCHED_TOKENS` | 单步调度 token 预算 |
| `MAX_NUM_SEQS` | 服务端最大并发序列数；客户端发送并发需另行设置 |
| `GPU_MEMORY_UTILIZATION` | 设备内存预算比例 |
| `NUM_CPU_BLOCKS` | 主机 KV 池块数 |
| `COMPUTE_CHUNK` / `LOAD_CHUNK` | 前缀重算与重载粒度，单位为 token |
| `SERVE_MODE` | `runtime` 使用安全传输和复用模块；`plain` 使用原生 vLLM |
| `ENABLE_H2D` / `ENABLE_D2H` | `runtime` 模式下两个传输方向的密码钩子开关 |
| `RESULT_DIR` | 运行记录目录，默认 `runs/current` |
| `GHASH_LIBRARY` | AICPU 认证库路径，默认使用项目内构建产物 |
| `AIV_KEYSTREAM_LIBRARY` | 可选设备密钥流后端路径 |
| `ASCEND_ROOT` / `PYTHON` | CANN 安装目录与 Python 解释器 |
| `BASE_IMAGE` / `IMAGE` | Docker 基础镜像与项目镜像名称 |
| `CONTAINER_NAME` | 可选的固定容器名称 |

`.env`、模型权重、运行日志和构建产物保留在本地。

### 7. 常见问题

- **依赖或认证库找不到**：确认已加载 CANN 环境，`ASCEND_ROOT` 指向正确目录，再运行 `./configure.sh`。
- **启动时设备内存不足**：确认设备上没有残留服务，按模型大小调整 TP 或设备内存预算。
- **主机内存不足或初始化较慢**：减小 `NUM_CPU_BLOCKS`。主机 KV 池按模型层分配，大池需要更多内存及初始化时间。
- **请求长度超限**：提高 `MAX_MODEL_LEN` 后重新启动，或缩短输入及输出。
- **主机缓存没有复用**：确认前缀相同、写回已完成且设备缓存已清空；检查实际 host-cache match 与 H2D 重载日志。
- **需要从另一台机器访问**：使用 SSH 端口转发，将本地端口映射到服务器的 `127.0.0.1:8080`；API 请求中的模型名称须与 `MODEL_NAME` 一致。
