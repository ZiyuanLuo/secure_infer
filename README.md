# secure_infer

面向 Ascend NPU 的 KV cache 卸载与复用运行时，包含 vLLM 服务、逐层加载、预测性加密、异步写回、前缀复用决策和 STOP/MORE 控制。

## 代码结构

~~~text
.
├── .env.example          # 本地配置模板
├── Dockerfile            # 容器环境
├── Makefile              # 本地库与认证内核构建
├── start.sh              # 一键构建并在后台启动容器服务
├── manage.sh             # 状态、日志、停止、测试和请求入口
├── configure.sh          # 依赖检查与本地编译
├── run.sh                # 配置推理环境并启动 vLLM
├── query.sh              # 读取配置并发送请求
├── request.py            # 精确前后缀构造、预热和流式计时
├── infer.py              # 独立推理示例
└── secure_infer/
    ├── native/           # C++ 传输、密码钩子与认证内核
    ├── kv_offload/       # 主机缓存、逐层等待与传输生命周期
    ├── reuse/            # 在线代价估计、策略选择和 STOP/MORE
    ├── serving/          # vLLM 服务入口
    └── tests/            # 决策、层间交接和生命周期测试
~~~

## 一键启动

宿主机使用 Linux、Ascend NPU 驱动和 Docker。执行账号需要 Docker 使用权限。容器提供 CANN、PyTorch、torch-npu、vLLM 和 vllm-ascend，模型由本地权重目录提供。设备、驱动与容器中的 CANN 版本需兼容。

~~~bash
cp .env.example .env
vi .env
./start.sh
~~~

在 .env 中设置 MODEL_PATH、DEVICE_IDS、TP_SIZE、上下文长度。也可通过参数指定模型：

~~~bash
./start.sh 模型目录
~~~

脚本按需构建镜像，自动挂载项目、模型和驱动，编译 libsecure_infer.so，再后台启动。容器名由项目位置和账号生成；容器内使用 root 访问设备；停止时将生成文件的所有权归还调用账号。移动项目或更换账号无需改脚本。

~~~bash
./manage.sh status
./manage.sh logs
./manage.sh test
./manage.sh stop
~~~

status 显示容器状态；模型加载完成以日志中的 Application startup complete 和健康检查为准。stop 停止并移除项目容器，之后可再次执行 start.sh。

~~~bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
~~~

已安装完整 Ascend 推理依赖时，可直接执行 ./configure.sh 和 ./run.sh。

## 配置参数

| 参数 | 含义 |
|---|---|
| MODEL_PATH / MODEL_NAME | 模型目录和 API 模型名称 |
| DEVICE_IDS / TP_SIZE | 可见 NPU 编号与张量并行度 |
| HOST / PORT | 服务监听地址和端口 |
| MAX_MODEL_LEN | 输入加输出 token 总长度上限 |
| MAX_NUM_BATCHED_TOKENS | 单步 token 预算 |
| MAX_NUM_SEQS | 当前控制器使用 1 |
| GPU_MEMORY_UTILIZATION | 设备内存使用比例 |
| NUM_CPU_BLOCKS | 主机 KV 缓存块数 |
| COMPUTE_CHUNK / LOAD_CHUNK | 计算与加载粒度，必须为正的 128 倍数 |
| PREFIX_TOKENS / SUFFIX_TOKENS | 请求脚本默认前后缀 token 数 |
| SERVE_MODE | runtime 使用项目连接器；plain 使用原生 vLLM 且不预加载本地库 |
| ENABLE_H2D / ENABLE_D2H | 本地库钩子开关；Python 连接器仍管理传输任务 |
| GHASH_LIBRARY / AIV_KEYSTREAM_LIBRARY | 可选认证与密钥流库 |
| ASCEND_ROOT | CANN 目录；为空时在容器中自动查找 |
| PYTHON | 容器内的 Python 解释器 |
| BASE_IMAGE / IMAGE | 基础镜像和构建镜像名称 |
| CONTAINER_NAME | 可选固定容器名称 |
| RESULT_DIR | 结果目录，默认 runs/current |

.env、权重、日志和编译产物不提交到版本库。部署路径只写入本地 .env。可选设备库放入项目目录后可配置为项目相对路径。

## 构造前缀和后缀

请求脚本调用服务端 tokenizer，用 token ID 数组精确构造输入。每次先发送只有共享前缀的请求，等待异步 D2H 写回，再调用 `/reset_prefix_cache?reset_external=false` 清空设备缓存并保留主机缓存，最后测量“共享前缀 + 后缀”。输出通过 ignore_eos 固定 token 数。服务入口默认设置 `VLLM_SERVER_DEV_MODE=1` 以启用此实验接口。

~~~bash
./manage.sh query --prefix-tokens 512 --suffix-tokens 512 --output-tokens 16

./manage.sh query --prefix-tokens 256 --suffix-tokens 1024 --output-tokens 16

# 6K 前缀、8K 后缀：先将 MAX_MODEL_LEN 设为至少 14352 并重启
./manage.sh query --prefix-tokens 6144 --suffix-tokens 8192 --output-tokens 16 --output runs/p6144_s8192.json
~~~

服务外部也可直接用 Python 标准库发送请求：

~~~bash
python3 request.py --url http://127.0.0.1:8080 --model secure-infer --prefix-tokens 512 --suffix-tokens 1024
~~~

PREFIX_TOKENS 和 SUFFIX_TOKENS 仅设置请求脚本默认值；控制器按真实缓存匹配确定命中长度。缓存块大小为 128 tokens，建议前缀采用 128 的倍数。输入加输出长度必须小于等于 MAX_MODEL_LEN。

输出包含实际输入和输出 token 数、TTFT、TPOT、端到端总时延和响应文本。TTFT 从发起 HTTP 请求到首个非空文本块，TPOT 根据首末文本块时间和服务端输出 token 数计算，总时延计到流式完成标记。流式块可能合并 token，因此为客户端观测口径。

`--store-wait 8` 设置写回等待秒数。每次重复前都会清空设备缓存。`--no-device-reset` 可用于设备缓存命中测试。HTTP 200 不代表设备缓存一定清空成功：异步任务尚未释放块时，vLLM 可能拒绝清空。测量时需结合服务日志中的成功清空记录、实际 host-cache match 和 H2D 加载记录确认路径。脚本为每次目标请求生成不同的后缀，避免重复命中已写回的后缀 KV；仍需按实际缓存命中长度核对。

## 明文对照

保持模型、设备、上下文与请求参数一致，分别将 .env 中 SERVE_MODE 设置为 runtime 和 plain，停止并重启服务后测量：

~~~bash
./start.sh
./manage.sh query --prefix-tokens 512 --suffix-tokens 1024 --output runs/result.json
./manage.sh stop
~~~

两个模式分别完成预热，使用相同输出长度和缓存预热/驱逐流程。

## 实现与实验进度

已有主机缓存匹配、逐层加载、在线估计、STOP/MORE 和异步写回代码。已完成 8 卡 70B 服务启动、短请求和三组 GPU 前缀缓存与明文的初步对照，每组一次。此前三组未进入完整主机 KV 重载路径，不能据此确定该路径的性能。

密钥供应与对端密钥协商集成尚未完成。进程内密码上下文注册完成不表示对端密钥协商完成。完整安全路径与流水化解压缩的端到端效果尚未由这轮实验验证。

## 本次整理验证

- 本地库重新编译成功，17 项单元测试通过。
- 使用新 start.sh 在 8 卡 70B 模型上完成启动，健康检查返回 HTTP 200。
- 前缀 128 / 后缀 256、前缀 256 / 后缀 1024 两组请求均完整生成，实际输入长度分别为 384、1280 tokens。
- 容器内 manage.sh query 和宿主机 Python 请求均已验证。
- 验证沿用本机已有的 CANN 8.5、vLLM 0.14.1 环境镜像；Dockerfile 默认的公开基础镜像尚未在本轮重新拉取验证。
- 本次运行保留 ENABLE_H2D=0、ENABLE_D2H=0 的原测试配置，未新增完整安全路径的性能结论。
