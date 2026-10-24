# 多机 PP 部署（TP=8）

每个节点使用八张 Ascend NPU。`PP_SIZE` 决定流水线阶段数，每个节点负责一个阶段。`node.sh` 支持 PP=2、3、4；`serve_deepseek.sh` 默认 PP=2。本目录名称沿用最初的四节点配置。

已验证：两台服务器、16张910B4、DeepSeek-R1-Distill-Llama-70B真实BF16权重，PP=2、TP=8、Ray后端、原生明文推理。两组512和2048输入token的请求均完成32个输出token。模型共80层，两个阶段各40层。

## 环境

使用已有镜像 `quay.io/ascend/vllm-ascend:v0.14.0rc1`，包含 CANN 8.5.0、vLLM 0.14.1+empty、vLLM-Ascend 0.14.0rc1、Ray 2.48.0。各节点使用相同镜像，启动脚本会检查本次验证使用的镜像ID。

各节点的管理网络和NPU网络需要互通。本机配置的管理网卡为 `enp67s0f5`，脚本用它设置 GLOO/TP socket 接口。容器使用 host 网络和 privileged 模式访问设备。启动前脚本检查八张卡的健康及占用，不停止其他任务。

## 启动 Ray 节点

以下命令都从项目根目录执行。先在每个节点设置：

```bash
export PP_SIZE=2
export HEAD_IP=192.168.0.235  # Ray head地址，所有节点保持一致
export NODE_IP=192.168.0.235 # 改成当前节点的管理地址
export MODEL_VOLUME=your_model_volume
```

`MODEL_VOLUME` 为该节点已有的 Docker volume 名称。模型可以通过以下任一种方式提供：

- 本机模型目录：将 `MODEL_BIND_SOURCE` 设为完整模型目录，脚本只读挂载该目录。
- volume中的模型：将 `MODEL_SUBDIR` 设为模型在volume内的相对目录，如 `DeepSeek-R1-Distill-Llama-70B`。

两种方式都会让服务使用相同的容器内模型别名。各节点须包含完整且一致的权重、config和tokenizer文件。

```bash
# 使用volume中的模型
export MODEL_SUBDIR=DeepSeek-R1-Distill-Llama-70B

# head节点
bash deploy/pp4_tp8/node.sh head "$NODE_IP"

# 其余节点：先将NODE_IP改成该节点地址，再执行
bash deploy/pp4_tp8/node.sh worker "$NODE_IP"
```

Ray head使用26479端口，可通过 `RAY_PORT` 调整，所有节点保持一致。容器名根据PP自动生成，例如 `paci-pp2-tp8`。在head检查资源：

```bash
docker exec "paci-pp${PP_SIZE}-tp8" ray status
```

PP=2时应看到两个节点、16个NPU资源。PP=3或4时，在相应数量的节点上设置同一个 `PP_SIZE` 并启动Ray。

## 启动 DeepSeek 服务

仅在head执行：

```bash
PP_SIZE=2 bash deploy/pp4_tp8/serve_deepseek.sh
```

脚本调用项目的 `secure_infer.serving.entrypoint`，传入PP、TP和Ray后端参数。当前参数为：

| 参数 | 值 |
|---|---|
| PP / TP | PP_SIZE / 8 |
| 服务模式 | plain |
| 显存利用率 | 0.95 |
| 最大上下文 | 8192 tokens |
| 每步token预算 | 1024 |
| 最大并发 | 1 |
| KV块大小 | 128 tokens |
| 执行模式 | eager |
| API端口 | 18080 |

服务就绪以 `Application startup complete` 和健康接口为准：

```bash
curl --noproxy '*' http://127.0.0.1:18080/health
curl --noproxy '*' http://127.0.0.1:18080/v1/models
curl --noproxy '*' http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek70b-pp2","messages":[{"role":"user","content":"What is 2 + 3? Answer briefly."}],"max_tokens":96,"temperature":0}'
```

PP=3或4时，服务名相应变为 `deepseek70b-pp3` 或 `deepseek70b-pp4`。调整PP后需要重启节点容器和模型服务。

## 验证范围

本轮完成了真实BF16权重加载、两机PP通信、prefill、decode及流式请求。PP=3和4是可配置选项，本轮实测为PP=2。此处启动原生明文服务，未接PACI密码库、KV offload、controller或压缩。

## 停止

在每个节点执行，只操作本次创建的容器：

```bash
docker stop "paci-pp${PP_SIZE}-tp8"
docker rm "paci-pp${PP_SIZE}-tp8"
```

模型volume和宿主机模型目录保留。

## 参考

- [对应版本的vLLM-Ascend多机Ray部署说明](https://github.com/vllm-project/vllm-ascend/blob/v0.14.0rc1/docs/source/tutorials/ray.md)
