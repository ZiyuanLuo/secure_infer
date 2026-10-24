# CacheGen 量化规则与 NPU 熵解码原型

日期：2026-10-24。此目录保存独立原型，尚未接入 secure_infer 的真实请求。

## 实现

沿用 CacheGen 每 token 最大绝对值缩放规则，选择 8-bit 符号。符号再经 DEFLATE 无损编码，设备端用 miniz 3.0.2 解码。原版 CacheGen 的 CUDA 算术编码器并未移植；本实现替换了熵编码格式。
一次 AICPU 调用完成熵解码和 ARM NEON BF16 恢复。每层按 token 范围分为 8 条独立码流，在不同设备流上提交，保留可独立认证的边界。这个改动消除了单个长熵码流的串行依赖。

## 当前证据

真实数据为先前导出的 70B 模型首层 K/V 投影，2048 tokens、8 KV heads、128 head dimension。只有首层；不是全模型 KV 或任务质量评测。所有恢复值与 CPU 量化参考的 BF16 位表示一致，不能称为与原始 BF16 无损一致。
在 910B4 / CANN 8.5.1 的空闲 NPU 0 测量，宿主环境未升级；使用已有容器镜像。各配置预热一次、计时三次，以下为中位数。计时包含 Python 提交与设备同步，不含量化编码、H2D、密码处理或模型计算。压缩率计入 FP32 scales、每条码流 16 字节对齐及80字节认证记录预算；尚未计入最终索引表或整个缓存容器。

| 数据（2048 tokens） | 单流压缩率 | 八流压缩率 | 单流恢复 ms | 八流恢复 ms |
|---|---:|---:|---:|---:|
| K | 2.417× | 2.416× | 21.684 | 5.524 |
| V | 3.163× | 3.157× | 23.870 | 6.089 |

无损 BF16 的 byte shuffle / token XOR + DEFLATE 在相同样本上约1.27–1.45×，未达到2×。4-bit路径虽有更高体积收益，但首层V的归一化均方误差约0.20；8-bit约0.00134。这是张量误差，不是任务得分下降。

## 尚需完成

1. 将压缩字节（含scales、码流索引）接入实际 AES-GCM 和分块流水，比较同精度未压缩符号路径，区分量化收益和熵编码收益。
2. 接入真实多层KV卸载与重载，将量化/编码和元数据传输纳入成本，测TTFT、TPOT和E2E。
3. 在中后层、不同输入和任务集上验证压缩率与回答质量。
4. 并行解码约达到4×改善，但与认证共同运行时是否争用AICPU还未测，不能宣称端到端收益。

## 参考来源

- CacheGen 官方源码： https://github.com/UChi-JCL/CacheGen ，核对版本 `6bed34ca9d495289955ed754cf2ad0c43346ee48`。
- miniz 3.0.2： https://github.com/richgel999/miniz/tree/3.0.2 。保留源码版权，使用独立 inflate-only adapter，未依赖设备系统的libz。

数据文件 characterization.json 和 device_decode_lanes*.json 保存在本地并由本目录 .gitignore 排除。


## 2026-10-24 后续实现：Bitshuffle + LZ4

DEFLATE 的解码成本较高，当前优先路线改为 CacheGen 风格的8-bit符号、128符号分组的位重排和LZ4。使用 LZ4 1.9.4 官方源码的 freestanding 解码器；位逆变换与BF16恢复共用一次NEON遍历。该组合采用既有 Bitshuffle 思路，不声称发明新的熵编码。

`encoder.cpp` 在主机上用ARM NEON执行 BF16读取、逐token最大值归约、8-bit量化和位重排，使用OpenMP并行，并调用系统LZ4快速编码。`test_encoder.py` 检查符号、缩放值、位重排与NumPy参考完全相等，并独立解开LZ4流。1/128/512/2048-token测试均通过。

组件流水包含真实 pinned H2D、AES-GCM、AES-CTR、恢复和GEMM。12个128-token块上，位重排+LZ4压缩率2.376倍；四条恢复流的总时间1.730 ms。此指标是组件时延，不能称为整模型E2E。

## 整模型验证程序

`model_bench.py` 使用标准Transformers模型，直接生成前缀的全部层KV，再从主机恢复缓存、计算未缓存后缀、逐token生成输出。`RestoredCache` 在各层首次更新前等待该层全部KV恢复及认证成功；后续层在独立流上提前恢复。使用独立后台线程与有界lookahead。输入来自给定文本文件，不以重复token填充。

配置包括 `plain`、`bf16`、`int8`、`lz4`、`bslz4`、`bslz4_fast`；除明文外均执行真实AES-GCM。`bf16_serial`用于取消加载与模型计算重叠的对照。当前已测试的 BF16 与压缩主结果均开启同样的逐层流水。

结果必须区分：
- `encode_and_stage_ms`：主机缓存已取得之后，编码、认证加密、pinned staging和设备缓冲准备成本；不包含最初前缀prefill或D2H本身。
- `e2e_ms`：缓存已准备好后，从安全恢复启动到本次请求最后一个生成token完成。包含真实模型prefill/decode和恢复等待，不包含上述预先缓存准备。
- `ratio`：全部层原始BF16字节除以实际编码数据、tag、AAD、nonce及索引预算字节；不是单层估计。

原生编码将4K缓存的准备时间从Python高压缩版本的约28.5秒降到约2.7秒；BF16基线约2.3秒。因此不能把暖缓存请求的降幅直接说成一次“生成缓存+加载缓存”生命周期的降幅。

已测模型为 Qwen3-14B，40层，Ascend 910B3单张空闲卡，CANN8.5.1，输出16 tokens，代码文本输入。4K前缀/128后缀的第二组配对（各方法已运行一次后）：

| 配置 | 全层压缩率 | TTFT ms | TPOT ms | 请求E2E ms | 缓存准备 ms |
|---|---:|---:|---:|---:|---:|
| BF16安全逐层加载 | 1.000× | 516.94 | 84.72 | 1787.69 | 2301.22 |
| Bitshuffle+LZ4快速编码 | 2.036× | 324.83 | 83.88 | 1583.06 | 2700.59 |

该配对的请求E2E降低11.45%，TTFT降低37.16%。16个输出token完全相同。只传INT8符号的单次结果E2E为1589.51 ms，TTFT为311.46 ms；这一点不足以证明熵编码相对INT8有独立整模型时延收益。压缩相比BF16的收益包含量化的体积收益。相同输出不等价于完成了任务质量评测。

目前整模型入口为独立Transformers缓存加载器，尚未接入项目vLLM连接器及其PACI预测器、控制器。八卡70B验证也尚未完成。

## 运行

加载既有CANN环境，并选择空闲设备后，在仓库根目录运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=6 bash experiments/cachegen_npu/run_model.sh   --model MODEL_DIRECTORY   --text-file ../../secure_infer/native/secure_infer.cpp   --prefix 4096 --suffix 128 --output-tokens 16   --modes plain,bf16,bslz4_fast,int8,bf16,bslz4_fast   --output model_results.json
```

脚本在实验目录工作；文本相对路径也相对于实验目录。需要NumPy、PyTorch/torch-npu、Transformers、cryptography、系统liblz4、OpenMP、CANN ASC/AICPU编译器及Ninja。未改变系统安装或升级环境。

- [CacheGen官方源码](https://github.com/UChi-JCL/CacheGen)
- [Bitshuffle官方说明](https://github.com/kiyo-masui/bitshuffle)
- [LZ4 1.9.4源码](https://github.com/lz4/lz4/tree/v1.9.4/lib)


## 完整传输计时：8K前缀 / 512后缀

后续协议将tag和AAD与密文放在同一pinned传输记录中，在请求的H2D计时内实际复制。旧结果中的tag和AAD预先在设备上，因此最终完整传输结果应以本组为准。模型仍为Qwen3-14B、910B3、40层、输出16 tokens；同一进程交错运行BF16与压缩各两次，下表使用第二次。

| 配置 | 全层压缩率 | TTFT ms | TPOT ms | E2E ms | 缓存编码和准备 ms |
|---|---:|---:|---:|---:|---:|
| BF16安全流水 | 1.000× | 1033.87 | 76.39 | 2179.64 | 5036.11 |
| 原生Bitshuffle+LZ4流水 | 2.034× | 646.91 | 78.53 | 1824.87 | 5939.57 |

请求E2E降低16.28%，TTFT降低37.43%。所有40层在消费前检查认证和解码状态，16个输出token一致。TPOT没有显示收益。压缩的缓存准备成本仍可能更高；将本次缓存准备和一次请求相加时，这个配对尚无净收益。这里证明的是已经生成并准备好的KV缓存重载收益，不证明首次缓存创建和一次重载的总收益。编码器应进一步与异步写回集成，并在真实缓存复用轨迹上验证摊销。

完整传输结果保存在本地 `model_wire_p8192_s512.json` 和同名日志；不提交原始结果。后续工作保持：接入现有vLLM/PACI路径、测八卡70B，以及验证更广的输入与质量。当前目标仍在推进。
