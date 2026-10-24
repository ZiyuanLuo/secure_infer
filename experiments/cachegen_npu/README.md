# CacheGen 风格量化与 NPU 压缩恢复

日期：2026-10-24。此目录保存独立原型，采用每 token 8-bit 量化、Bitshuffle 和 LZ4，在 AICPU 上融合位逆变换与 BF16 恢复。已用真实 Transformers 模型验证逐层安全恢复；vLLM/PACI 连接器集成仍是后续工作。压缩相对原始 BF16 有量化误差；LZ4 本身无损。

## 早期 DEFLATE 实现

沿用 CacheGen 每 token 最大绝对值缩放规则，选择 8-bit 符号。符号再经 DEFLATE 无损编码，设备端用 miniz 3.0.2 解码。原版 CacheGen 的 CUDA 算术编码器并未移植；本实现替换了熵编码格式。
一次 AICPU 调用完成熵解码和 ARM NEON BF16 恢复。每层按 token 范围分为 8 条独立码流，在不同设备流上提交，保留可独立认证的边界。这个改动消除了单个长熵码流的串行依赖。

## 早期首层证据

真实数据为先前导出的 70B 模型首层 K/V 投影，2048 tokens、8 KV heads、128 head dimension。只有首层；不是全模型 KV 或任务质量评测。所有恢复值与 CPU 量化参考的 BF16 位表示一致，不能称为与原始 BF16 无损一致。
在 910B4 / CANN 8.5.1 的空闲 NPU 0 测量，宿主环境未升级；使用已有容器镜像。各配置预热一次、计时三次，以下为中位数。计时包含 Python 提交与设备同步，不含量化编码、H2D、密码处理或模型计算。压缩率计入 FP32 scales、每条码流 16 字节对齐及80字节认证记录预算；尚未计入最终索引表或整个缓存容器。

| 数据（2048 tokens） | 单流压缩率 | 八流压缩率 | 单流恢复 ms | 八流恢复 ms |
|---|---:|---:|---:|---:|
| K | 2.417× | 2.416× | 21.684 | 5.524 |
| V | 3.163× | 3.157× | 23.870 | 6.089 |

无损 BF16 的 byte shuffle / token XOR + DEFLATE 在相同样本上约1.27–1.45×，未达到2×。4-bit路径虽有更高体积收益，但首层V的归一化均方误差约0.20；8-bit约0.00134。这是张量误差，不是任务得分下降。

## 早期证据的范围

以上仅为解码组件结果。后续实际 AES-GCM、全层 KV 和模型端到端测量见下文；组件加速不能直接作为端到端加速。

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

## 分片与准备成本验证

`characterize_shards.py` 将首层样本按 KV head 切成 TP=1/2/4/8 的分片，再分别编码，不跨卡收集 KV。八分片时每个 token 每卡有 128 个通道。计入 scales、tag、AAD、nonce 和索引预算后，K/V 合计压缩率如下：

| 每条码流的 token 数 | K 压缩率 | V 压缩率 | K/V 合计压缩率 |
|---|---:|---:|---:|
| 128 | 1.997× | 2.828× | 2.341× |
| 512 | 2.024× | 9.847× | 3.358× |

这些是首层样本的分片实验，不能当作八卡整模型压缩率。所有分片的 LZ4、位重排和符号恢复均经过独立检查。

`kernel/bitshuffle_neon.h` 用 NEON 实现位转置和字节转置，编码端和恢复算子共用。`test_bitshuffle_device.py` 交错比较旧、新 AICPU 算子，覆盖 128/256/1024 通道、128/512-token 码流、K/V 及四条设备流。12 组输出均与量化参考的 BF16 位表示相等。512-token 组的恢复时间下降约 2.4%–12.3%；小块中有接近持平的结果。主机端单独的位转置优化收益约 1%，不能用它解释整个准备阶段的降幅。

主机准备路径还删除了未使用的 FP32 转换，复用量化工作区，按层初始化状态数组，并避免复制整个 LZ4 输出缓冲区。`model_bench.py` 分别报告编码、AES-GCM、记录分配和最后同步的时间。`test_encoder_precision.py` 用全部有限 BF16 位模式及 INT8 舍入边界附近的值检查倒数乘法优化；149248 个符号均与直接除法参考一致。

仅修改主机准备路径后的同进程第二组对照，缓存准备加一次请求从 5381.92 ms 降到 5182.12 ms。更新设备算子后的另一组则为 5052.40 ms 对 5066.25 ms，近乎持平。因此目前稳定的收益来自已准备缓存的请求，单次“缓存准备＋重载”的净收益尚不能视为稳定结论。新设备算子下，该请求 E2E 为 BF16 2201.76 ms、压缩 1785.35 ms，压缩率 2.034×，40 层认证检查通过，16 个输出 token 一致。

最新代码的完整复核保存在 `model_workspace_p8192_s512.json`。配置不变，同进程交错测量各两次，第二组如下：

| 配置 | TTFT ms | TPOT ms | 请求 E2E ms | 缓存准备 ms |
|---|---:|---:|---:|---:|
| BF16 安全逐层流水 | 1031.77 | 79.80 | 2228.69 | 3344.08 |
| Bitshuffle+LZ4 安全逐层流水 | 643.52 | 80.35 | 1848.84 | 3721.62 |

压缩率仍为 2.034×，请求 E2E 降低 17.04%，TTFT 降低 37.63%；40 层消费前认证检查通过，16 个输出 token 一致。缓存准备加一次请求分别为 5572.77 ms 和 5570.46 ms，接近持平。TPOT 没有显示改善。
