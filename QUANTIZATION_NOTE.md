# LLAISYS 量化实现笔记（INT8 / INT4 / GPTQ/AWQ）

> 日期：2026-03  
> 分支：`feat/quantization`

---

## 1. 目标与范围

本次量化迭代目标：

1. 支持 **INT8 W8A16**（权重 INT8、激活 FP16/FP32）
2. 支持 **INT4 W4A16**（权重 INT4 打包、激活 FP16/FP32）
3. 在不改动对外推理 API 的前提下，兼容 **GPTQ/AWQ** 权重加载
4. 保持可回退：原 FP32 路径继续可用

---

## 2. 最终效果（结论）

- INT8（per-channel symmetric）：约 **2x** 压缩
- INT4（per-group symmetric, g=128）：约 **3.76x** 压缩
- GPTQ/AWQ：通过 Python 侧加载时转换，复用同一 INT4 推理链路
- INT4 kernel：完成一轮低风险优化（每线程处理 1 packed-byte，写 2 个输出）

---

## 3. 实现架构

### 3.1 INT8 路径（已稳定）

- 量化公式（按行）：
  - `scale = max(abs(W_row)) / 127`
  - `W_int8 = round(W / scale).clamp(-128, 127)`
- 推理时：
  - `W_fp32 = W_int8 * scale`（dequantize）
  - 再走已有 `linear`

关键点：

- C++ 侧在 `linear_maybe_dequant` 中按 `dtype` 自动分流
- `I8` 走 INT8 dequantize

### 3.2 INT4 路径（已稳定）

- 量化粒度：per-group（默认 `group_size=128`）
- 值域：`[-8, 7]`
- 打包：2 个 INT4 存入 1 个 `uint8`
  - `byte = ((val1 + 8) << 4) | ((val0 + 8) & 0x0F)`
- 推理时：
  - 从 `U8` 解包成 2 个 INT4
  - 按 group 查 scale，反量化到 FP32
  - 再走已有 `linear`

关键点：

- `U8` 作为 INT4 packed 存储类型
- group_size 由 `scale` 形状自动推导，避免新增对外接口

### 3.3 GPTQ/AWQ 兼容路径（加载时转换）

加载阶段识别到 GPTQ/AWQ 后，将其 `qweight/qzeros/scales` 转为内部 INT4 格式：

1. 解包 `int32` 中的 4-bit 权重
2. 反量化回 FP32
3. 重新量化为内部 symmetric INT4
4. 打包为 `uint8` + 生成内部 `scale`

注意：

- 增加了 **zero-point +1 修正**（兼容常见 AutoGPTQ 存储约定）
- 这样 C++ 推理侧无需新增 GPTQ/AWQ 专用 op

---

## 4. 关键代码位置

- 量化脚本：`scripts/quantize.py`
- 模型加载与 GPTQ/AWQ 转换：`python/llaisys/models/qwen2.py`
- 推理服务 tokenizer 路径修复：`python/server/app.py`
- 量化推理路由：`src/llaisys/models/qwen2.cpp`
- INT8/INT4 dequantize 入口：`src/ops/dequantize/op.cpp`
- INT4 CUDA kernel：`src/ops/dequantize/nvidia/dequantize_nvidia.cu`

---

## 5. 踩坑与修复

### 5.1 dequant cache 键冲突导致非法访存

问题：早期把 dequant 缓冲区按 `numel` 做 key，不同 shape 但同元素数的矩阵会复用同一 buffer，导致 CUDA illegal memory access。  
修复：改为按 `(rows, cols)` 作为 key。

### 5.2 GPTQ 转换后输出乱码

问题：未做 zero-point +1 修正，反量化偏移导致语义明显劣化。  
修复：`z = z + 1` 后再执行反量化，输出恢复可读。

### 5.3 tokenizer 在线依赖导致离线不稳

问题：服务端曾硬编码从远端 repo 拉 tokenizer，网络波动会阻塞启动。  
修复：改为从 `resolved model path` 加载 tokenizer，支持本地快照与离线运行。

---

## 6. 复现命令

### 6.1 构建

```bash
cd /home/bbq/llaisys
xmake f --nv-gpu=y -y
xmake build -y
xmake install -y
```

### 6.2 INT8 量化

```bash
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model \
  --bits 8
```

### 6.3 INT4 量化

```bash
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model_int4 \
  --bits 4 --group-size 128
```

### 6.4 启动服务（INT4）

```bash
cd python
python3 -m server.app \
  --model /home/bbq/llaisys/quantized_model_int4 \
  --device nvidia --port 8000
```

### 6.5 启动服务（GPTQ/AWQ）

```bash
cd python
python3 -m server.app \
  --model <hf_repo_or_local_snapshot> \
  --device nvidia --port 8000
```

---

## 7. 当前已知边界

- GPTQ/AWQ 目前是“加载时转换”，首启有额外 CPU 时间
- 尚未做“转换结果落盘缓存”（可作为下一步优化）
- INT4 kernel 目前完成第一轮优化，后续可继续做向量化读取与 block 参数调优

---

## 8. AWQ 大模型适配记录（2026-03-06）

### 8.1 配置驱动改造

原代码中 `Qwen2.__init__` 将模型架构参数（层数、隐藏维度、注意力头数等）硬编码为 DeepSeek-R1-Distill-Qwen-1.5B 的值。已改为**从 `config.json` 自动读取**，保留硬编码值作为回退默认值。

改动文件：`python/llaisys/models/qwen2.py`

映射关系：

| HF config.json 字段 | Meta 字段 | 1.5B 默认值 |
|---|---|---|
| `num_hidden_layers` | `nlayer` | 28 |
| `hidden_size` | `hs` | 1536 |
| `num_attention_heads` | `nh` | 12 |
| `num_key_value_heads` | `nkvh` | 2 |
| `hidden_size / num_attention_heads` | `dh` | 128（自动计算） |
| `intermediate_size` | `di` | 8960 |
| `vocab_size` | `voc` | 151936 |
| `rms_norm_eps` | `epsilon` | 1e-6 |
| `rope_theta` | `theta` | 10000.0 |
| `eos_token_id` | `end_token` | 151643（支持 int 或 list 格式） |

`maxseq` 取值优先级：用户显式传入 `max_seq_len` > `sliding_window` > `max_position_embeddings`（上限截断到 32768）。

### 8.2 AWQ GEMM 格式（区别于 GPTQ）

实测 `Qwen/Qwen2-7B-Instruct-AWQ`（AutoAWQ 量化）时发现，AWQ GEMM 的 packing 轴与 GPTQ **不同**：

| | GPTQ | AWQ GEMM |
|---|---|---|
| `qweight` | `[in_features // 8, out_features]` 行打包 | `[in_features, out_features // 8]` **列打包** |
| `qzeros` | `[groups, out // 8]` | `[groups, out // 8]`（相同） |
| zero-point 修正 | 需要 +1（AutoGPTQ 约定） | **不需要**（AWQ 直接存储） |

已新增 `_convert_awq_layer()` 方法，并在加载时根据 `quant_method` 字段自动选择转换器。

### 8.3 Qwen2-7B-Instruct-AWQ 显存分析

| 组件 | 大小 (GB) | 说明 |
|---|---|---|
| Embedding | 2.18 | `[152064, 3584]` FP32 未量化 |
| LM Head | 2.18 | `[152064, 3584]` FP32 未量化 |
| INT4 权重（28 层） | 3.47 | packed U8 + scale F32 |
| KV-Cache（seq=512） | 0.06 | 2 × 28 × 512 × 4 × 128 × 4B |
| Dequant 缓存 | 0.27 | 最大层 `down_proj [3584, 18944]` |
| CUDA + buffers | 0.35 | |
| **总计** | **8.51** | **超出 8GB 显存约 0.5GB** |

**根因**：Embedding 和 lm_head **未被 AWQ 量化**，各占 2.18GB FP32。

### 8.4 待解决：OOM 优化方案

以下方案可减少显存使用：

1. **FP16 存储 Embedding/LM Head**：AWQ 模型的原始 dtype 为 FP16，当前代码强制转 FP32 造成浪费。改为 FP16 存储可节省 **2.18 GB**，但需要 C++ 侧支持 FP16 embedding lookup 和 lm_head matmul。
2. **量化 LM Head 为 INT8**：对 lm_head.weight 做 per-channel INT8 量化，节省 ~1.6GB。
3. **分层 dequantize**：避免缓存所有层的 dequant 结果，改为用完即释放。

---

## 9. 推荐下一步

1. **解决 7B OOM**：优先方案 — FP16 存储 embedding/lm_head（改动最小，收益最大）
2. 增加 GPTQ/AWQ 转换缓存文件（首次转换后落盘，后续直接加载）
3. 输出统一性能表：FP32 / INT8 / INT4 / GPTQ-AWQ（tokens/s、首 token 延迟、显存）
4. 继续优化 INT4 kernel（vectorized load、访存合并）
