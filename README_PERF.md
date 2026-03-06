# LLAISYS GPU 推理性能优化报告

> **模型**: DeepSeek-R1-Distill-Qwen-1.5B (28 层, 12 Query Heads, 2 KV Heads, hidden_size=1536)
>
> **硬件**: NVIDIA RTX 4060 Laptop GPU (Ada Lovelace, SM 8.9, 24 SMs, 8GB VRAM)
>
> **软件**: CUDA 12.5, cuBLAS, xmake, WSL2 Ubuntu
>
> **优化周期**: 2026 年 2 月

---

## 一、总览

LLAISYS 是一个从零实现的轻量级 LLM 推理引擎。在完成全部 8 个 CUDA 算子（add、embedding、rms_norm、rope、swiglu、self_attention、argmax、linear）的 FP32/FP16/BF16 多精度支持后，GPU 推理的**功能正确性**已验证通过（与 HuggingFace Transformers 输出逐 token 完全一致），但推理性能极差——生成 88 个 token 耗时 **118 秒**，而 HuggingFace 仅需约 3 秒。

本文记录了从 118s 到 2.73s 的完整优化过程，最终实现 **43 倍加速**，性能与 HuggingFace 持平甚至略优。

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  优化前       │     │  第一轮优化   │     │  第二轮优化   │     │  最终结果     │
│  118.0s      │ ──▸ │  59.0s       │ ──▸ │  57.4s       │ ──▸ │  2.73s       │
│  0.75 tok/s  │     │  1.49 tok/s  │     │  1.53 tok/s  │     │  27.0 tok/s  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ×2 加速              ×1.03 加速          ×21 加速
```

### 1.1 2026-03 量化迭代补充（INT8 / INT4 / GPTQ/AWQ）

在上述 GPU 基础优化完成后，新增了 Weight-Only 量化链路，并完成了端到端验证：

- **INT8 W8A16（per-channel symmetric）**：约 **2x** 压缩
- **INT4 W4A16（per-group symmetric, g=128）**：约 **3.76x** 压缩
- **GPTQ/AWQ 兼容**：加载时在 Python 侧转换到内部 INT4 打包格式（无需新增 C++ 推理 API）
- **INT4 CUDA kernel 优化**：从「1 输出/线程」改为「1 packed byte/线程（2 输出）」，减少重复读取与线程调度开销

快速复现实验命令：

```bash
# 1) INT8 量化
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model \
  --bits 8

# 2) INT4 量化
python3 scripts/quantize.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./quantized_model_int4 \
  --bits 4 --group-size 128

# 3) 启动 INT4 服务
cd python && python3 -m server.app \
  --model /home/bbq/llaisys/quantized_model_int4 \
  --device nvidia --port 8000

# 4) GPTQ/AWQ 模型直接加载（若 config.json 含 quantization_config）
cd python && python3 -m server.app \
  --model <hf_repo_or_local_snapshot> \
  --device nvidia --port 8000
```

说明：

- `python/server/app.py` 已改为从 **resolved model path** 加载 tokenizer，支持本地快照与离线场景
- GPTQ 格式使用 `qweight/qzeros/scales` 转换时包含 **zero-point +1 修正**（兼容 AutoGPTQ 常见存储约定）

---

## 二、性能分析方法论

### 2.1 Nsight Systems (nsys) 系统级分析

使用 `nsys profile --stats=true --trace=cuda` 对完整推理过程进行端到端采集，重点关注 CUDA Runtime API 层面的耗时分布。nsys 的优势在于以极低开销采集全局调用统计，快速定位 **Host-side 瓶颈**（如过多的同步调用、显存分配等）。

```bash
/opt/nvidia/nsight-systems/2024.2.3/bin/nsys profile \
    --stats=true -o /tmp/llaisys_profile \
    python3 test/test_infer.py --device nvidia
```

通过 `nsys stats --report cuda_api_sum` 提取 CUDA API 调用汇总表，按 `Total Time` 降序排列，即可精准定位热点。

---

## 三、初始性能基线 (118s)

### 3.1 nsys 分析结果

| CUDA API 调用 | 耗时 | 占比 | 调用次数 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 71.9s | 59.0% | 81,000 |
| `cudaFree` | 27.0s | 22.2% | 94,011 |
| `cudaMalloc` | 13.4s | 11.0% | 73,812 |
| `cudaLaunchKernel` | 5.4s | 4.4% | 370,823 |
| 其他 | 4.1s | 3.4% | — |
| **合计 Host 开销** | **112.3s / 118s = 95%** | | |

**关键发现**: 95% 的时间消耗在 Host 端的 CUDA Runtime 调用上，GPU 计算本身几乎不构成瓶颈。推理引擎的性能问题完全来自 **CPU-GPU 交互开销**。

### 3.2 根因溯源

通过代码审查，定位到三个核心病因：

1. **`cublasCreate` / `cublasDestroy` 逐次调用** — `linear` 算子每次执行时创建 cuBLAS handle 并销毁，而 `cublasCreate` 内部隐式调用 `cudaDeviceSynchronize`。该操作被执行约 240 次/token × 88 token ≈ **21,120 次**。

2. **`cudaMalloc` / `cudaFree` 逐次调用** — `self_attention` 算子每次执行时分配 5 个临时缓冲区，执行完毕后立即释放。28 次/token × 88 token × 5 缓冲区 ≈ **12,320 次 malloc + 12,320 次 free**。

3. **逐 head 循环** — `self_attention` 采用 `for (head = 0; head < n_head; head++)` 逐头计算，每个 head 启动约 8 个 CUDA kernel，12 heads × 8 = 96 次 kernel launch/call × 28 层 × 88 token ≈ **370K 次 kernel launch**。

---

## 四、六项核心优化

### 优化 1：cuBLAS Handle 全局复用

**问题**: `linear` 和 `self_attention` 在每次 op 调用时执行 `cublasCreate()` + `cublasDestroy()`，每次 create 内部触发 `cudaDeviceSynchronize`，造成 CPU-GPU 全同步。

**解决方案**: 使用 `static thread_local cublasHandle_t` 实现 handle 的惰性初始化与全局复用。整个推理生命周期内只创建一次 handle，永不销毁。

**修改文件与位置**:
- `src/ops/linear/nvidia/linear_nvidia.cu` — `get_cublas_handle()` 函数 (L56-61)
- `src/ops/self_attention/nvidia/self_attention_nvidia.cu` — `get_cublas_handle()` 函数 (L190-195)

```cpp
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}
```

**设计思想**: 在 AI 推理引擎中，cuBLAS handle 是线程级资源，其生命周期应等同于推理服务进程本身。`static thread_local` 既保证线程安全，又避免全局锁。这是所有生产级推理框架（TensorRT、vLLM、llama.cpp）的标准做法。

---

### 优化 2：显存缓冲区 Grow-Only 缓存

**问题**: `self_attention` 每次调用分配 5 个临时 GPU 缓冲区（Q、K、V 各一个 gather 缓冲区 + scores 矩阵 + output 缓冲区），计算完毕后立即 `cudaFree`。`cudaMalloc`/`cudaFree` 本身是重量级同步操作。

**解决方案**: 采用 **Grow-Only 缓存策略** — 使用 `static` 指针持有缓冲区，仅在所需大小超过当前容量时才重新分配（旧缓冲区先释放再分配更大的），否则直接复用。

**修改文件与位置**:
- `src/ops/self_attention/nvidia/self_attention_nvidia.cu` — 静态缓冲区声明 (L200-211) 与 `ensure_buf()` 函数 (L212-216)

```cpp
static float *s_q_all = nullptr;   static size_t s_q_all_sz = 0;
static float *s_k_all = nullptr;   static size_t s_k_all_sz = 0;
static float *s_v_all = nullptr;   static size_t s_v_all_sz = 0;
static float *s_scores = nullptr;  static size_t s_scores_sz = 0;
static float *s_o_all = nullptr;   static size_t s_o_all_sz = 0;

static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;          // 容量足够，零开销复用
    if (ptr) cudaFree(ptr);           // 不够才释放旧的
    CUDA_CHECK(cudaMalloc(&ptr, need)); // 分配更大的
    cur = need;
}
```

**设计思想**: 在自回归解码中，KV cache 长度单调递增，缓冲区大小只会增长。Grow-Only 策略完美匹配此访问模式，在稳态下实现 **零分配推理（zero-allocation inference）**。首个 token 触发少量分配后，后续所有 token 均复用已有缓冲区。

**效果**: `cudaMalloc` 从 73,812 次降至 740 次，`cudaFree` 从 94,011 次降至 680 次。

---

### 优化 3：Batched Self-Attention（消除逐 Head 循环）

**问题**: 原实现用 `for (h = 0; h < n_head; h++)` 逐 head 串行计算注意力。每个 head 依次执行 gather → GEMM(Q×K^T) → scale → mask → softmax → GEMM(P×V) → scatter，共 ~8 次 kernel launch。对于 12 个 query heads，每次 self_attention 调用产生 ~96 次 kernel launch。

**解决方案**: 将所有 head 的数据 gather 到 `[n_head, seq_len, dim]` 的 batched 布局中，使用 `cublasSgemmStridedBatched` 一次性完成所有 head 的矩阵乘法。scale、causal mask、softmax 也一次性对所有 head 并行执行。

**修改文件与位置**:
- `src/ops/self_attention/nvidia/self_attention_nvidia.cu` — 完全重写
  - Batched gather kernels: `gather_all_q_kernel` (L51-63), `gather_expand_kv_kernel` (L72-90)
  - Batched scale + mask: `scale_kernel` (L108-111), `causal_mask_batched_kernel` (L116-130)
  - Batched softmax: `softmax_row_kernel` (L134-164)
  - Batched scatter: `scatter_all_heads_kernel` (L97-106)
  - 核心调度: `self_attention_impl<T>()` (L226-363)
  - GEMM 调用: `cublasSgemmStridedBatched` (L298, L338)

**计算流程（7 步）**:

```
Step 1: gather_all_q        T[seq,nh,dh] → float[nh,seq,dh]     1 kernel
Step 2: gather_expand_kv    T[tot,nkvh,dh] → float[nh,tot,dh]   2 kernels (K,V)
Step 3: batched GEMM        scores = Q × K^T                     1 cublasSgemmStridedBatched
Step 4: scale + causal mask  全 head 并行                         2 kernels
Step 5: softmax              nh×seq 行并行                        1 kernel
Step 6: batched GEMM        output = softmax(scores) × V         1 cublasSgemmStridedBatched
Step 7: scatter_all_heads   float[nh,seq,dh] → T[seq,nh,dh]     1 kernel
                                                         合计: 8 kernels (vs 原来 96)
```

**GQA 支持**: `gather_expand_kv_kernel` 在 gather 阶段完成 GQA head 扩展（`kv_h = h / group_size`），使 2 个 KV head 扩展为 12 个 query head 对应的副本，后续 batched GEMM 无需特殊处理。

**效果**: kernel launch 次数从 ~370,823 降至 ~164,015（减少 56%）。

---

### 优化 4：Residual 指针交换（零拷贝残差连接）

**问题**: Transformer 每层有两次残差连接（attention 后和 MLP 后），原实现在每次残差连接前调用 `memcpyOnDevice(residual, hidden_states, ...)` 进行 Device-to-Device 拷贝以保存残差。28 层 × 2 次 × 88 token = **4,928 次 D2D memcpy**，每次拷贝 1536×4 = 6,144 字节。

**解决方案**: 使用 `std::swap(model->residual, model->hidden_states)` 交换两个 `shared_ptr<Tensor>` 的指针。交换后 `residual` 指向原来的 `hidden_states` 数据，`hidden_states` 指向空闲的 `residual` 缓冲区供后续计算写入。这是一个 **O(1) 的指针交换操作**，不涉及任何 GPU 内存移动。

**修改文件与位置**:
- `src/llaisys/models/qwen2.cpp`
  - Attention 残差: L196 `std::swap(model->residual, model->hidden_states)`
  - MLP 残差: L238 `std::swap(model->residual, model->hidden_states)`
  - RMS Norm 输入改为从 `model->residual` 读取 (L199, L239)

```cpp
// 优化前：D2D memcpy（6KB/次，4928 次）
model->memcpyOnDevice(model->residual->data(),
                       model->hidden_states->data(), hidden_size_bytes);
ops::rms_norm(model->norm_out, model->hidden_states, ...);

// 优化后：零拷贝指针交换
std::swap(model->residual, model->hidden_states);
ops::rms_norm(model->norm_out, model->residual, ...);  // 从 residual 读
```

**设计思想**: 这是经典的 **double-buffering / ping-pong buffer** 技术。两个 tensor 交替充当"输入"和"输出"角色，通过指针交换实现零开销的角色切换。此模式在 NCCL 通信重叠、流水线并行等场景中广泛应用。

---

### 优化 5：异步内存传输（消除 CPU-GPU 同步屏障）

**问题**: 推理循环中的 H2D（token_id、position_id 写入 GPU）和 D2D（KV cache 更新）均使用同步 `cudaMemcpy`，每次调用都会阻塞 CPU 直到传输完成并且此前所有 GPU 工作结束。每个 token 有 2 次 H2D + 2 次 D2D = **4 次同步屏障**。

**解决方案**: 将 `memcpy_sync` 替换为 `memcpy_async`（使用 default stream `nullptr`），使 H2D 和 D2D 传输与后续 kernel 在同一 stream 上顺序执行而不阻塞 CPU。仅在 token 结束时的 D2H（读取 argmax 结果）保留同步 `memcpy_sync`，作为唯一的同步点。

**修改文件与位置**:
- `src/llaisys/models/qwen2.cpp`
  - `memcpyH2D()`: L59 `memcpy_sync` → `memcpy_async(..., nullptr)`
  - `memcpyOnDevice()`: L79 `memcpy_sync` → `memcpy_async(..., nullptr)`
  - `memcpyD2H()`: L69 保持 `memcpy_sync`（必须同步以获取结果）

**设计思想**: GPU 编程的核心原则——**最小化同步点（minimize synchronization points）**。在自回归解码循环中，理想状态是每个 token 仅有一个同步点：读取 argmax 结果的 D2H memcpy。所有其他操作（H2D、D2D、kernel launch）都应在 GPU stream 上异步流水执行。

---

### 优化 6：GPU 显存池释放（消除 VRAM 争用）

**问题**: 测试脚本 `test_infer.py` 先运行 HuggingFace 推理（占用 ~3.5GB GPU 显存），然后 `del model; gc.collect()` 删除模型对象。但 **PyTorch 的 CUDA 内存池不会自动归还显存给操作系统**——它保留已分配的 VRAM 用于后续 PyTorch 操作复用。随后 LLAISYS 加载权重需要 ~7.2GB，总计超过 8GB VRAM 限制，导致 GPU 被迫反复在显存和系统内存之间换页（在 WSL2 环境下尤为严重），推理时间从 2.73s 膨胀到 57s。

**解决方案**: 在 `del model; gc.collect()` 之后添加 `torch.cuda.empty_cache()`，强制 PyTorch 将缓存的 VRAM 归还给 CUDA driver。

**修改文件与位置**:
- `test/test_infer.py` — L116 `torch.cuda.empty_cache()`

```python
del model
gc.collect()
torch.cuda.empty_cache()  # 强制释放 PyTorch CUDA 内存池
```

**效果**: LLAISYS 推理时间从 57s 降至 2.73s（消除显存争用后的真实性能）。

---

## 五、nsys 性能分析对比

### 5.1 优化前（基线 118s）

| CUDA API | 耗时 (s) | 调用次数 | 根因 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 71.9 | 81,000 | cuBLAS create/destroy 隐式触发 |
| `cudaFree` | 27.0 | 94,011 | self_attention 逐次释放临时缓冲区 |
| `cudaMalloc` | 13.4 | 73,812 | self_attention 逐次分配临时缓冲区 |
| `cudaLaunchKernel` | 5.4 | 370,823 | 逐 head 循环，每 head ~8 个 kernel |

### 5.2 第一轮优化后（优化 1+2，59s）

| CUDA API | 耗时 (s) | 调用次数 | 变化 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 0 | 0 | ✅ **完全消除**（handle 复用） |
| `cudaFree` | 0.39 | 680 | ↓ 99.3%（grow-only 缓存） |
| `cudaMalloc` | 0.27 | 740 | ↓ 99.0%（grow-only 缓存） |
| `cudaMemcpy` | 31.0 | 10,689 | 新瓶颈：同步 memcpy |
| `cudaLaunchKernel` | 31.0 | 370,823 | 未优化（逐 head 循环仍在） |

### 5.3 第二轮优化后（优化 3+4+5+6，2.73s）

| CUDA API | 耗时 (s) | 调用次数 | 变化 |
|---|---|---|---|
| `cudaMemcpy` (sync) | — | 仅 D2H | ↓ 每 token 仅 1 次同步点 |
| `cudaLaunchKernel` | — | 164,015 | ↓ 56%（batched attention） |
| `cudaMalloc` | 0.29 | 740 | 持平（已优化到位） |
| `cudaFree` | 0.39 | 680 | 持平（已优化到位） |

---

## 六、最终性能对比

### 6.1 端到端推理（88 token 生成）

| 指标 | 优化前 | 优化 1+2 | 优化 3+4 | 优化 5+6（最终） | HuggingFace |
|---|---|---|---|---|---|
| 总时间 | 118.0s | 59.0s | 57.4s | **2.73s** | 2.89s |
| 吞吐量 | 0.75 tok/s | 1.49 tok/s | 1.53 tok/s | **27.0 tok/s** | 26.2 tok/s |
| 单 token 延迟 | ~1341ms | ~670ms | ~652ms | **~30ms** | ~33ms |
| 相对加速比 | 1× | 2.0× | 2.1× | **43.2×** | — |

### 6.2 加速归因分析

| 优化项 | 加速贡献 | 核心指标变化 |
|---|---|---|
| cuBLAS Handle 复用 | ~1.7× | `cudaDeviceSynchronize` 81K→0 次 |
| Grow-Only 缓冲区缓存 | ~1.2× | `cudaMalloc`+`cudaFree` 168K→1.4K 次 |
| Batched Self-Attention | ~1.03× | `cudaLaunchKernel` 370K→164K 次 |
| Residual 指针交换 | ~1.01× | D2D memcpy 消除 4,928 次 |
| 异步 memcpy | 与优化 6 合并 | 消除 H2D/D2D 同步屏障 |
| GPU 显存池释放 | **~21×** | 消除 VRAM 争用导致的换页 |

> **注**: 优化 5+6 的巨大加速来自于消除 WSL2 环境下 GPU 显存不足时的换页开销。在显存充足的环境中，优化 1-4 已将真实推理时间降至 ~2.7s。

---

## 七、设计哲学总结

本次优化遵循两个核心 AI Infra 设计原则：

### 7.1 减少同步（Minimize Synchronization）

GPU 推理引擎的性能杀手不是 GPU 计算本身，而是 **CPU-GPU 同步**。每一次 `cudaDeviceSynchronize`、`cudaMemcpy`（同步版本）、`cublasCreate` 都是一个 **流水线气泡（pipeline bubble）**——CPU 停转等待 GPU 完成所有排队工作。

优化策略：
- **资源生命周期提升**: cuBLAS handle 从"逐 op"提升为"进程级"（优化 1）
- **分配策略优化**: 从"每次 malloc/free"改为"grow-only 缓存"（优化 2）
- **传输异步化**: H2D 和 D2D 改为 `memcpy_async`，每 token 仅保留 1 个同步点用于读取 argmax 结果（优化 5）

最终效果：每 token 的 CPU-GPU 同步从 **~920 次** 降至 **1 次**。

### 7.2 零拷贝（Zero-Copy）

在内存带宽受限的推理场景中，每一次不必要的数据搬移都是性能损失。

优化策略：
- **指针交换替代 memcpy**: 残差连接通过 `std::swap` 实现 O(1) 零拷贝（优化 4）
- **Batched 计算替代逐 head 循环**: 一次 gather 所有 head 到连续 buffer，一次 `cublasSgemmStridedBatched` 完成所有 head 的矩阵乘法，避免反复的 gather/scatter（优化 3）
- **Grow-Only 缓冲区复用**: 稳态下完全无 GPU 内存分配开销（优化 2）

---

## 八、修改文件清单

| 文件路径 | 修改内容 | 涉及优化 |
|---|---|---|
| `src/ops/linear/nvidia/linear_nvidia.cu` | `get_cublas_handle()` 全局 handle (L56-61) | 优化 1 |
| `src/ops/self_attention/nvidia/self_attention_nvidia.cu` | 完全重写：全局 handle (L190-195)、grow-only 缓存 (L200-216)、batched attention (L226-363) | 优化 1, 2, 3 |
| `src/llaisys/models/qwen2.cpp` | 指针交换 (L196, L238)、异步 memcpy (L59, L79) | 优化 4, 5 |
| `test/test_infer.py` | `torch.cuda.empty_cache()` (L116) | 优化 6 |

---

## 九、验证结果

- ✅ 8 个 CUDA 算子 × 3 种精度 (F32/F16/BF16) 全部单元测试通过
- ✅ 端到端推理输出与 HuggingFace Transformers **逐 token 完全一致**
- ✅ 推理速度 2.73s vs HuggingFace 2.89s，**略优于 HuggingFace**
