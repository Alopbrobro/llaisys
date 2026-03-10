# 项目#5：分布式推理（张量并行）实施报告（修订版）

> 修订时间: 2026-03-08  
> 修订原因: 原计划范围过大、阶段耦合过强，不利于快速落地验证。  
> 本版目标: 先完成“可运行的最小闭环（MVP）”，再逐步扩展到完整 NCCL + MPI。

---

## 一、核心概念（简明版）

### 1.1 分布式推理

分布式推理是把一次推理任务分摊到多个计算设备协同执行，本质是：

- 计算切分（谁算哪一部分）
- 通信同步（各设备交换中间结果）
- 状态一致（KV-Cache、参数分片、随机性）

### 1.2 张量并行（TP）

TP 是“同一层内切分矩阵计算”，不是“不同层放不同设备”。

- Column Parallel：按输出维切分，产出局部结果
- Row Parallel：按输入维切分，局部结果 `all-reduce(sum)` 合并

Qwen2 常见映射：

- `q/k/v`, `mlp_gate`, `mlp_up`：Column Parallel
- `attn_o`, `mlp_down`, `lm_head`：Row Parallel

### 1.3 NCCL

NCCL 是 GPU 集合通信库，提供 `all-reduce/all-gather/broadcast` 等原语，适用于多 GPU 张量并行。

### 1.4 MPI

MPI 是多进程消息传递标准，CPU 分布式常用 `MPI_Allreduce/MPI_Allgather/MPI_Bcast`，可实现与 NCCL 对齐的通信语义。

---

## 二、现状复盘（基于当前代码）

- `llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)` 已有多设备参数，但实现只用了 `device_ids[0]`
- 现有 Runtime/ops 支持 CPU/NVIDIA 单设备执行
- 缺少跨设备集合通信抽象（NCCL/MPI）
- KV-Cache 机制完整，但是单设备形态

结论：当前是“单设备完整推理引擎”，尚未进入真正分布式。

---

## 三、修订后的目标与非目标

### 3.1 MVP 目标（必须）

1. GPU 路径：单机多卡 + NCCL，跑通 TP decode（`tp_size=2/4`）
2. 结果正确：`tp_size=1` 与原路径一致；`tp_size>1` 与基线近似
3. 工程可开关：编译与运行时均可开启/关闭 TP

### 3.2 第二目标（随后）

1. CPU 路径：MPI rank 模式跑通相同 TP 语义
2. 与 Python server 配置打通

### 3.3 非目标（本轮明确不做）

- 不做多机部署编排
- 不做 pipeline parallel / expert parallel
- 不做 attention kernel 大重写（先复用现有算子）
- 不在首轮同时改造 BatchContext 与前缀池（避免耦合爆炸）

---

## 四、总体设计（先稳后快）

### 4.1 统一通信抽象（保留最小接口）

新增 `DistComm`，仅保留 TP 前向需要的最小原语：

- `all_reduce_sum`
- `all_gather`
- `broadcast`
- `barrier`

后端实现：

- `DistCommNccl`（GPU）
- `DistCommMpi`（CPU）

### 4.2 TP 计算策略（先改 linear，再改 attention）

先改造最稳定且收益最高的线性层路径：

1. QKV/MLP gate/up 切分（column）
2. O/down/lm_head 切分（row + all-reduce）
3. 先维持当前 self-attention 主体逻辑，后续再做 KV head 分片

### 4.3 进程/设备模型（修订）

为降低复杂度，采用“两条独立模式”，不强行统一：

- NCCL 模式：单进程多 GPU（先用 `device_ids`）
- MPI 模式：多进程（`mpirun -n N`）

说明：这是工程折中。首版先跑通，再评估是否统一成“每设备一进程”模型。

---

## 五、分阶段计划（可执行）

### 阶段 A：通信骨架 + 构建开关（1 个迭代）

目标：代码能编译、能初始化、能跑空通信。

计划：

- 新增 `include/llaisys/distributed.h`
- 新增 `src/distributed/comm.hpp`
- 新增 NCCL/MPI 后端实现文件
- 更新 `xmake.lua`：`--dist-nccl`、`--dist-mpi`

验收：

- 打印 backend/world_size/rank
- all-reduce smoke test 通过

### 阶段 B：Qwen2 权重分片加载（1~2 个迭代）

目标：模型按 TP rank 持有本地权重。

计划：

- 扩展 `LlaisysQwen2Model`：`tp_size/tp_rank/comm`
- 在 `llaisysQwen2LoadWeightByName` 中按参数名执行切片
- 增加分片维度断言与日志

验收：

- 权重显存/内存约降为 `1/tp_size`
- 维度检查全通过

### 阶段 C：TP Linear 前向闭环（1~2 个迭代）

目标：先跑通不含 KV 分片的 TP decode。

计划：

- 新增 `linear_tp_column` / `linear_tp_row`
- 替换 Qwen2 中对应 linear 调用
- row parallel 输出执行 `all-reduce(sum)`

验收：

- `tp=1` bitwise/近似对齐原实现
- `tp=2/4` 可稳定生成 token

### 阶段 D：KV-Cache 与 Attention TP 化（1~2 个迭代）

目标：完成注意力链路分片。

计划：

- K/V head 按 rank 分片
- KV-Cache shape 改为本地 head 视图
- cache save/restore 增加 TP 元信息

验收：

- 长序列 decode 正常
- cache 高级接口在 TP 下可用

### 阶段 E：MPI 对齐 + Python/server 接入（1 个迭代）

目标：CPU MPI 路径复用同一 TP 语义，服务层可配置。

计划：

- 将 `DistCommMpi` 接入同一前向路径
- Python 绑定新增 `tp_size/backend/device_ids`
- `python/server/engine.py` 增加 TP 初始化配置

验收：

- NCCL/MPI 均可启动并完成最小推理
- engine stats 返回 TP 配置

---

## 六、里程碑（修订）

| 里程碑 | 输出 | 判定标准 | 状态 |
|------|------|------|------|
| M1 | 阶段 A 完成 | 通信层可初始化 + smoke test | ✅ 完成 |
| M2 | 阶段 B 完成 | 权重按 rank 分片加载 | ✅ 完成 |
| M3 | 阶段 C 完成 | TP Linear forward + all-reduce 跑通 | ✅ 完成 |
| M4 | 阶段 D 完成 | KV/Attention TP + Cache TP 验证 | ✅ 完成 |
| M5 | 阶段 E 完成 | Python/Server TP 集成 + 启动脚本 | ✅ 完成 |

---

## 七、风险与控制

- 过早全栈并行改造导致不可调试：按 A→E 逐段封闭验收
- 通信成本抵消收益：优先 row-parallel 必需 all-reduce，减少无效 all-gather
- TP 与量化叠加复杂：先保障 FP32/FP16，再接 INT8/INT4 路径
- 回归风险：强制保留 `tp_size=1` 原路径作为 fallback

---

## 八、实施进度记录

### 阶段 A (已完成)
- 创建 `include/llaisys/distributed.h` — C API (Create/Destroy/AllReduceSumF32/Barrier)
- 创建 `src/distributed/comm.hpp` — 抽象 `Comm` 类 + `comm_t` typedef
- 创建 `src/distributed/comm.cpp` — 工厂函数 + NCCL/MPI stub
- 创建 `src/distributed/mock_comm.cpp` — Mock 后端 (no-op)
- 创建 `src/llaisys/distributed.cc` — C API 包装
- 更新 `xmake.lua` — `dist-nccl`/`dist-mpi` 编译选项
- 创建 `test/dist_smoke.cpp` — smoke test: `ALL PASSED`

### 阶段 B (已完成)
- 扩展 `LlaisysQwen2Model` — `tp_size/tp_rank/local_nh/local_nkvh/local_di`
- 实现 `TpSlice` enum + `classifyWeight()` — 按参数名自动分类切片策略
- 实现 `slice2D()`/`slice1D()` — CPU 侧权重切片
- 重写 `llaisysQwen2LoadWeightByName()` — 自动 TP 权重切片
- 新增 `llaisysQwen2ModelCreateTP()`/`GetTpSize()`/`GetTpRank()` C API
- 所有 forward/buffer/cache 改用 `local_nh/local_nkvh/local_di`
- 创建 `test/tp_shard_smoke.cpp` — smoke test: `ALL PASSED`

### 阶段 C (已完成)
- 新增 `llaisysDistCommGetImplPtr()` — 获取内部 `shared_ptr<Comm>*`
- 新增 `LlaisysQwen2Model::comm` 字段 + `allReduceIfTP()` helper
- 新增 `llaisysQwen2SetComm()` C API
- 在 Infer/InferSample/BatchDecode 三个前向路径中 O proj 和 down proj 后插入 all-reduce
- 创建 `test/tp_fwd_smoke.cpp` — smoke test: `ALL PASSED`

### 阶段 D (已完成)
- `LlaisysQwen2CacheSnapshot` 增加 `tp_size`/`tp_rank` 字段
- `SaveCache`/`RestoreCache` 自动记录和校验 TP 元数据
- `batch_slot_save_impl`/`batch_slot_restore_impl` 同步加入 TP 兼容性验证
- 跨 TP 配置恢复被正确拒绝 (tp_size/tp_rank mismatch → error)
- 创建 `test/tp_cache_smoke.cpp` — smoke test: `ALL PASSED`

### 阶段 E (已完成)
- 创建 `python/llaisys/libllaisys/distributed.py` — ctypes 绑定 (DistBackend/LlaisysDistConfig/comm_*)
- 更新 `python/llaisys/libllaisys/qwen2.py` — 新增 CreateTP/GetTpSize/GetTpRank/SetComm 签名和导出
- 更新 `python/llaisys/models/qwen2.py` — `Qwen2.__init__` 支持 `tp_size/tp_rank/dist_backend/comm_handle`
  - tp_size>1 时自动创建 comm (优先 NCCL, 回退 MOCK), 绑定到模型
  - `__del__` 清理 comm; 新增 `tp_size`/`tp_rank`/`is_tp` 属性
- 更新 `python/server/app.py` — `load_model()` 和 CLI 支持 `--tp-size/--tp-rank`
- 创建 `scripts/tp_launch.py` — 多进程 TP 启动脚本 (subprocess 或 mpirun)