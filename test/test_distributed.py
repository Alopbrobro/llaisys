#!/usr/bin/env python3
"""
分布式推理（张量并行）综合测试脚本

测试内容:
  1. Mock 通信后端初始化与 API
  2. NCCL 后端可用性与 stub 状态
  3. Python 层 Qwen2 TP 模型构建 (mock)
  4. TP 属性验证 (tp_size/tp_rank/is_tp)
  5. 通信配置正确性
  6. 多 world_size 配置

运行:
  cd /home/alopbrobro8848/llaisys/python
  python3 -m test.test_distributed
或
  cd /home/alopbrobro8848/llaisys
  PYTHONPATH=python python3 test/test_distributed.py
"""

import sys
import os
import ctypes
import traceback

# 确保能导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from llaisys.libllaisys.distributed import (
    DistBackend,
    LlaisysDistConfig,
    comm_create,
    comm_destroy,
    comm_world_size,
    comm_rank,
    comm_backend,
    backend_available,
    all_reduce_sum_f32,
    barrier,
)

passed = 0
failed = 0
errors = []

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        errors.append((name, str(e)))
        failed += 1


# ═══════════════════════════════════════════════════════════════
# 测试组 1: 后端可用性检查
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 1] 后端可用性")

def test_mock_available():
    assert backend_available(DistBackend.MOCK) == 1, "Mock 后端应始终可用"

def test_nccl_available():
    result = backend_available(DistBackend.NCCL)
    print(f"    NCCL available = {result}")
    # 只记录, 不强制要求

def test_mpi_available():
    result = backend_available(DistBackend.MPI)
    print(f"    MPI available = {result}")

test("Mock 后端可用", test_mock_available)
test("NCCL 后端检查", test_nccl_available)
test("MPI 后端检查", test_mpi_available)


# ═══════════════════════════════════════════════════════════════
# 测试组 2: Mock Comm 基本功能
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 2] Mock Comm 基本功能")

def test_mock_create_destroy():
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.MOCK
    cfg.world_size = 1
    cfg.rank = 0
    cfg.local_device = 0
    comm = comm_create(cfg)
    assert comm is not None and comm != 0, "comm 创建失败"
    comm_destroy(comm)

def test_mock_metadata():
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.MOCK
    cfg.world_size = 4
    cfg.rank = 2
    cfg.local_device = 2
    comm = comm_create(cfg)
    assert comm_world_size(comm) == 4, f"world_size should be 4, got {comm_world_size(comm)}"
    assert comm_rank(comm) == 2, f"rank should be 2, got {comm_rank(comm)}"
    comm_destroy(comm)

def test_mock_allreduce():
    """Mock allReduce 是 no-op (数据不变)"""
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.MOCK
    cfg.world_size = 2
    cfg.rank = 0
    cfg.local_device = 0
    comm = comm_create(cfg)
    
    data = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    all_reduce_sum_f32(comm, data, 4)
    
    # Mock 下 data 不应改变
    assert data[0] == 1.0 and data[3] == 4.0, "Mock allReduce 不应修改数据"
    comm_destroy(comm)

def test_mock_barrier():
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.MOCK
    cfg.world_size = 2
    cfg.rank = 0
    cfg.local_device = 0
    comm = comm_create(cfg)
    barrier(comm)  # 应不崩溃
    comm_destroy(comm)

def test_mock_multi_rank():
    """模拟多个 rank 的 comm 创建"""
    comms = []
    for rank in range(4):
        cfg = LlaisysDistConfig()
        cfg.backend = DistBackend.MOCK
        cfg.world_size = 4
        cfg.rank = rank
        cfg.local_device = rank
        comm = comm_create(cfg)
        assert comm_world_size(comm) == 4
        assert comm_rank(comm) == rank
        comms.append(comm)
    for c in comms:
        comm_destroy(c)

test("Mock Comm 创建/销毁", test_mock_create_destroy)
test("Mock Comm 元数据查询", test_mock_metadata)
test("Mock AllReduce (no-op)", test_mock_allreduce)
test("Mock Barrier", test_mock_barrier)
test("Mock 多 Rank 创建", test_mock_multi_rank)


# ═══════════════════════════════════════════════════════════════
# 测试组 3: NCCL Stub 行为检查
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 3] NCCL Stub 行为")

def test_nccl_stub_create():
    """NCCL stub 可以创建 (编译时启用了 ENABLE_DIST_NCCL)"""
    if not backend_available(DistBackend.NCCL):
        print("    ⚠️  NCCL 编译时未启用, 跳过")
        return
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.NCCL
    cfg.world_size = 2
    cfg.rank = 0
    cfg.local_device = 0
    comm = comm_create(cfg)
    assert comm is not None and comm != 0
    assert comm_world_size(comm) == 2
    assert comm_rank(comm) == 0
    comm_destroy(comm)

def test_nccl_stub_allreduce_throws():
    """NCCL stub 的 allReduce 应抛出异常 (未实际接入 NCCL)
    
    注意: 不实际调用 allReduce, 因为 C++ 异常会导致 abort().
    仅验证 NCCL stub 可以创建和查询元数据, allReduce 不可用是已知行为.
    """
    if not backend_available(DistBackend.NCCL):
        print("    ⚠️  NCCL 编译时未启用, 跳过")
        return
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.NCCL
    cfg.world_size = 2
    cfg.rank = 0
    cfg.local_device = 0
    comm = comm_create(cfg)
    assert comm_world_size(comm) == 2
    assert comm_rank(comm) == 0
    
    # 不调用 allReduce — NCCL stub 会 throw, C++ 异常穿过 ctypes 会导致 abort
    print("    ⚠️  NCCL stub allReduce 未实现 (调用会 abort), 跳过实际调用")
    print("    → 这是当前的关键差距: NCCL 后端需要接入真正的 NCCL 库")
    
    comm_destroy(comm)

test("NCCL Stub 创建", test_nccl_stub_create)
test("NCCL Stub AllReduce 行为", test_nccl_stub_allreduce_throws)


# ═══════════════════════════════════════════════════════════════
# 测试组 4: C++ Smoke Tests 验证 (通过 subprocess)
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 4] C++ Smoke Tests")

import subprocess

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'linux', 'x86_64', 'release')

def run_cpp_test(name, binary):
    binary_path = os.path.join(BUILD_DIR, binary)
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"二进制文件不存在: {binary_path}")
    result = subprocess.run([binary_path], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"exit code={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}")
    if "FAIL" in result.stdout:
        raise RuntimeError(f"测试输出包含 FAIL:\n{result.stdout}")
    print(f"    输出: {result.stdout.strip().split(chr(10))[-1]}")

def test_dist_smoke():
    run_cpp_test("dist-smoke", "llaisys-dist-smoke")

def test_tp_shard_smoke():
    run_cpp_test("tp-shard-smoke", "llaisys-tp-shard-smoke")

def test_tp_fwd_smoke():
    run_cpp_test("tp-fwd-smoke", "llaisys-tp-fwd-smoke")

def test_tp_cache_smoke():
    run_cpp_test("tp-cache-smoke", "llaisys-tp-cache-smoke")

test("dist-smoke (通信层)", test_dist_smoke)
test("tp-shard-smoke (权重分片)", test_tp_shard_smoke)
test("tp-fwd-smoke (TP 前向推理)", test_tp_fwd_smoke)
test("tp-cache-smoke (TP 缓存)", test_tp_cache_smoke)


# ═══════════════════════════════════════════════════════════════
# 测试组 5: GPU 环境检测
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 5] GPU 环境检测")

def test_gpu_count():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"], 
                                capture_output=True, text=True, timeout=10)
        gpus = result.stdout.strip().split('\n')
        print(f"    检测到 {len(gpus)} 张 GPU:")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu.strip()}")
        assert len(gpus) >= 2, f"分布式推理需要至少 2 张 GPU, 当前 {len(gpus)}"
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi 不可用")

test("GPU 数量与型号", test_gpu_count)


# ═══════════════════════════════════════════════════════════════
# 测试组 6: TP 配置矩阵验证
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 6] TP 配置矩阵")

def test_tp_configs():
    """验证不同 tp_size x rank 组合的 comm 创建"""
    for tp_size in [1, 2, 4, 8]:
        for rank in range(tp_size):
            cfg = LlaisysDistConfig()
            cfg.backend = DistBackend.MOCK
            cfg.world_size = tp_size
            cfg.rank = rank
            cfg.local_device = rank % 8
            comm = comm_create(cfg)
            assert comm_world_size(comm) == tp_size
            assert comm_rank(comm) == rank
            comm_destroy(comm)
    print(f"    验证了 tp_size=[1,2,4,8] × 全部 rank 共 15 个配置")

def test_invalid_rank():
    """rank >= world_size 应失败 (C++ 抛出 invalid_argument)
    
    注意: C++ 异常穿过 ctypes 会导致 abort, 所以用 subprocess 隔离测试.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')}')
from llaisys.libllaisys.distributed import DistBackend, LlaisysDistConfig, comm_create
cfg = LlaisysDistConfig()
cfg.backend = DistBackend.MOCK
cfg.world_size = 2
cfg.rank = 2
cfg.local_device = 0
comm = comm_create(cfg)
print("SHOULD_NOT_REACH")
"""],
        capture_output=True, text=True, timeout=10
    )
    # 期望进程退出(crash), 且不应打印 SHOULD_NOT_REACH
    assert "SHOULD_NOT_REACH" not in result.stdout, "无效 rank 没有被正确拒绝"
    print("    C++ 层正确拒绝了 rank >= world_size")

test("多 TP 配置矩阵", test_tp_configs)
test("无效 rank 处理", test_invalid_rank)


# ═══════════════════════════════════════════════════════════════
# 测试组 7: Python Qwen2 TP 接口 (低级)
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 7] Python Qwen2 TP 低级接口")

from llaisys.libllaisys.qwen2 import (
    LlaisysQwen2Meta,
    model_create,
    model_create_tp,
    model_destroy,
    model_get_tp_size,
    model_get_tp_rank,
    model_set_comm,
    load_weight,
    model_infer,
)

def test_qwen2_create_tp():
    """直接调用低级 C API 创建 TP 模型 (CPU, 小型模型)"""
    meta = LlaisysQwen2Meta()
    meta.dtype = 13  # FP32
    meta.nlayer = 2
    meta.hs = 64
    meta.nh = 8
    meta.nkvh = 2
    meta.dh = 8
    meta.di = 32
    meta.maxseq = 32
    meta.voc = 50
    meta.epsilon = 1e-6
    meta.theta = 10000.0
    meta.end_token = 0
    
    # tp_size=2, rank=0
    model = model_create_tp(ctypes.byref(meta), 0, 0, 2, 0)  # device=CPU=0
    assert model is not None and model != 0, "TP 模型创建失败"
    
    tp_size = model_get_tp_size(model)
    tp_rank = model_get_tp_rank(model)
    assert tp_size == 2, f"tp_size should be 2, got {tp_size}"
    assert tp_rank == 0, f"tp_rank should be 0, got {tp_rank}"
    
    model_destroy(model)

def test_qwen2_tp_with_mock_comm_infer():
    """创建 TP 模型 + Mock Comm + 加载假权重 + 推理"""
    import numpy as np
    
    meta = LlaisysQwen2Meta()
    meta.dtype = 13
    meta.nlayer = 1  # 最小层数
    meta.hs = 32
    meta.nh = 4
    meta.nkvh = 2
    meta.dh = 8
    meta.di = 16
    meta.maxseq = 16
    meta.voc = 20
    meta.epsilon = 1e-6
    meta.theta = 10000.0
    meta.end_token = 0
    
    # 创建 Mock Comm
    dist_cfg = LlaisysDistConfig()
    dist_cfg.backend = DistBackend.MOCK
    dist_cfg.world_size = 2
    dist_cfg.rank = 0
    dist_cfg.local_device = 0
    comm = comm_create(dist_cfg)
    
    # 创建 TP 模型
    model = model_create_tp(ctypes.byref(meta), 0, 0, 2, 0)
    model_set_comm(model, comm)
    
    # 加载假权重 (全部用 0.01 填充)
    def load_w(name, shape, val=0.01):
        data = np.full(shape, val, dtype=np.float32)
        sh = (ctypes.c_int64 * len(shape))(*shape)
        load_weight(model, name.encode('utf-8'), data.ctypes.data, len(shape), sh, 13)
    
    # 顶层权重
    load_w("model.embed_tokens.weight", (meta.voc, meta.hs))
    load_w("model.norm.weight", (meta.hs,), 1.0)
    load_w("lm_head.weight", (meta.voc, meta.hs))
    
    # Layer 0 权重
    q_out = meta.nh * meta.dh
    kv_out = meta.nkvh * meta.dh
    load_w("model.layers.0.input_layernorm.weight", (meta.hs,), 1.0)
    load_w("model.layers.0.post_attention_layernorm.weight", (meta.hs,), 1.0)
    load_w("model.layers.0.self_attn.q_proj.weight", (q_out, meta.hs))
    load_w("model.layers.0.self_attn.q_proj.bias", (q_out,))
    load_w("model.layers.0.self_attn.k_proj.weight", (kv_out, meta.hs))
    load_w("model.layers.0.self_attn.k_proj.bias", (kv_out,))
    load_w("model.layers.0.self_attn.v_proj.weight", (kv_out, meta.hs))
    load_w("model.layers.0.self_attn.v_proj.bias", (kv_out,))
    load_w("model.layers.0.self_attn.o_proj.weight", (meta.hs, q_out))
    load_w("model.layers.0.mlp.gate_proj.weight", (meta.di, meta.hs))
    load_w("model.layers.0.mlp.up_proj.weight", (meta.di, meta.hs))
    load_w("model.layers.0.mlp.down_proj.weight", (meta.hs, meta.di))
    
    # 推理
    tokens = (ctypes.c_int64 * 3)(1, 2, 3)
    out = model_infer(model, tokens, 3)
    print(f"    推理输出 token = {out} (有效范围 0..{meta.voc-1})")
    assert 0 <= out < meta.voc, f"输出 token {out} 超出范围"
    
    model_destroy(model)
    comm_destroy(comm)

test("Qwen2 TP 模型创建 (低级 API)", test_qwen2_create_tp)
test("Qwen2 TP + Mock Comm 推理闭环", test_qwen2_tp_with_mock_comm_infer)


# ═══════════════════════════════════════════════════════════════
# 测试组 8: tp_launch.py 脚本检查
# ═══════════════════════════════════════════════════════════════
print("\n[Test Group 8] tp_launch.py 脚本")

def test_tp_launch_syntax():
    """验证 tp_launch.py 语法正确"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'tp_launch.py')
    result = subprocess.run([sys.executable, "-c", f"import py_compile; py_compile.compile('{script}', doraise=True)"],
                           capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"语法检查失败: {result.stderr}")

def test_tp_launch_help():
    """验证 tp_launch.py --help 可用"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'tp_launch.py')
    result = subprocess.run([sys.executable, script, "--help"],
                           capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"--help 失败: {result.stderr}"
    assert "--tp-size" in result.stdout, "--tp-size 参数未找到"
    assert "--model" in result.stdout, "--model 参数未找到"

test("tp_launch.py 语法检查", test_tp_launch_syntax)
test("tp_launch.py --help", test_tp_launch_help)


# ═══════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  分布式推理测试汇总: {passed} 通过, {failed} 失败")
print("=" * 60)

if errors:
    print("\n失败的测试:")
    for name, err in errors:
        print(f"  ❌ {name}: {err}")

sys.exit(1 if failed > 0 else 0)
