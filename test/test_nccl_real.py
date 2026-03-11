#!/usr/bin/env python3
"""
NCCL 多 GPU 真实通信测试

使用 subprocess 启动多个进程, 每个进程绑定一张 GPU,
通过真正的 NCCL allReduce 验证跨卡通信.

用法:
    PYTHONPATH=python python3 test/test_nccl_real.py
"""

import subprocess
import sys
import os
import tempfile
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PYTHON_DIR = os.path.join(PROJECT_DIR, 'python')

# NCCL unique ID 共享文件
NCCL_ID_FILE = os.path.join(tempfile.gettempdir(), f'llaisys_nccl_test_{os.getpid()}')

# 每个 rank 的 worker 脚本
WORKER_SCRIPT = r'''
import sys
import os
import ctypes
import json

sys.path.insert(0, os.environ["LLAISYS_PYTHON_DIR"])

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
gpu_id = int(os.environ["GPU_ID"])
result_file = os.environ["RESULT_FILE"]

results = {"rank": rank, "tests": {}}

try:
    from llaisys.libllaisys.distributed import (
        DistBackend, LlaisysDistConfig,
        comm_create, comm_destroy,
        comm_world_size, comm_rank,
        all_reduce_sum_f32, barrier,
    )

    # Test 1: NCCL comm 创建
    cfg = LlaisysDistConfig()
    cfg.backend = DistBackend.NCCL
    cfg.world_size = world_size
    cfg.rank = rank
    cfg.local_device = gpu_id

    comm = comm_create(cfg)
    assert comm is not None and comm != 0
    assert comm_world_size(comm) == world_size
    assert comm_rank(comm) == rank
    results["tests"]["nccl_create"] = "PASS"
    print(f"[rank {rank}] NCCL comm created on GPU {gpu_id}", flush=True)

    # Test 2: allReduce 验证
    # 每个 rank 发送 [rank+1, rank+1, rank+1, rank+1]
    # allReduce sum 后结果应为 [sum(1..world_size), ...] = [world_size*(world_size+1)/2, ...]
    data = (ctypes.c_float * 4)(float(rank + 1), float(rank + 1), float(rank + 1), float(rank + 1))
    
    # 需要在 GPU 内存中操作 — 使用 CUDA
    import ctypes as ct
    
    # 加载 cudart
    try:
        cudart = ct.CDLL("libcudart.so")
    except OSError:
        cudart = ct.CDLL("libcudart.so.12")
    
    cudart.cudaSetDevice(gpu_id)
    
    # GPU 内存分配
    gpu_ptr = ct.c_void_p()
    nbytes = 4 * ct.sizeof(ct.c_float)
    cudart.cudaMalloc(ct.byref(gpu_ptr), nbytes)
    
    # H2D 拷贝
    cudart.cudaMemcpy(gpu_ptr, ct.cast(data, ct.c_void_p), nbytes, 1)  # cudaMemcpyHostToDevice=1
    
    # allReduce on GPU memory
    all_reduce_sum_f32(comm, ct.cast(gpu_ptr, ct.POINTER(ct.c_float)), 4)
    
    # D2H 拷贝
    result_data = (ct.c_float * 4)()
    cudart.cudaMemcpy(ct.cast(result_data, ct.c_void_p), gpu_ptr, nbytes, 2)  # cudaMemcpyDeviceToHost=2
    
    cudart.cudaFree(gpu_ptr)
    
    expected = float(world_size * (world_size + 1) // 2)
    actual = [result_data[i] for i in range(4)]
    print(f"[rank {rank}] allReduce result: {actual}, expected: [{expected}]*4", flush=True)
    
    if all(abs(v - expected) < 0.01 for v in actual):
        results["tests"]["nccl_allreduce"] = "PASS"
    else:
        results["tests"]["nccl_allreduce"] = f"FAIL: got {actual}, expected [{expected}]*4"

    # Test 3: barrier
    barrier(comm)
    results["tests"]["nccl_barrier"] = "PASS"
    print(f"[rank {rank}] barrier completed", flush=True)

    comm_destroy(comm)
    results["tests"]["nccl_destroy"] = "PASS"
    results["status"] = "SUCCESS"

except Exception as e:
    import traceback
    results["status"] = "ERROR"
    results["error"] = str(e)
    results["traceback"] = traceback.format_exc()
    print(f"[rank {rank}] ERROR: {e}", flush=True)

# 写结果
with open(result_file, "w") as f:
    json.dump(results, f)
'''


def run_nccl_test(world_size=2, gpu_ids=None):
    """启动多进程 NCCL 测试"""
    if gpu_ids is None:
        gpu_ids = list(range(world_size))
    
    print(f"\n{'='*60}")
    print(f"  NCCL 多 GPU 真实通信测试 (world_size={world_size}, GPUs={gpu_ids})")
    print(f"{'='*60}\n")
    
    # 准备各 rank 的结果文件
    result_files = {}
    processes = []
    
    # 清理旧的 ID 文件
    if os.path.exists(NCCL_ID_FILE):
        os.remove(NCCL_ID_FILE)
    
    # 启动所有 rank (rank 0 先启动以生成 unique ID)
    for rank in range(world_size):
        result_file = os.path.join(tempfile.gettempdir(), f'nccl_test_rank{rank}_{os.getpid()}.json')
        result_files[rank] = result_file
        
        env = os.environ.copy()
        env['RANK'] = str(rank)
        env['WORLD_SIZE'] = str(world_size)
        env['GPU_ID'] = str(gpu_ids[rank])
        env['LLAISYS_PYTHON_DIR'] = PYTHON_DIR
        env['LLAISYS_NCCL_ID_FILE'] = NCCL_ID_FILE
        env['RESULT_FILE'] = result_file
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[rank])
        
        # 注意: 当 CUDA_VISIBLE_DEVICES 只设一个设备时, 设备 ID 变为 0
        env['GPU_ID'] = '0'
        
        proc = subprocess.Popen(
            [sys.executable, '-c', WORKER_SCRIPT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((rank, proc))
        
        # rank 0 先启动并等一小会, 确保 unique ID 文件写入
        if rank == 0:
            time.sleep(0.5)
    
    # 等待所有进程完成
    all_passed = True
    for rank, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=60)
            print(f"[rank {rank}] stdout: {stdout.decode().strip()}")
            if stderr.decode().strip():
                # 过滤 NCCL 日志
                for line in stderr.decode().strip().split('\n'):
                    if 'NCCL' in line or 'error' in line.lower() or 'Error' in line:
                        print(f"[rank {rank}] stderr: {line.strip()}")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[rank {rank}] TIMEOUT!")
            all_passed = False
            continue
    
    # 收集结果
    print(f"\n{'─'*60}")
    print(f"  测试结果详情")
    print(f"{'─'*60}")
    
    for rank in range(world_size):
        result_file = result_files[rank]
        if os.path.exists(result_file):
            with open(result_file) as f:
                result = json.load(f)
            
            status = result.get('status', 'UNKNOWN')
            print(f"\n  Rank {rank} ({status}):")
            
            if status == 'ERROR':
                print(f"    Error: {result.get('error', 'unknown')}")
                if 'traceback' in result:
                    for line in result['traceback'].split('\n')[-5:]:
                        if line.strip():
                            print(f"    {line}")
                all_passed = False
            else:
                for test_name, test_result in result.get('tests', {}).items():
                    icon = '✅' if test_result == 'PASS' else '❌'
                    print(f"    {icon} {test_name}: {test_result}")
                    if test_result != 'PASS':
                        all_passed = False
        else:
            print(f"\n  Rank {rank}: 结果文件不存在")
            all_passed = False
        
        # 清理
        if os.path.exists(result_file):
            os.remove(result_file)
    
    # 清理 NCCL ID 文件
    if os.path.exists(NCCL_ID_FILE):
        os.remove(NCCL_ID_FILE)
    
    return all_passed


def main():
    # 检查 GPU 数量
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    gpu_count = len(result.stdout.strip().split('\n'))
    print(f"检测到 {gpu_count} 张 GPU")
    
    all_ok = True
    
    # Test 1: 2 GPU NCCL
    if gpu_count >= 2:
        ok = run_nccl_test(world_size=2, gpu_ids=[2, 3])  # 用空闲的 GPU 2, 3
        all_ok = all_ok and ok
    else:
        print("⚠️  GPU 不足 2 张, 跳过 2-GPU 测试")
    
    # Test 2: 4 GPU NCCL (如果有足够的 GPU)
    if gpu_count >= 4:
        # 清理旧文件
        if os.path.exists(NCCL_ID_FILE):
            os.remove(NCCL_ID_FILE)
        ok = run_nccl_test(world_size=4, gpu_ids=[2, 3, 4, 1])  # 用空闲的
        all_ok = all_ok and ok
    else:
        print("⚠️  GPU 不足 4 张, 跳过 4-GPU 测试")
    
    print(f"\n{'='*60}")
    if all_ok:
        print("  🎉 NCCL 多 GPU 通信测试: 全部通过!")
    else:
        print("  ❌ NCCL 多 GPU 通信测试: 有失败项")
    print(f"{'='*60}")
    
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
