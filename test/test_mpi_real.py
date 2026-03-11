#!/usr/bin/env python3
"""MPI 后端多进程通信测试.

使用 mpirun 启动多个进程, 验证 MPI allReduce 和 barrier 的正确性.
"""

import os
import sys
import subprocess
import tempfile
import json

# MPI 路径
MPI_PREFIX = os.environ.get("MPI_HOME", "/opt/hpcx/ompi")
MPIRUN = os.path.join(MPI_PREFIX, "bin", "mpirun")

# Worker 脚本: 在 MPI 进程内执行通信操作
WORKER_SCRIPT = r"""
import os, sys, json, ctypes

# 设置路径
sys.path.insert(0, os.environ.get("LLAISYS_PYTHON_PATH", "python"))

from llaisys.libllaisys.distributed import (
    DistBackend, LlaisysDistConfig, backend_available,
    comm_create, comm_destroy, comm_backend, comm_world_size, comm_rank,
    all_reduce_sum_f32, barrier,
)

def main():
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    my_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))

    # 创建 MPI comm
    config = LlaisysDistConfig()
    config.backend = DistBackend.MPI
    config.world_size = world_size
    config.rank = my_rank
    config.local_device = 0

    comm = comm_create(config)

    results = {}
    results["rank"] = comm_rank(comm)
    results["world_size"] = comm_world_size(comm)
    results["backend"] = comm_backend(comm)

    # 测试 1: allReduce on CPU data
    # 每个 rank 发送 [rank+1, rank+1, rank+1, rank+1]
    # 预期结果: [sum(1..N), sum(1..N), sum(1..N), sum(1..N)]
    n = 4
    buf = (ctypes.c_float * n)(*([float(my_rank + 1)] * n))
    all_reduce_sum_f32(comm, buf, n)
    expected_sum = sum(range(1, world_size + 1))
    results["allreduce_result"] = [buf[i] for i in range(n)]
    results["allreduce_expected"] = float(expected_sum)
    results["allreduce_correct"] = all(abs(buf[i] - expected_sum) < 1e-5 for i in range(n))

    # 测试 2: 不同数据的 allReduce
    # rank k 发送 [k*10, k*10+1, k*10+2, k*10+3]
    buf2 = (ctypes.c_float * n)(*[float(my_rank * 10 + i) for i in range(n)])
    all_reduce_sum_f32(comm, buf2, n)
    expected2 = [sum(r * 10 + i for r in range(world_size)) for i in range(n)]
    results["allreduce2_result"] = [buf2[i] for i in range(n)]
    results["allreduce2_expected"] = expected2
    results["allreduce2_correct"] = all(abs(buf2[i] - expected2[i]) < 1e-4 for i in range(n))

    # 测试 3: barrier
    barrier(comm)
    results["barrier_ok"] = True

    # 测试 4: 大数据量 allReduce (1024 elements)
    big_n = 1024
    big_buf = (ctypes.c_float * big_n)(*[float(my_rank + 1)] * big_n)
    all_reduce_sum_f32(comm, big_buf, big_n)
    big_correct = all(abs(big_buf[i] - expected_sum) < 1e-4 for i in range(big_n))
    results["big_allreduce_correct"] = big_correct

    # 测试 5: 零长度 allReduce (不应崩溃)
    empty_buf = (ctypes.c_float * 1)(0.0)
    all_reduce_sum_f32(comm, empty_buf, 0)
    results["empty_allreduce_ok"] = True

    comm_destroy(comm)
    results["destroy_ok"] = True

    # 输出 JSON 结果到文件
    out_file = os.environ.get("RESULT_FILE", f"/tmp/mpi_test_rank{my_rank}.json")
    with open(out_file, "w") as f:
        json.dump(results, f)

    print(f"[MPI rank {my_rank}] DONE: allreduce={results['allreduce_correct']}, "
          f"allreduce2={results['allreduce2_correct']}, "
          f"big={results['big_allreduce_correct']}", flush=True)

if __name__ == "__main__":
    main()
"""


def run_mpi_test(np_count, label=""):
    """Run MPI test with np_count processes."""
    print(f"\n{'='*60}")
    print(f"  MPI 测试: {np_count} 进程 {label}")
    print(f"{'='*60}")

    # 写入 worker 脚本到临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(WORKER_SCRIPT)
        worker_path = f.name

    result_files = [f"/tmp/mpi_test_rank{r}.json" for r in range(np_count)]

    # 清理旧结果
    for rf in result_files:
        if os.path.exists(rf):
            os.remove(rf)

    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{MPI_PREFIX}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["OPAL_PREFIX"] = MPI_PREFIX
        env["LLAISYS_PYTHON_PATH"] = "python"

        cmd = [
            MPIRUN,
            "-np", str(np_count),
            sys.executable, worker_path,
        ]

        print(f"  CMD: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=60,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        print(f"  返回码: {proc.returncode}")
        if proc.stdout.strip():
            for line in proc.stdout.strip().split("\n"):
                print(f"  stdout: {line}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().split("\n")[-10:]:
                print(f"  stderr: {line}")

        if proc.returncode != 0:
            print(f"  ❌ mpirun 返回非零: {proc.returncode}")
            return False

        # 读取每个 rank 的结果
        all_pass = True
        for r in range(np_count):
            rf = result_files[r]
            if not os.path.exists(rf):
                print(f"  ❌ rank {r} 没有输出结果文件")
                all_pass = False
                continue

            with open(rf) as f:
                res = json.load(f)

            checks = [
                ("rank", res.get("rank") == r, f"rank={res.get('rank')}, expected={r}"),
                ("world_size", res.get("world_size") == np_count, f"ws={res.get('world_size')}"),
                ("backend", res.get("backend") == 2, f"backend={res.get('backend')}"),
                ("allreduce", res.get("allreduce_correct"), f"result={res.get('allreduce_result')}"),
                ("allreduce2", res.get("allreduce2_correct"), f"result={res.get('allreduce2_result')}"),
                ("barrier", res.get("barrier_ok"), ""),
                ("big_allreduce", res.get("big_allreduce_correct"), ""),
                ("empty_allreduce", res.get("empty_allreduce_ok"), ""),
                ("destroy", res.get("destroy_ok"), ""),
            ]

            for name, passed, detail in checks:
                status = "✅" if passed else "❌"
                detail_str = f" ({detail})" if detail else ""
                print(f"  {status} rank {r} {name}{detail_str}")
                if not passed:
                    all_pass = False

        if all_pass:
            print(f"\n  ✅ {np_count} 进程 MPI 测试全部通过!")
        else:
            print(f"\n  ❌ {np_count} 进程 MPI 测试存在失败")

        return all_pass

    except subprocess.TimeoutExpired:
        print(f"  ❌ MPI 测试超时 (60s)")
        return False
    finally:
        os.unlink(worker_path)
        for rf in result_files:
            if os.path.exists(rf):
                os.remove(rf)


def main():
    print("=" * 60)
    print("  MPI 后端通信测试")
    print("=" * 60)

    # 检查 mpirun 可用
    if not os.path.isfile(MPIRUN):
        print(f"❌ mpirun 不存在: {MPIRUN}")
        sys.exit(1)
    print(f"✅ mpirun: {MPIRUN}")

    results = {}

    # 2 进程测试
    results["np2"] = run_mpi_test(2, "基础 allReduce + barrier")

    # 4 进程测试
    results["np4"] = run_mpi_test(4, "4路 allReduce")

    # 8 进程测试
    results["np8"] = run_mpi_test(8, "8路 allReduce")

    # 汇总
    print(f"\n{'='*60}")
    print("  MPI 测试汇总")
    print(f"{'='*60}")
    total = 0
    passed = 0
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} {name}")
        total += 1
        if ok:
            passed += 1

    print(f"\n  总计: {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
