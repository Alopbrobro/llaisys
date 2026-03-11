#!/usr/bin/env python3
"""
LLAISYS TP Launcher — 使用 subprocess 在单机上启动多个 TP rank 进程.

用法:
  # 2-GPU TP 推理 (server 模式)
  python scripts/tp_launch.py --tp-size 2 --model /path/to/model --device nvidia

  # 使用 MPI (需要 mpirun 可用)
  python scripts/tp_launch.py --tp-size 2 --model /path/to/model --use-mpi

  # 仅 CLI 聊天 (非 server)
  python scripts/tp_launch.py --tp-size 2 --model /path/to/model --mode cli

无 MPI 时, 使用 subprocess 为每个 rank 启动 server.app:main(), 
各 rank 监听不同端口 (rank 0 = base_port, rank 1 = base_port+1, ...).
rank 0 是主进程, 其余为 worker.

注意: 当前 Mock 通信后端下多进程不共享数据 (仅用于结构验证).
     真正的多 GPU TP 需配合 NCCL 后端使用.
"""

import argparse
import os
import subprocess
import sys
import signal


def main():
    parser = argparse.ArgumentParser(description="LLAISYS TP Launcher")
    parser.add_argument("--tp-size", type=int, required=True,
                        help="Tensor parallelism degree")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace repo id")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "nvidia"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000,
                        help="Base port (rank 0); rank i uses port+i")
    parser.add_argument("--use-mpi", action="store_true",
                        help="Use mpirun to launch (requires MPI installed)")
    parser.add_argument("--mode", type=str, default="server",
                        choices=["server", "cli"],
                        help="Launch mode: server (HTTP API) or cli (chat)")

    args = parser.parse_args()

    if args.use_mpi:
        _launch_mpi(args)
    else:
        _launch_subprocess(args)


def _launch_mpi(args):
    """使用 mpirun 启动 tp_size 个进程."""
    cmd = [
        "mpirun", "-np", str(args.tp_size),
        "--allow-run-as-root",
        sys.executable, "-m", "server.app",
        "--model", args.model,
        "--device", args.device,
        "--host", args.host,
        "--port", str(args.port),
        "--tp-size", str(args.tp_size),
        # tp-rank 由 MPI 环境变量注入, 需在 server.app 中读取
        # 或使用 OMPI_COMM_WORLD_RANK (OpenMPI)
    ]
    print(f"[tp-launch] MPI command: {' '.join(cmd)}")
    os.execvp("mpirun", cmd)


def _launch_subprocess(args):
    """使用 subprocess 启动多个独立进程, 每个进程一个 TP rank."""
    processes = []
    python = sys.executable
    
    # 确定工作目录 (项目 python/ 目录)
    work_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")

    print(f"[tp-launch] Launching {args.tp_size} TP ranks via subprocess ...")

    for rank in range(args.tp_size):
        port = args.port + rank
        cmd = [
            python, "-m", "server.app",
            "--model", args.model,
            "--device", args.device,
            "--host", args.host,
            "--port", str(port),
            "--tp-size", str(args.tp_size),
            "--tp-rank", str(rank),
        ]
        env = os.environ.copy()
        # TP 模式: 所有 rank 共享相同的 GPU 列表, device_id=tp_rank 选择对应 GPU
        # (不能每个 rank 只看到 1 张卡, 否则 device_id=tp_rank > 0 时会失败)
        gpu_list = ",".join(str(i) for i in range(args.tp_size))
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", gpu_list)
        
        print(f"  rank {rank}: port={port}, CUDA_VISIBLE_DEVICES={rank}")
        proc = subprocess.Popen(cmd, cwd=work_dir, env=env)
        processes.append(proc)

    print(f"[tp-launch] All {args.tp_size} ranks launched. Rank 0 at port {args.port}.")
    print(f"[tp-launch] Press Ctrl+C to stop all ranks.")

    # 等待所有进程, 或被 Ctrl+C 中断
    def _cleanup(signum, frame):
        print("\n[tp-launch] Terminating all ranks ...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # 阻塞等待所有进程
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
