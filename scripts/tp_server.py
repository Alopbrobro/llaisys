#!/usr/bin/env python3
"""
LLAISYS TP Server Launcher — 张量并行推理服务器

Rank 0: 运行 Web 服务器 (FastAPI), 接收用户请求并协调推理
Rank 1-N: 运行 "follower" 循环, 从 Rank 0 接收推理指令并同步执行

架构:
  用户 → Web API (rank 0) → TP Coordinator → 所有 rank 同步 prefill/decode
                                              ↓
                                         NCCL allReduce (GPU 直通)

Usage:
  python scripts/tp_server.py --tp-size 4 \
    --model /path/to/model --device nvidia --port 8000
"""

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import signal
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description="LLAISYS TP Server")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="nvidia")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids (default: 0..tp_size-1)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    if args.gpus:
        gpu_list = args.gpus
    else:
        gpu_list = ",".join(str(i) for i in range(args.tp_size))

    # 创建 coordination socket 地址
    coord_port_base = 29500
    coord_file = f"/tmp/llaisys_tp_coord_{os.getpid()}"

    # 清理 NCCL ID 文件
    nccl_id_file = "/tmp/llaisys_nccl_id"
    if os.path.exists(nccl_id_file):
        os.remove(nccl_id_file)

    # Worker 脚本路径
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "tp_worker.py")

    python = sys.executable
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processes = []

    print(f"[tp-server] Launching {args.tp_size} TP ranks ...")
    print(f"[tp-server] GPUs: {gpu_list}")
    print(f"[tp-server] Model: {args.model}")
    print(f"[tp-server] Coordination base port: {coord_port_base}")

    for rank in range(args.tp_size):
        cmd = [
            python, worker_script,
            "--model", args.model,
            "--device", args.device,
            "--tp-size", str(args.tp_size),
            "--tp-rank", str(rank),
            "--coord-port", str(coord_port_base),
            "--max-seq-len", str(args.max_seq_len),
        ]

        if rank == 0:
            # Rank 0 runs the web server
            cmd += ["--server", "--host", args.host, "--port", str(args.port)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
        env["PYTHONPATH"] = os.path.join(work_dir, "python") + ":" + env.get("PYTHONPATH", "")

        log_file = f"/tmp/tp_server_rank{rank}.log"
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, cwd=work_dir, env=env,
                                    stdout=lf, stderr=subprocess.STDOUT)
        processes.append(proc)
        print(f"  rank {rank}: PID={proc.pid}, log={log_file}")

        if rank == 0:
            time.sleep(2)  # 让 rank 0 先启动创建 NCCL ID

    print(f"[tp-server] All ranks launched.")
    print(f"[tp-server] Web UI will be at http://{args.host}:{args.port}/")
    print(f"[tp-server] Monitoring logs... Press Ctrl+C to stop.")

    def _cleanup(signum, frame):
        print("\n[tp-server] Shutting down all ranks ...")
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        # Clean up nccl id file
        if os.path.exists(nccl_id_file):
            os.remove(nccl_id_file)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Block waiting for all processes
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
