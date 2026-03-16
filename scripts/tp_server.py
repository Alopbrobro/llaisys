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


def _find_pids_on_port(port: int) -> list:
    """Find PIDs listening on *port* by parsing /proc/net/tcp{,6} (pure Python)."""
    hex_port = f"{port:04X}"
    inodes = set()

    for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    # parts[1] = local_address (hex IP:PORT), parts[3] = state
                    if parts[3] == "0A" and parts[1].endswith(f":{hex_port}"):
                        inodes.add(parts[9])
        except FileNotFoundError:
            continue

    if not inodes:
        return []

    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    link = os.readlink(f"{fd_dir}/{fd}")
                    if link.startswith("socket:[") and link[8:-1] in inodes:
                        pids.append(int(entry))
                        break
                except (OSError, ValueError):
                    continue
        except (OSError, PermissionError):
            continue

    return pids


def _find_related_tp_workers(coord_port: int) -> list:
    """Find all tp_worker processes using the same --coord-port (listeners + followers)."""
    pids = []
    target = f"--coord-port {coord_port}"
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            if "tp_worker" in cmdline and target in cmdline:
                pids.append(int(entry))
        except (OSError, PermissionError):
            continue
    return pids


def _kill_stale_port(port: int):
    """Kill any stale tp_worker processes occupying the coordination port."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.close()
        return
    except OSError:
        pass

    print(f"[tp-server] Port {port} is occupied by a stale process, cleaning up ...")

    stale_pids = set(_find_pids_on_port(port)) | set(_find_related_tp_workers(port))
    my_pid = os.getpid()
    stale_pids.discard(my_pid)

    if not stale_pids:
        print(f"[tp-server] WARNING: Could not identify stale process on port {port}")
        return

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[tp-server] Sent SIGTERM to stale PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[tp-server] No permission to kill PID {pid}")

    # Wait for processes to exit, escalate to SIGKILL if needed
    deadline = time.time() + 5
    while time.time() < deadline:
        still_alive = []
        for pid in stale_pids:
            try:
                os.kill(pid, 0)  # check if alive
                still_alive.append(pid)
            except ProcessLookupError:
                pass
        if not still_alive:
            break
        time.sleep(0.5)
    else:
        for pid in still_alive:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[tp-server] Sent SIGKILL to stale PID {pid}")
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1)

    print(f"[tp-server] Stale processes cleaned up")


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

    # 清理上次残留: NCCL ID 文件 + coordination 端口
    nccl_id_file = "/tmp/llaisys_nccl_id"
    if os.path.exists(nccl_id_file):
        os.remove(nccl_id_file)
    _kill_stale_port(coord_port_base)

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

    # Monitor processes: detect crashes and print relevant logs
    while True:
        all_done = True
        any_crashed = False
        for i, p in enumerate(processes):
            ret = p.poll()
            if ret is None:
                all_done = False
            elif ret != 0 and not getattr(p, "_reported", False):
                p._reported = True
                any_crashed = True
                log_file = f"/tmp/tp_server_rank{i}.log"
                print(f"\n[tp-server] *** rank {i} (PID={p.pid}) crashed with exit code {ret} ***")
                try:
                    with open(log_file, "r", errors="replace") as f:
                        lines = f.readlines()
                    tail = lines[-30:] if len(lines) > 30 else lines
                    print(f"[tp-server] Last lines of {log_file}:")
                    for line in tail:
                        print(f"  | {line}", end="")
                    print()
                except Exception:
                    print(f"[tp-server] (could not read {log_file})")

        if any_crashed:
            print("[tp-server] Shutting down all remaining ranks ...")
            for j, pp in enumerate(processes):
                if pp.poll() is None:
                    pp.terminate()
            for j, pp in enumerate(processes):
                try:
                    pp.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pp.kill()
            if os.path.exists(nccl_id_file):
                os.remove(nccl_id_file)
            sys.exit(1)

        if all_done:
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
