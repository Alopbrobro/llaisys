#!/usr/bin/env python3
"""
LLAISYS TP Worker — TP 感知的推理工作进程

每个 TP rank 运行一个此进程:
  - rank 0: 运行 Web 服务器 + 推理引擎, 通过 TCP 向 followers 广播 prefill/decode 命令
  - rank 1-N: 运行 follower 循环, 等待 rank 0 的命令并同步执行相同的 C++ 模型操作

协调协议 (TCP):
  rank 0 → followers: JSON 消息
    {"cmd": "prefill", "slot_id": 0, "token_ids": [...], "temperature": 0.8, "top_k": 50, "top_p": 0.9}
    {"cmd": "decode",  "active_slots": [0,1], "current_tokens": [123,456], "temperature": 0.8, "top_k": 50, "top_p": 0.9}
    {"cmd": "slot_reset", "slot_id": 0}
    {"cmd": "slot_restore", "slot_id": 0, ...}  (不实际支持, followers 无 snapshot 数据)
    {"cmd": "shutdown"}
  followers → rank 0: "ok\n" (每条命令执行完毕后回复)
"""

import argparse
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="[rank %(rank)s] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


logger = logging.getLogger("tp_worker")


# ── 消息收发工具 ─────────────────────────────────────────────────

def send_msg(sock: socket.socket, data: dict):
    """发送一条 JSON 消息 (4 字节长度头 + UTF-8 payload)."""
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> dict:
    """接收一条 JSON 消息."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack("!I", header)[0]
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """确保接收恰好 n 字节."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── TP Coordinator (rank 0 端) ───────────────────────────────────

class TPCoordinator:
    """Rank 0 创建, 管理到所有 follower 的 TCP 连接."""

    def __init__(self, tp_size: int, coord_port: int):
        self.tp_size = tp_size
        self.coord_port = coord_port
        self.conns = {}  # rank → socket
        self._server_sock = None

    def start_listening(self, retries: int = 30, retry_interval: float = 1.0):
        """绑定端口并开始监听 (不阻塞等待连接).

        在模型加载之前调用, 使 follower 在 rank 0 加载模型期间就能建立连接,
        避免 follower 因为端口未监听而超时崩溃.

        如果端口被前一次运行的残留进程占用, 会自动重试直到端口可用.
        """
        if self._server_sock is not None:
            return

        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", self.coord_port))
                sock.listen(self.tp_size)
                self._server_sock = sock
                logger.info(f"Coordinator listening on port {self.coord_port}")
                return
            except OSError as e:
                sock.close()
                if attempt < retries - 1:
                    logger.warning(
                        f"Port {self.coord_port} in use ({e}), "
                        f"retrying in {retry_interval}s ... ({attempt + 1}/{retries})"
                    )
                    time.sleep(retry_interval)
                else:
                    raise OSError(
                        f"Failed to bind port {self.coord_port} after {retries} attempts. "
                        f"A stale process may still be using it. "
                        f"Try: fuser -k {self.coord_port}/tcp"
                    ) from e

    def wait_for_followers(self, timeout: float = 300):
        """等待 tp_size-1 个 follower 连接."""
        if self._server_sock is None:
            self.start_listening()
        self._server_sock.settimeout(timeout)

        logger.info(f"Waiting for {self.tp_size - 1} followers on port {self.coord_port} ...")

        while len(self.conns) < self.tp_size - 1:
            conn, addr = self._server_sock.accept()
            # 接收 follower 自报 rank
            msg = recv_msg(conn)
            if msg and "rank" in msg:
                rank = msg["rank"]
                self.conns[rank] = conn
                logger.info(f"Follower rank {rank} connected from {addr}")
            else:
                conn.close()

        logger.info(f"All {self.tp_size - 1} followers connected")

    def broadcast(self, cmd: dict):
        """向所有 follower 广播一条命令."""
        for rank, conn in self.conns.items():
            send_msg(conn, cmd)

    def wait_all(self):
        """等待所有 follower 回复 'ok'."""
        for rank, conn in self.conns.items():
            msg = recv_msg(conn)
            # 'ok' response

    def broadcast_and_wait(self, cmd: dict):
        """广播命令并等待所有 follower 执行完成."""
        self.broadcast(cmd)
        self.wait_all()

    def shutdown(self):
        """通知所有 follower 关闭."""
        try:
            self.broadcast({"cmd": "shutdown"})
        except Exception:
            pass
        for conn in self.conns.values():
            try:
                conn.close()
            except Exception:
                pass
        if self._server_sock:
            self._server_sock.close()


# ── BatchContext TP Wrapper ──────────────────────────────────────

class TPBatchContext:
    """包装原始 BatchContext, 在 prefill/decode 前广播到 followers."""

    def __init__(self, real_ctx, coordinator: TPCoordinator):
        self._ctx = real_ctx
        self._coord = coordinator

    def prefill(self, slot_id, token_ids, temperature=0.8, top_k=50, top_p=0.9):
        """同步所有 rank 执行 prefill."""
        # 1. 广播命令到 followers
        self._coord.broadcast({
            "cmd": "prefill",
            "slot_id": slot_id,
            "token_ids": list(token_ids),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        # 2. Rank 0 自己也执行
        result = self._ctx.prefill(
            slot_id=slot_id,
            token_ids=token_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        # 3. 等待 followers 完成
        self._coord.wait_all()
        return result

    def decode(self, active_slots, current_tokens, temperature=0.8, top_k=50, top_p=0.9):
        """同步所有 rank 执行 decode."""
        self._coord.broadcast({
            "cmd": "decode",
            "active_slots": list(active_slots),
            "current_tokens": list(current_tokens),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        result = self._ctx.decode(
            active_slots=active_slots,
            current_tokens=current_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        self._coord.wait_all()
        return result

    def slot_reset(self, slot_id):
        """同步所有 rank 执行 slot_reset."""
        self._coord.broadcast({"cmd": "slot_reset", "slot_id": slot_id})
        self._ctx.slot_reset(slot_id)
        self._coord.wait_all()

    def slot_save(self, slot_id):
        """只在 rank 0 执行, follower 不需要 snapshot."""
        return self._ctx.slot_save(slot_id)

    def slot_restore(self, slot_id, snapshot):
        """只在 rank 0 执行 (follower snapshot 不可跨进程传输)."""
        return self._ctx.slot_restore(slot_id, snapshot)

    def slot_get_pos(self, slot_id):
        return self._ctx.slot_get_pos(slot_id)


# ── Monkey-patch Engine 的 create_batch_context ──────────────────

def patch_engine_for_tp(engine, coordinator: TPCoordinator):
    """Patch InferenceEngine 使其 worker loop 使用 TPBatchContext."""
    original_worker = engine._worker_loop

    def patched_worker():
        """替换 _worker_loop, 使 batch_ctx 被 TPBatchContext 包装."""
        from server.engine import RequestStatus
        import time as _time

        engine_logger = logging.getLogger("llaisys.engine")
        engine_logger.info("TP-patched continuous batching worker loop started")

        # 创建原始 BatchContext
        real_batch_ctx = engine.model.create_batch_context(
            engine.max_batch_size, engine.max_seq_per_slot
        )
        # 包装为 TP 版本
        batch_ctx = TPBatchContext(real_batch_ctx, coordinator)

        running = {}
        free_slots = list(range(engine.max_batch_size))

        while engine._running:
            # ── 1. Admit ──
            admitted = 0
            while free_slots and engine.queue.waiting_count > 0:
                new_requests = engine.queue.get_pending(max_count=1)
                if not new_requests:
                    break
                req = new_requests[0]

                if req.status == RequestStatus.CANCELLED:
                    engine.queue.mark_done(req.request_id)
                    continue

                slot_id = free_slots.pop(0)
                req.started_at = _time.time()
                req.status = RequestStatus.PREFILLING

                try:
                    # Prefix matching (disabled for TP to avoid complexity)
                    batch_ctx.slot_reset(slot_id)
                    remaining_ids = req.input_ids

                    if remaining_ids:
                        first_token = batch_ctx.prefill(
                            slot_id=slot_id,
                            token_ids=remaining_ids,
                            temperature=req.params.temperature,
                            top_k=req.params.top_k,
                            top_p=req.params.top_p,
                        )
                    else:
                        first_token = batch_ctx.prefill(
                            slot_id=slot_id,
                            token_ids=[req.input_ids[-1]],
                            temperature=req.params.temperature,
                            top_k=req.params.top_k,
                            top_p=req.params.top_p,
                        )

                    first_token = int(first_token)

                except Exception as e:
                    engine_logger.error(f"Prefill error for {req.request_id}: {e}", exc_info=True)
                    free_slots.append(slot_id)
                    engine._finish_request(req, error=e)
                    engine.queue.mark_done(req.request_id)
                    continue

                req.generated_tokens.append(first_token)
                req.last_token = first_token
                req.status = RequestStatus.DECODING

                if req.stream and req.output_queue:
                    engine._send_token_async(req, first_token)

                if (first_token == engine._eos_token_id or
                        len(req.generated_tokens) >= req.params.max_tokens):
                    free_slots.append(slot_id)
                    batch_ctx.slot_reset(slot_id)
                    engine._finish_request(req)
                    engine.queue.mark_done(req.request_id)
                else:
                    running[slot_id] = req
                    admitted += 1

            # ── 2. Decode ──
            if running:
                active_slots = list(running.keys())
                current_tokens = [running[s].last_token for s in active_slots]

                first_req = running[active_slots[0]]
                batch_temp = first_req.params.temperature
                batch_top_k = first_req.params.top_k
                batch_top_p = first_req.params.top_p

                try:
                    next_tokens = batch_ctx.decode(
                        active_slots=active_slots,
                        current_tokens=current_tokens,
                        temperature=batch_temp,
                        top_k=batch_top_k,
                        top_p=batch_top_p,
                    )
                except Exception as e:
                    engine_logger.error(f"Batch decode error: {e}", exc_info=True)
                    for sid in list(running.keys()):
                        req = running.pop(sid)
                        free_slots.append(sid)
                        batch_ctx.slot_reset(sid)
                        engine._finish_request(req, error=e)
                        engine.queue.mark_done(req.request_id)
                    continue

                # ── 3. Handle results ──
                finished_slots = []
                for slot_id, next_tok in zip(active_slots, next_tokens):
                    req = running[slot_id]
                    next_tok = int(next_tok)
                    req.generated_tokens.append(next_tok)
                    req.last_token = next_tok

                    if req.stream and req.output_queue:
                        engine._send_token_async(req, next_tok)

                    if (next_tok == engine._eos_token_id or
                            len(req.generated_tokens) >= req.params.max_tokens or
                            req.status == RequestStatus.CANCELLED):
                        finished_slots.append(slot_id)

                for slot_id in finished_slots:
                    req = running.pop(slot_id)
                    batch_ctx.slot_reset(slot_id)
                    free_slots.append(slot_id)
                    engine._finish_request(req)
                    engine.queue.mark_done(req.request_id)

            # ── 4. Wait ──
            if not running and engine.queue.waiting_count == 0:
                engine.queue.wait_for_requests(timeout=0.05)

        # Shutdown: notify followers
        coordinator.shutdown()
        engine_logger.info("TP-patched worker loop exited")

    engine._worker_loop = patched_worker


# ── Follower Loop ────────────────────────────────────────────────

def run_follower(model, tp_rank: int, coord_host: str, coord_port: int,
                 max_batch_size: int = 4, max_seq_per_slot: int = 2048):
    """Follower 进程主循环: 连接 rank 0, 等待并执行命令."""

    logger.info(f"Connecting to rank 0 coordinator at {coord_host}:{coord_port} ...")

    max_retries = 600  # 32B 模型首次 AWQ→FP16 转换可能需要 10+ 分钟
    sock = None
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((coord_host, coord_port))
            sock.settimeout(None)
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            sock.close()
            sock = None
            if attempt % 30 == 0:
                logger.info(f"Waiting for coordinator ... attempt {attempt}/{max_retries} ({e})")
            time.sleep(1)

    if sock is None:
        logger.error(f"Failed to connect to coordinator after {max_retries} attempts")
        sys.exit(1)

    # 自报 rank
    send_msg(sock, {"rank": tp_rank})
    logger.info(f"Connected to coordinator. Starting follower loop.")

    # 创建 BatchContext
    batch_ctx = model.create_batch_context(max_batch_size, max_seq_per_slot)
    logger.info(f"BatchContext created (batch_size={max_batch_size}, max_seq={max_seq_per_slot})")

    while True:
        msg = recv_msg(sock)
        if msg is None:
            logger.info("Connection closed by coordinator")
            break

        cmd = msg.get("cmd")

        if cmd == "shutdown":
            logger.info("Received shutdown command")
            break

        elif cmd == "prefill":
            batch_ctx.prefill(
                slot_id=msg["slot_id"],
                token_ids=msg["token_ids"],
                temperature=msg.get("temperature", 0.8),
                top_k=msg.get("top_k", 50),
                top_p=msg.get("top_p", 0.9),
            )
            send_msg(sock, {"status": "ok"})

        elif cmd == "decode":
            batch_ctx.decode(
                active_slots=msg["active_slots"],
                current_tokens=msg["current_tokens"],
                temperature=msg.get("temperature", 0.8),
                top_k=msg.get("top_k", 50),
                top_p=msg.get("top_p", 0.9),
            )
            send_msg(sock, {"status": "ok"})

        elif cmd == "slot_reset":
            batch_ctx.slot_reset(msg["slot_id"])
            send_msg(sock, {"status": "ok"})

        else:
            logger.warning(f"Unknown command: {cmd}")
            send_msg(sock, {"status": "ok"})

    sock.close()
    logger.info("Follower loop exited")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLAISYS TP Worker")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="nvidia")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--coord-port", type=int, default=29500,
                        help="TCP port for TP coordination (rank 0 listens)")
    parser.add_argument("--max-seq-len", type=int, default=2048)

    # Rank 0 only:
    parser.add_argument("--server", action="store_true",
                        help="Run web server (rank 0 only)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # 配置 logging
    logger.addFilter(RankFilter(args.tp_rank))
    for handler in logging.root.handlers:
        handler.addFilter(RankFilter(args.tp_rank))

    logger.info(f"Starting TP worker: tp_size={args.tp_size}, tp_rank={args.tp_rank}")

    # Rank 0: 在加载模型之前先启动 coordinator 监听,
    # 这样 follower 在 rank 0 漫长的模型加载期间就能建立 TCP 连接.
    coordinator = None
    if args.server:
        coordinator = TPCoordinator(args.tp_size, args.coord_port)
        coordinator.start_listening()

    # 加载模型 (每个 rank 加载自己分片的权重)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

    import llaisys
    from llaisys.libllaisys import DeviceType

    if args.device == "nvidia":
        device = DeviceType.NVIDIA
    elif args.device == "metax":
        device = DeviceType.METAX
    else:
        device = DeviceType.CPU

    logger.info(f"Loading model ...")
    model = llaisys.models.Qwen2(
        args.model, device,
        tp_size=args.tp_size, tp_rank=args.tp_rank,
        max_seq_len=args.max_seq_len,
    )
    logger.info(f"Model loaded successfully")

    if args.server:
        # ── Rank 0: Web Server + Coordinator ──
        logger.info("Running as rank 0 (server + coordinator)")

        coordinator.wait_for_followers()

        # 加载 tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True)

        # 创建 InferenceEngine 并 patch
        from server.engine import InferenceEngine
        engine = InferenceEngine(model, tokenizer)
        patch_engine_for_tp(engine, coordinator)
        engine.start()

        # 设置 app 全局变量
        from server import app as app_module
        app_module.MODEL = model
        app_module.TOKENIZER = tokenizer
        app_module.ENGINE = engine
        app_module.SESSION_MGR = __import__("server.session", fromlist=["SessionManager"]).SessionManager(model)

        logger.info(f"Starting web server on {args.host}:{args.port}")

        import uvicorn
        uvicorn.run(app_module.app, host=args.host, port=args.port,
                    log_level="info")
    else:
        # ── Rank 1-N: Follower ──
        logger.info("Running as follower")
        run_follower(
            model=model,
            tp_rank=args.tp_rank,
            coord_host="127.0.0.1",
            coord_port=args.coord_port,
        )


if __name__ == "__main__":
    main()
