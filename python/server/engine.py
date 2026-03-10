"""
LLAISYS Inference Engine — 请求队列 + 异步推理服务

Phase 5 (项目#4) 核心组件:
  - InferenceRequest: 单个推理请求
  - SamplingParams: 采样参数
  - RequestQueue: 线程安全的请求池
  - InferenceEngine: 后台 worker 线程, 循环处理请求
    - 阶段1: 逐请求串行处理
    - 阶段3: 连续批处理调度

用法:
    engine = InferenceEngine(model, tokenizer)
    engine.start()
    
    # 提交请求 (非阻塞)
    request = engine.submit(input_ids, params, session_id, stream=True)
    
    # 等待结果
    result = await request.future
    # 或流式读取
    async for token_id in request.stream_tokens():
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Sequence

logger = logging.getLogger("llaisys.engine")


# ── 请求状态 ──────────────────────────────────────────────────────

class RequestStatus(str, Enum):
    WAITING = "waiting"        # 在队列中等待
    PREFILLING = "prefilling"  # 正在 prefill
    DECODING = "decoding"      # 正在 decode (逐 token 生成)
    DONE = "done"              # 完成
    CANCELLED = "cancelled"    # 已取消
    ERROR = "error"            # 出错


# ── 采样参数 ─────────────────────────────────────────────────────

@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 512


# ── 推理请求 ─────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """单个推理请求."""
    request_id: str
    input_ids: List[int]           # tokenize 后的完整输入序列
    params: SamplingParams
    session_id: str
    stream: bool = False

    # 异步通信
    future: asyncio.Future = field(default=None, repr=False)
    output_queue: asyncio.Queue = field(default=None, repr=False)
    loop: asyncio.AbstractEventLoop = field(default=None, repr=False)

    # 生成状态
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: List[int] = field(default_factory=list)
    last_token: int = 0
    kv_cache_snapshot: object = None  # C++ KV-Cache 快照句柄

    # 时间戳
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    # 错误信息
    error_message: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req-{uuid.uuid4().hex[:12]}"

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.DONE, RequestStatus.CANCELLED, RequestStatus.ERROR)

    async def stream_tokens(self) -> AsyncGenerator[int, None]:
        """异步生成器, 逐个 yield token_id."""
        if self.output_queue is None:
            return
        while True:
            token = await self.output_queue.get()
            if token is None:  # 结束信号
                break
            yield token


# ── 请求队列 ─────────────────────────────────────────────────────

class RequestQueue:
    """线程安全的请求池."""

    def __init__(self, max_size: int = 1024):
        self._lock = threading.Lock()
        self._waiting: List[InferenceRequest] = []
        self._active: Dict[str, InferenceRequest] = {}  # request_id → request
        self._max_size = max_size
        self._not_empty = threading.Condition(self._lock)

    def submit(self, request: InferenceRequest) -> bool:
        """提交请求到等待队列. 返回是否成功."""
        with self._lock:
            if len(self._waiting) >= self._max_size:
                return False
            self._waiting.append(request)
            self._not_empty.notify()
            return True

    def get_pending(self, max_count: int = 1) -> List[InferenceRequest]:
        """从等待队列取出至多 max_count 个请求."""
        with self._lock:
            count = min(max_count, len(self._waiting))
            if count == 0:
                return []
            batch = self._waiting[:count]
            self._waiting = self._waiting[count:]
            for req in batch:
                self._active[req.request_id] = req
            return batch

    def wait_for_requests(self, timeout: float = 0.1) -> bool:
        """等待直到有请求可用或超时. 返回是否有请求."""
        with self._not_empty:
            if not self._waiting:
                self._not_empty.wait(timeout)
            return len(self._waiting) > 0

    def mark_done(self, request_id: str):
        """标记请求完成, 从活跃集合中移除."""
        with self._lock:
            self._active.pop(request_id, None)

    def cancel(self, request_id: str) -> bool:
        """取消请求. 返回是否成功."""
        with self._lock:
            # 从等待队列中移除
            for i, req in enumerate(self._waiting):
                if req.request_id == request_id:
                    req.status = RequestStatus.CANCELLED
                    self._waiting.pop(i)
                    return True
            # 标记活跃请求为取消
            req = self._active.get(request_id)
            if req:
                req.status = RequestStatus.CANCELLED
                return True
            return False

    @property
    def waiting_count(self) -> int:
        with self._lock:
            return len(self._waiting)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._waiting) == 0 and len(self._active) == 0


# ── 推理引擎 ─────────────────────────────────────────────────────

class InferenceEngine:
    """
    后台推理引擎 — 连续批处理 (Continuous Batching).
    
    每轮迭代:
      1. 从队列取新请求 → 分配 slot → prefill
      2. 对所有活跃 slot 执行一步批量 decode
      3. 完成的请求移出 batch, 释放 slot
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        max_batch_size: int = 4,
        max_seq_per_slot: int = 2048,
        max_queue_size: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_per_slot = max_seq_per_slot

        self.queue = RequestQueue(max_size=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._eos_token_id = getattr(model, '_end_token', 151643)

        # KV-Cache 前缀树池
        self._cache_pool = None
        try:
            self._cache_pool = model.create_cache_pool()
            logger.info("KV-Cache prefix pool created")
        except Exception:
            logger.warning("KV-Cache prefix pool not available")

        # 统计
        self._total_requests = 0
        self._total_tokens = 0

    def start(self):
        """启动后台 worker 线程."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="inference-engine",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(f"Inference engine started (max_batch_size={self.max_batch_size})")

    def stop(self):
        """停止后台 worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        logger.info("Inference engine stopped")

    def submit(
        self,
        input_ids: List[int],
        params: SamplingParams,
        session_id: str,
        stream: bool = False,
        loop: asyncio.AbstractEventLoop = None,
    ) -> InferenceRequest:
        """提交推理请求 (非阻塞).
        
        Returns:
            InferenceRequest 对象, 可通过 .future 等待结果,
            或通过 .stream_tokens() 流式读取.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

        request = InferenceRequest(
            request_id=f"req-{uuid.uuid4().hex[:12]}",
            input_ids=input_ids,
            params=params,
            session_id=session_id,
            stream=stream,
            future=loop.create_future(),
            output_queue=asyncio.Queue() if stream else None,
            loop=loop,
        )

        if not self.queue.submit(request):
            request.status = RequestStatus.ERROR
            request.error_message = "Request queue is full"
            loop.call_soon_threadsafe(
                request.future.set_exception,
                RuntimeError("Request queue is full"),
            )
        else:
            logger.debug(f"Request {request.request_id} submitted (queue={self.queue.waiting_count})")

        return request

    # ── 连续批处理 Worker 循环 ───────────────────────────────────

    def _worker_loop(self):
        """后台 worker 主循环 — 连续批处理."""
        logger.info("Continuous batching worker loop started")

        # 创建 BatchContext (per-slot KV-Cache on device)
        batch_ctx = self.model.create_batch_context(
            self.max_batch_size, self.max_seq_per_slot
        )

        # slot_id → InferenceRequest 映射
        running: Dict[int, InferenceRequest] = {}
        free_slots: List[int] = list(range(self.max_batch_size))

        while self._running:
            # ── 1. Admit: 从队列取新请求, 分配空闲 slot, 执行 prefill ──
            admitted = 0
            while free_slots and self.queue.waiting_count > 0:
                new_requests = self.queue.get_pending(max_count=1)
                if not new_requests:
                    break
                req = new_requests[0]

                if req.status == RequestStatus.CANCELLED:
                    self.queue.mark_done(req.request_id)
                    continue

                slot_id = free_slots.pop(0)
                req.started_at = time.time()
                req.status = RequestStatus.PREFILLING

                try:
                    # 前缀匹配: 查找可复用的 KV-Cache
                    prefix_snapshot = None
                    match_len = 0
                    if self._cache_pool:
                        try:
                            prefix_snapshot, match_len = self.model.cache_pool_lookup(
                                self._cache_pool, req.input_ids
                            )
                        except Exception:
                            pass

                    # 如果有前缀匹配, 恢复到 slot
                    if prefix_snapshot and match_len > 0:
                        batch_ctx.slot_restore(slot_id, prefix_snapshot)
                        remaining_ids = req.input_ids[match_len:]
                        logger.debug(
                            f"Request {req.request_id}: prefix match "
                            f"{match_len}/{len(req.input_ids)} tokens"
                        )
                    else:
                        batch_ctx.slot_reset(slot_id)
                        remaining_ids = req.input_ids

                    # Prefill
                    if remaining_ids:
                        first_token = batch_ctx.prefill(
                            slot_id=slot_id,
                            token_ids=remaining_ids,
                            temperature=req.params.temperature,
                            top_k=req.params.top_k,
                            top_p=req.params.top_p,
                        )
                    else:
                        # 完全匹配, 用最后一个 token 做一步 decode
                        first_token = batch_ctx.prefill(
                            slot_id=slot_id,
                            token_ids=[req.input_ids[-1]],
                            temperature=req.params.temperature,
                            top_k=req.params.top_k,
                            top_p=req.params.top_p,
                        )

                    first_token = int(first_token)

                except Exception as e:
                    logger.error(f"Prefill error for {req.request_id}: {e}", exc_info=True)
                    free_slots.append(slot_id)
                    self._finish_request(req, error=e)
                    self.queue.mark_done(req.request_id)
                    continue

                req.generated_tokens.append(first_token)
                req.last_token = first_token
                req.status = RequestStatus.DECODING

                # 流式输出首个 token
                if req.stream and req.output_queue:
                    self._send_token_async(req, first_token)

                # 检查是否已完成
                if (first_token == self._eos_token_id or
                        len(req.generated_tokens) >= req.params.max_tokens):
                    free_slots.append(slot_id)
                    batch_ctx.slot_reset(slot_id)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)
                else:
                    running[slot_id] = req
                    admitted += 1

            # ── 2. Decode: 对所有活跃 slot 执行一步批量 decode ──
            if running:
                active_slots = list(running.keys())
                current_tokens = [running[s].last_token for s in active_slots]

                # 使用第一个请求的采样参数 (批量 decode 共享参数)
                # 注: 不同请求可能有不同参数, 这里用首个请求的参数作为 batch 参数
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
                    logger.error(f"Batch decode error: {e}", exc_info=True)
                    # 所有活跃请求报错
                    for sid in list(running.keys()):
                        req = running.pop(sid)
                        free_slots.append(sid)
                        batch_ctx.slot_reset(sid)
                        self._finish_request(req, error=e)
                        self.queue.mark_done(req.request_id)
                    continue

                # ── 3. 处理结果, 移除已完成的请求 ──
                finished_slots: List[int] = []
                for slot_id, next_tok in zip(active_slots, next_tokens):
                    req = running[slot_id]
                    next_tok = int(next_tok)
                    req.generated_tokens.append(next_tok)
                    req.last_token = next_tok

                    # 流式输出
                    if req.stream and req.output_queue:
                        self._send_token_async(req, next_tok)

                    # 检查终止条件
                    if (next_tok == self._eos_token_id or
                            len(req.generated_tokens) >= req.params.max_tokens or
                            req.status == RequestStatus.CANCELLED):
                        finished_slots.append(slot_id)

                # 清理已完成的请求
                for slot_id in finished_slots:
                    req = running.pop(slot_id)

                    # 保存到前缀树池
                    if self._cache_pool:
                        try:
                            snapshot = batch_ctx.slot_save(slot_id)
                            if snapshot:
                                all_tokens = req.input_ids + req.generated_tokens
                                self.model.cache_pool_insert(
                                    self._cache_pool, all_tokens, snapshot
                                )
                        except Exception:
                            pass

                    batch_ctx.slot_reset(slot_id)
                    free_slots.append(slot_id)
                    self._finish_request(req)
                    self.queue.mark_done(req.request_id)

            # ── 4. 空闲等待 ──
            if not running and self.queue.waiting_count == 0:
                self.queue.wait_for_requests(timeout=0.05)

        logger.info("Worker loop exited")

    def _send_token_async(self, req: InferenceRequest, token_id: int):
        """线程安全地向 asyncio 队列发送 token."""
        if req.loop and req.output_queue:
            req.loop.call_soon_threadsafe(req.output_queue.put_nowait, token_id)

    def _finish_request(self, req: InferenceRequest, error: Exception = None):
        """标记请求完成, 设置 future 结果."""
        req.finished_at = time.time()

        if error:
            req.status = RequestStatus.ERROR
            req.error_message = str(error)
            if req.loop and req.future and not req.future.done():
                req.loop.call_soon_threadsafe(req.future.set_exception, error)
        else:
            req.status = RequestStatus.DONE
            if req.loop and req.future and not req.future.done():
                req.loop.call_soon_threadsafe(
                    req.future.set_result, req.generated_tokens
                )

        # 流式结束信号
        if req.stream and req.output_queue and req.loop:
            req.loop.call_soon_threadsafe(req.output_queue.put_nowait, None)

        elapsed = req.finished_at - req.started_at if req.started_at > 0 else 0
        n_tokens = len(req.generated_tokens)
        tps = n_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Request {req.request_id} finished: "
            f"{n_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s) "
            f"status={req.status.value}"
        )

        self._total_requests += 1
        self._total_tokens += n_tokens

    # ── 状态查询 ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回引擎状态统计."""
        return {
            "running": self._running,
            "waiting_requests": self.queue.waiting_count,
            "active_requests": self.queue.active_count,
            "max_batch_size": self.max_batch_size,
            "total_requests_served": self._total_requests,
            "total_tokens_generated": self._total_tokens,
        }
