#!/usr/bin/env python3
"""
Phase 4 集成测试 — 多用户并发推理.

用法:
    # 先启动服务 (另一个终端):
    cd python && python -m server.app --model ../quantized_model --device cpu

    # 然后运行测试:
    python -m test.test_concurrent

测试内容:
    1. 单请求基本功能
    2. 多请求并发 (非流式)
    3. 多请求并发 (流式)
    4. 引擎状态查询
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import List, Optional

import aiohttp

BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_TOKENS = 32


# ── 工具函数 ─────────────────────────────────────────────────────

async def chat_completion(
    session: aiohttp.ClientSession,
    messages: List[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    temperature: float = 0.7,
    session_id: Optional[str] = None,
    label: str = "",
) -> dict:
    """发送 chat completion 请求并返回结果."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if session_id:
        payload["session_id"] = session_id

    t0 = time.time()

    if stream:
        # SSE 流式
        tokens_text = []
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": text, "status": resp.status, "label": label}

            # 逐行读取 SSE 流
            buffer = b""
            while True:
                chunk = await resp.content.readany()
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data)
                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                tokens_text.append(content)
                        except json.JSONDecodeError:
                            pass
                else:
                    continue
                break  # [DONE] received

        elapsed = time.time() - t0
        full_text = "".join(tokens_text)
        return {
            "text": full_text,
            "n_chunks": len(tokens_text),
            "elapsed": elapsed,
            "label": label,
            "stream": True,
        }
    else:
        # 非流式
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
        ) as resp:
            elapsed = time.time() - t0
            if resp.status != 200:
                text = await resp.text()
                return {"error": text, "status": resp.status, "label": label}
            result = await resp.json()
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            return {
                "text": text,
                "usage": usage,
                "elapsed": elapsed,
                "label": label,
                "stream": False,
            }


async def get_engine_stats(session: aiohttp.ClientSession) -> dict:
    """查询引擎状态."""
    async with session.get(f"{BASE_URL}/v1/engine/stats") as resp:
        return await resp.json()


# ── 测试用例 ─────────────────────────────────────────────────────

async def test_single_request():
    """测试1: 单请求基本功能."""
    print("\n" + "=" * 60)
    print("TEST 1: Single request (non-streaming)")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        result = await chat_completion(
            session,
            messages=[{"role": "user", "content": "Hello, what is 2+2?"}],
            max_tokens=DEFAULT_MAX_TOKENS,
            label="single",
        )

        if "error" in result:
            print(f"  FAIL: {result['error']}")
            return False

        print(f"  Response: {result['text'][:100]}...")
        print(f"  Time: {result['elapsed']:.2f}s")
        print(f"  Usage: {result.get('usage', {})}")
        print("  PASS")
        return True


async def test_concurrent_non_stream():
    """测试2: 多请求并发 (非流式)."""
    print("\n" + "=" * 60)
    print("TEST 2: Concurrent requests (non-streaming)")
    print("=" * 60)

    prompts = [
        "What is the capital of France?",
        "Explain what a neural network is in one sentence.",
        "Write a haiku about programming.",
        "What is 7 * 8?",
    ]

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            task = chat_completion(
                session,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                label=f"req-{i}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    total_time = time.time() - t0

    all_ok = True
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAIL - {r['error']}")
            all_ok = False
        else:
            print(f"  {r['label']}: {r['text'][:60]}... ({r['elapsed']:.2f}s)")

    print(f"\n  Total wall time: {total_time:.2f}s")
    sum_individual = sum(r.get("elapsed", 0) for r in results if "error" not in r)
    print(f"  Sum of individual times: {sum_individual:.2f}s")
    if sum_individual > 0:
        speedup = sum_individual / total_time
        print(f"  Concurrency speedup: {speedup:.2f}x")
    print(f"  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


async def test_concurrent_stream():
    """测试3: 多请求并发 (流式)."""
    print("\n" + "=" * 60)
    print("TEST 3: Concurrent requests (streaming)")
    print("=" * 60)

    prompts = [
        "Count from 1 to 5.",
        "Say hello in three languages.",
        "What color is the sky?",
    ]

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            task = chat_completion(
                session,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=True,
                label=f"stream-{i}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    total_time = time.time() - t0

    all_ok = True
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAIL - {r['error']}")
            all_ok = False
        else:
            print(f"  {r['label']}: {r['n_chunks']} chunks, "
                  f"\"{r['text'][:50]}...\" ({r['elapsed']:.2f}s)")

    print(f"\n  Total wall time: {total_time:.2f}s")
    print(f"  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


async def test_engine_stats():
    """测试4: 引擎状态查询."""
    print("\n" + "=" * 60)
    print("TEST 4: Engine stats")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        stats = await get_engine_stats(session)
        print(f"  Stats: {json.dumps(stats, indent=2)}")
        if stats.get("running"):
            print("  PASS")
            return True
        else:
            print("  FAIL: engine not running")
            return False


async def test_stress_batch():
    """测试5: 压力测试 — 超过 max_batch_size 的并发请求."""
    print("\n" + "=" * 60)
    print("TEST 5: Stress test (many concurrent requests)")
    print("=" * 60)

    n_requests = 12
    prompts = [f"What is {i} + {i*2}?" for i in range(1, n_requests + 1)]

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            task = chat_completion(
                session,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                label=f"stress-{i}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    total_time = time.time() - t0

    successes = sum(1 for r in results if "error" not in r)
    failures = sum(1 for r in results if "error" in r)

    print(f"  Requests: {n_requests}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  Total wall time: {total_time:.2f}s")

    for r in results:
        if "error" in r:
            print(f"  {r['label']}: ERROR {r.get('status', '?')}")
        else:
            print(f"  {r['label']}: \"{r['text'][:40]}...\" ({r['elapsed']:.2f}s)")

    all_ok = successes == n_requests
    print(f"  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── Main ─────────────────────────────────────────────────────────

async def main():
    """运行所有测试."""
    print("=" * 60)
    print("LLAISYS Concurrent Inference Tests")
    print(f"Server: {BASE_URL}")
    print("=" * 60)

    # 检查服务器是否可达
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"Server not healthy: {resp.status}")
                    sys.exit(1)
                print("Server is healthy\n")
    except Exception as e:
        print(f"Cannot connect to server at {BASE_URL}: {e}")
        print("Please start the server first:")
        print("  cd python && python -m server.app --model ../quantized_model --device cpu")
        sys.exit(1)

    results = {}

    # 按顺序运行每个测试
    results["single"] = await test_single_request()
    results["concurrent_non_stream"] = await test_concurrent_non_stream()
    results["concurrent_stream"] = await test_concurrent_stream()
    results["engine_stats"] = await test_engine_stats()
    results["stress"] = await test_stress_batch()

    # 汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
