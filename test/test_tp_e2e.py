#!/usr/bin/env python3
"""
端到端 TP 推理测试 — 使用真实 NCCL 后端在多 GPU 上运行 Qwen2.5-32B-Instruct-AWQ

测试流程:
  1. tp=1 单卡基线推理（获取参考输出）
  2. tp=2 双卡 NCCL TP 推理（对比结果）

用法:
    PYTHONPATH=python python3 test/test_tp_e2e.py --model /path/to/model --gpus 2,3
"""

import sys
import os
import subprocess
import tempfile
import time
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PYTHON_DIR = os.path.join(PROJECT_DIR, 'python')

# Worker 脚本: 在单个 GPU 上加载模型并推理
WORKER_SCRIPT = r'''
import sys
import os
import time
import json

sys.path.insert(0, os.environ["LLAISYS_PYTHON_DIR"])

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
model_path = os.environ["MODEL_PATH"]
result_file = os.environ["RESULT_FILE"]
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "2048"))
prompt_tokens = json.loads(os.environ["PROMPT_TOKENS"])

results = {"rank": rank, "world_size": world_size}

try:
    from llaisys.models.qwen2 import Qwen2
    from llaisys.libllaisys import DeviceType
    from llaisys.libllaisys.distributed import DistBackend

    t0 = time.time()

    if world_size == 1:
        # 单卡模式
        print(f"[rank {rank}] 创建单卡模型 ...", flush=True)
        model = Qwen2(
            model_path=model_path,
            device=DeviceType.NVIDIA,
            max_seq_len=max_seq_len,
        )
    else:
        # TP 模式
        print(f"[rank {rank}] 创建 TP 模型 (tp_size={world_size}, tp_rank={rank}) ...", flush=True)
        model = Qwen2(
            model_path=model_path,
            device=DeviceType.NVIDIA,
            max_seq_len=max_seq_len,
            tp_size=world_size,
            tp_rank=rank,
            dist_backend=DistBackend.NCCL,
        )

    load_time = time.time() - t0
    print(f"[rank {rank}] 模型加载完成, 耗时 {load_time:.1f}s", flush=True)
    results["load_time_s"] = round(load_time, 1)

    tp_size = getattr(model, 'tp_size', 1)
    tp_rank = getattr(model, 'tp_rank', 0)
    results["tp_size"] = tp_size
    results["tp_rank"] = tp_rank

    # 推理: greedy decode (temperature=0)
    print(f"[rank {rank}] 开始推理 (tokens={prompt_tokens}) ...", flush=True)
    t1 = time.time()

    output_tokens = []
    max_gen = int(os.environ.get("MAX_GEN_TOKENS", "20"))

    # Prefill
    import ctypes
    from llaisys.libllaisys.qwen2 import model_infer, model_infer_sample, model_reset_cache

    # Reset cache
    model_reset_cache(model.model_handle)

    # Greedy: 使用 model_infer (argmax)
    tokens_arr = (ctypes.c_int64 * len(prompt_tokens))(*prompt_tokens)
    out_token = model_infer(model.model_handle, tokens_arr, len(prompt_tokens))
    output_tokens.append(int(out_token))
    print(f"[rank {rank}] prefill → token {out_token}", flush=True)

    # Autoregressive decode
    for step in range(max_gen - 1):
        tok_arr = (ctypes.c_int64 * 1)(output_tokens[-1])
        out_token = model_infer(model.model_handle, tok_arr, 1)
        output_tokens.append(int(out_token))

        # Check EOS
        if hasattr(model, '_end_token') and out_token == model._end_token:
            break

    infer_time = time.time() - t1
    print(f"[rank {rank}] 推理完成, 生成 {len(output_tokens)} tokens, 耗时 {infer_time:.2f}s", flush=True)
    print(f"[rank {rank}] 输出 tokens: {output_tokens}", flush=True)

    results["output_tokens"] = output_tokens
    results["infer_time_s"] = round(infer_time, 2)
    results["status"] = "SUCCESS"

except Exception as e:
    import traceback
    results["status"] = "ERROR"
    results["error"] = str(e)
    results["traceback"] = traceback.format_exc()
    print(f"[rank {rank}] ERROR: {e}", flush=True)
    traceback.print_exc()

with open(result_file, "w") as f:
    json.dump(results, f)
'''


def run_inference(model_path, world_size, gpu_ids, prompt_tokens, max_seq_len=2048, max_gen=20):
    """启动 TP 推理并收集结果"""
    nccl_id_file = os.path.join(tempfile.gettempdir(), f'llaisys_tp_test_{os.getpid()}')
    if os.path.exists(nccl_id_file):
        os.remove(nccl_id_file)

    processes = []
    result_files = {}

    for rank in range(world_size):
        result_file = os.path.join(tempfile.gettempdir(), f'tp_test_rank{rank}_{os.getpid()}.json')
        result_files[rank] = result_file

        env = os.environ.copy()
        env['RANK'] = str(rank)
        env['WORLD_SIZE'] = str(world_size)
        env['MODEL_PATH'] = model_path
        env['RESULT_FILE'] = result_file
        env['MAX_SEQ_LEN'] = str(max_seq_len)
        env['MAX_GEN_TOKENS'] = str(max_gen)
        env['PROMPT_TOKENS'] = json.dumps(prompt_tokens)
        env['LLAISYS_PYTHON_DIR'] = PYTHON_DIR
        env['LLAISYS_NCCL_ID_FILE'] = nccl_id_file
        # TP mode: all workers see all GPUs, device_id=tp_rank selects the right one
        # Single mode: restrict to the one GPU
        if world_size == 1:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        else:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids[:world_size])

        proc = subprocess.Popen(
            [sys.executable, '-c', WORKER_SCRIPT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((rank, proc))

        if rank == 0 and world_size > 1:
            time.sleep(1)

    # 等待完成 (32B model AWQ conversion can take ~150s)
    results = {}
    for rank, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=900)
            print(stdout.decode(), end='')
            stderr_text = stderr.decode()
            # 只打印关键错误信息
            for line in stderr_text.split('\n'):
                if any(k in line.lower() for k in ['error', 'fail', 'traceback', 'assert']):
                    print(f"  [rank {rank} stderr] {line.strip()}")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[rank {rank}] TIMEOUT!")
            results[rank] = {"status": "TIMEOUT"}
            continue

        if os.path.exists(result_files[rank]):
            with open(result_files[rank]) as f:
                results[rank] = json.load(f)
            os.remove(result_files[rank])
        else:
            results[rank] = {"status": "NO_RESULT"}

    # 清理
    if os.path.exists(nccl_id_file):
        os.remove(nccl_id_file)

    return results


def main():
    parser = argparse.ArgumentParser(description="TP E2E Inference Test")
    parser.add_argument('--model', type=str,
                        default='/home/alopbrobro8848/llaisys-cuda-engine/models/qwen2.5-32b-instruct-awq',
                        help='Model path')
    parser.add_argument('--gpus', type=str, default='2,3',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--max-seq-len', type=int, default=2048,
                        help='Max sequence length (controls KV-cache size)')
    parser.add_argument('--max-gen', type=int, default=20,
                        help='Max tokens to generate')
    parser.add_argument('--tp-sizes', type=str, default='1,2',
                        help='Comma-separated TP sizes to test (e.g., "1,2,4,8")')
    parser.add_argument('--baseline-tokens', type=str, default=None,
                        help='Known tp=1 baseline tokens (JSON array) to skip re-running tp=1')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    tp_sizes = [int(x) for x in args.tp_sizes.split(',')]

    # 简单 prompt tokens (数字比较小, 不依赖 tokenizer)
    # 这是 "Hello" 的大致 token IDs (对于 Qwen2 tokenizer)
    prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
    # 这对应: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n

    print("=" * 70)
    print(f"  端到端 TP 推理测试")
    print(f"  模型: {args.model}")
    print(f"  GPU: {gpu_ids}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  prompt tokens: {len(prompt_tokens)}")
    print(f"  TP sizes: {tp_sizes}")
    print("=" * 70)

    # ── 可选: 使用已知基线 ──
    tp1_tokens = None
    if args.baseline_tokens:
        tp1_tokens = json.loads(args.baseline_tokens)
        print(f"\n  使用已知 tp=1 基线: {tp1_tokens}")

    results_summary = {}
    all_success = True

    for tp_size in tp_sizes:
        if tp_size > len(gpu_ids):
            print(f"\n  ⚠️ 跳过 tp={tp_size}: 需要 {tp_size} 张 GPU, 仅有 {len(gpu_ids)} 张")
            continue

        print(f"\n{'─'*70}")
        if tp_size == 1:
            print(f"  [Test tp={tp_size}] 单卡推理 (GPU {gpu_ids[0]})")
        else:
            print(f"  [Test tp={tp_size}] NCCL 推理 (GPUs {gpu_ids[:tp_size]})")
        print(f"{'─'*70}")

        result = run_inference(
            args.model, world_size=tp_size, gpu_ids=gpu_ids[:tp_size],
            prompt_tokens=prompt_tokens,
            max_seq_len=args.max_seq_len,
            max_gen=args.max_gen,
        )

        tp_success = True
        for rank in range(tp_size):
            r = result.get(rank, {})
            if r.get('status') != 'SUCCESS':
                print(f"\n  ❌ tp={tp_size} rank {rank} 推理失败!")
                print(json.dumps(r, indent=2, ensure_ascii=False))
                tp_success = False
                all_success = False

        if tp_success:
            tokens_0 = result[0]['output_tokens']
            time_0 = result[0]['infer_time_s']
            load_0 = result[0]['load_time_s']

            # 打印各 rank 输出
            for rank in range(tp_size):
                print(f"  tp={tp_size} rank {rank} 输出: {result[rank]['output_tokens']}")
            print(f"  tp={tp_size} 推理时间: {time_0}s, 加载时间: {load_0}s")

            # 验证所有 rank 输出一致
            all_ranks_match = all(
                result[rank]['output_tokens'] == tokens_0 for rank in range(1, tp_size)
            )
            if all_ranks_match:
                print(f"\n  ✅ tp={tp_size} 所有 {tp_size} 个 rank 输出一致")
            else:
                print(f"\n  ⚠️ tp={tp_size} rank 间输出不完全一致")

            # 设置或对比基线
            if tp_size == 1:
                tp1_tokens = tokens_0
            if tp1_tokens is not None and tokens_0 == tp1_tokens:
                print(f"  ✅ tp={tp_size} 输出与 tp=1 完全一致!")
            elif tp1_tokens is not None:
                match = sum(1 for a, b in zip(tp1_tokens, tokens_0) if a == b)
                print(f"  ⚠️ tp={tp_size} vs tp=1: {match}/{len(tp1_tokens)} tokens 匹配")
                print(f"      tp=1: {tp1_tokens}")
                print(f"      tp={tp_size}: {tokens_0}")
                print(f"      (浮点精度差异导致的发散是预期行为)")

            results_summary[tp_size] = {
                'tokens': tokens_0, 'time': time_0, 'load': load_0,
                'ranks_match': all_ranks_match,
                'baseline_match': tp1_tokens is not None and tokens_0 == tp1_tokens
            }

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  端到端 TP 推理测试汇总")
    print(f"{'='*70}")
    tp1_time = results_summary.get(1, {}).get('time', 0)
    for tp_size in sorted(results_summary.keys()):
        s = results_summary[tp_size]
        line = f"  tp={tp_size} 加载: {s['load']}s | 推理: {s['time']}s | tokens: {len(s['tokens'])}"
        if tp1_time > 0 and s['time'] > 0 and tp_size > 1:
            speedup = tp1_time / s['time']
            line += f" | 加速比: {speedup:.2f}x"
        match_str = " | 与tp=1一致" if s['baseline_match'] else ""
        print(line + match_str)
    print(f"{'='*70}")

    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
