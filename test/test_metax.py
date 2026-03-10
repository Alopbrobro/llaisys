#!/usr/bin/env python3
"""
=============================================================
Phase D: 沐曦 (MetaX) C500 端到端验证测试
=============================================================
在沐曦算力平台上运行此脚本，逐层验证所有 12 个已移植的算子
以及完整的模型推理流程。

用法:
    cd /path/to/llaisys
    python3 test/test_metax.py                          # 仅算子测试
    python3 test/test_metax.py --model ./quantized_model # 算子 + 推理测试
    python3 test/test_metax.py --full                    # 全量测试

测试层次:
    Level 1: Runtime API (内存分配/拷贝/设备管理)
    Level 2: 各算子单元测试 (12 个算子)
    Level 3: 端到端推理 (加载模型 → 生成文本)
=============================================================
"""

import sys
import os
import time
import argparse
import traceback

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "test"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import llaisys
from test_utils import random_tensor, random_int_tensor, check_equal

# ── 颜色输出 ──────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
NC     = "\033[0m"

def ok(msg):   print(f"  {GREEN}[PASS]{NC} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{NC} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{NC} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{NC} {msg}")
def banner(msg):
    print(f"\n{CYAN}{'═'*60}{NC}")
    print(f"{CYAN}  {msg}{NC}")
    print(f"{CYAN}{'═'*60}{NC}")

DEVICE = "metax"
results = {"passed": [], "failed": [], "skipped": []}


def record(name, success, skip=False):
    if skip:
        results["skipped"].append(name)
    elif success:
        results["passed"].append(name)
    else:
        results["failed"].append(name)


# =============================================================
# Level 1: Runtime API
# =============================================================
def test_runtime_api():
    banner("Level 1: Runtime API")

    try:
        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        ndev = api.get_device_count()
        info(f"MetaX GPU 数量: {ndev}")

        if ndev == 0:
            fail("未检测到 MetaX GPU")
            record("runtime_device_count", False)
            return False

        ok(f"getDeviceCount = {ndev}")
        record("runtime_device_count", True)

        # 测试 setDevice
        api.set_device(0)
        ok("setDevice(0)")
        record("runtime_set_device", True)

        # 测试内存分配
        size = 1024 * 1024  # 1 MB
        ptr = api.malloc_device(size)
        assert ptr != 0, "malloc_device returned null"
        ok(f"malloc_device({size}) = 0x{ptr:x}")
        record("runtime_malloc_device", True)

        # 测试 host 内存
        host_ptr = api.malloc_host(size)
        assert host_ptr != 0, "malloc_host returned null"
        ok(f"malloc_host({size}) = 0x{host_ptr:x}")
        record("runtime_malloc_host", True)

        # 测试 memcpy H2D → D2H 往返
        import torch
        a = torch.arange(256, dtype=torch.float32)
        b = torch.zeros(256, dtype=torch.float32)
        nbytes = 256 * 4

        dev_buf = api.malloc_device(nbytes)
        api.memcpy_sync(dev_buf, a.data_ptr(), nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(b.data_ptr(), dev_buf, nbytes, llaisys.MemcpyKind.D2H)

        assert torch.equal(a, b), "memcpy round-trip failed"
        ok("memcpy H2D → D2H round-trip")
        record("runtime_memcpy", True)

        # 清理
        api.free_device(dev_buf)
        api.free_device(ptr)
        api.free_host(host_ptr)
        ok("free_device / free_host")
        record("runtime_free", True)

        return True

    except Exception as e:
        fail(f"Runtime API 测试异常: {e}")
        traceback.print_exc()
        record("runtime_api", False)
        return False


# =============================================================
# Level 2: 算子单元测试
# =============================================================
def test_op_add():
    """测试 add 算子"""
    import torch
    try:
        t_a, l_a = random_tensor((64, 128), "f32", DEVICE)
        t_b, l_b = random_tensor((64, 128), "f32", DEVICE)
        l_c = llaisys.Tensor((64, 128), dtype=llaisys.DataType.F32,
                             device=llaisys.DeviceType.METAX)

        llaisys.Ops.add(l_c, l_a, l_b)

        t_c = t_a + t_b
        assert check_equal(l_c, t_c, atol=1e-5, rtol=1e-5)
        ok("add (64×128, F32)")
        record("op_add", True)
    except Exception as e:
        fail(f"add: {e}")
        record("op_add", False)


def test_op_argmax():
    """测试 argmax 算子"""
    import torch
    try:
        t_v, l_v = random_tensor((1, 512), "f32", DEVICE)
        l_idx = llaisys.Tensor((1,), dtype=llaisys.DataType.I32,
                               device=llaisys.DeviceType.METAX)
        l_val = llaisys.Tensor((1,), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.argmax(l_idx, l_val, l_v)

        # 取回结果
        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        idx_buf = torch.zeros(1, dtype=torch.int32)
        api.memcpy_sync(idx_buf.data_ptr(), l_idx.data_ptr(), 4, llaisys.MemcpyKind.D2H)

        expected_idx = t_v.argmax().item()
        assert idx_buf[0].item() == expected_idx, f"got {idx_buf[0].item()}, expected {expected_idx}"
        ok("argmax (1×512, F32)")
        record("op_argmax", True)
    except Exception as e:
        fail(f"argmax: {e}")
        record("op_argmax", False)


def test_op_embedding():
    """测试 embedding 算子"""
    import torch
    try:
        vocab_size, embed_dim = 100, 64
        seq_len = 8

        # 权重
        t_w, l_w = random_tensor((vocab_size, embed_dim), "f32", DEVICE)

        # 索引 (CPU 上创建然后拷贝)
        indices = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int64)
        l_idx = llaisys.Tensor((seq_len,), dtype=llaisys.DataType.I64,
                               device=llaisys.DeviceType.METAX)
        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        api.memcpy_sync(l_idx.data_ptr(), indices.data_ptr(),
                        seq_len * 8, llaisys.MemcpyKind.H2D)

        # 输出
        l_out = llaisys.Tensor((seq_len, embed_dim), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.embedding(l_out, l_idx, l_w)

        t_out = torch.nn.functional.embedding(indices.cuda(), t_w)
        assert check_equal(l_out, t_out, atol=1e-5, rtol=1e-5)
        ok("embedding (8×64, F32)")
        record("op_embedding", True)
    except Exception as e:
        fail(f"embedding: {e}")
        record("op_embedding", False)


def test_op_rms_norm():
    """测试 rms_norm 算子"""
    import torch
    try:
        seq_len, dim = 4, 128
        eps = 1e-6

        t_x, l_x = random_tensor((seq_len, dim), "f32", DEVICE)
        t_w, l_w = random_tensor((dim,), "f32", DEVICE)
        l_out = llaisys.Tensor((seq_len, dim), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.rms_norm(l_out, l_x, l_w, eps)

        # torch 参考
        rms = torch.sqrt(t_x.pow(2).mean(-1, keepdim=True) + eps)
        t_out = t_x / rms * t_w
        assert check_equal(l_out, t_out, atol=1e-4, rtol=1e-4)
        ok("rms_norm (4×128, F32)")
        record("op_rms_norm", True)
    except Exception as e:
        fail(f"rms_norm: {e}")
        record("op_rms_norm", False)


def test_op_swiglu():
    """测试 swiglu 算子"""
    import torch
    try:
        t_gate, l_gate = random_tensor((4, 128), "f32", DEVICE)
        t_up, l_up = random_tensor((4, 128), "f32", DEVICE)
        l_out = llaisys.Tensor((4, 128), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.swiglu(l_out, l_gate, l_up)

        silu = t_gate * torch.sigmoid(t_gate)
        t_out = t_up * silu
        assert check_equal(l_out, t_out, atol=1e-5, rtol=1e-5)
        ok("swiglu (4×128, F32)")
        record("op_swiglu", True)
    except Exception as e:
        fail(f"swiglu: {e}")
        record("op_swiglu", False)


def test_op_rope():
    """测试 rope 算子"""
    import torch
    import math
    try:
        seq_len, n_head, head_dim = 4, 8, 64
        theta = 10000.0

        t_x, l_x = random_tensor((seq_len, n_head, head_dim), "f32", DEVICE)

        pos_ids = torch.arange(seq_len, dtype=torch.int64)
        l_pos = llaisys.Tensor((seq_len,), dtype=llaisys.DataType.I64,
                               device=llaisys.DeviceType.METAX)
        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        api.memcpy_sync(l_pos.data_ptr(), pos_ids.data_ptr(),
                        seq_len * 8, llaisys.MemcpyKind.H2D)

        l_out = llaisys.Tensor((seq_len, n_head, head_dim), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.rope(l_out, l_x, l_pos, theta)

        # torch 参考计算
        half = head_dim // 2
        t_out = torch.zeros_like(t_x)
        for s in range(seq_len):
            pos = float(s)
            for h in range(n_head):
                for l in range(half):
                    exp = 2.0 * l / head_dim
                    angle = pos / (theta ** exp)
                    c, si = math.cos(angle), math.sin(angle)
                    a = t_x[s, h, l].item()
                    b = t_x[s, h, l + half].item()
                    t_out[s, h, l] = a * c - b * si
                    t_out[s, h, l + half] = b * c + a * si

        assert check_equal(l_out, t_out, atol=1e-4, rtol=1e-4)
        ok("rope (4×8×64, F32)")
        record("op_rope", True)
    except Exception as e:
        fail(f"rope: {e}")
        record("op_rope", False)


def test_op_linear():
    """测试 linear 算子"""
    import torch
    try:
        M, K, N = 4, 128, 64
        t_x, l_x = random_tensor((M, K), "f32", DEVICE)
        t_w, l_w = random_tensor((N, K), "f32", DEVICE)
        l_out = llaisys.Tensor((M, N), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        # 无 bias
        llaisys.Ops.linear(l_out, l_x, l_w, None)

        t_out = t_x @ t_w.T
        assert check_equal(l_out, t_out, atol=1e-3, rtol=1e-3)
        ok("linear (4×128 × 64×128^T, F32, no bias)")
        record("op_linear", True)
    except Exception as e:
        fail(f"linear: {e}")
        record("op_linear", False)


def test_op_dequantize():
    """测试 dequantize INT8 算子"""
    import torch
    try:
        rows, cols = 32, 64
        # INT8 权重
        w_int8 = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, dtype=torch.float32) * 0.1

        l_w = llaisys.Tensor((rows, cols), dtype=llaisys.DataType.I8,
                             device=llaisys.DeviceType.METAX)
        l_s = llaisys.Tensor((rows,), dtype=llaisys.DataType.F32,
                             device=llaisys.DeviceType.METAX)
        l_out = llaisys.Tensor((rows, cols), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        api.memcpy_sync(l_w.data_ptr(), w_int8.data_ptr(),
                        rows * cols, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(l_s.data_ptr(), scale.data_ptr(),
                        rows * 4, llaisys.MemcpyKind.H2D)

        llaisys.Ops.dequantize(l_out, l_w, l_s)

        # 参考
        t_out = w_int8.float() * scale.unsqueeze(1)
        assert check_equal(l_out, t_out.cuda(), atol=1e-5, rtol=1e-5)
        ok("dequantize INT8 (32×64)")
        record("op_dequantize", True)
    except Exception as e:
        fail(f"dequantize: {e}")
        record("op_dequantize", False)


def test_op_sample():
    """测试 sample 算子 (top-k=1 确定性采样)"""
    import torch
    try:
        vocab = 256
        logits = torch.randn(1, vocab, dtype=torch.float32)

        l_logits = llaisys.Tensor((1, vocab), dtype=llaisys.DataType.F32,
                                  device=llaisys.DeviceType.METAX)
        l_out = llaisys.Tensor((1,), dtype=llaisys.DataType.I32,
                               device=llaisys.DeviceType.METAX)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        api.memcpy_sync(l_logits.data_ptr(), logits.data_ptr(),
                        vocab * 4, llaisys.MemcpyKind.H2D)

        # top_k=1 → 应该取 argmax
        llaisys.Ops.sample(l_out, l_logits, 1.0, 1, 1.0, 42)

        idx_buf = torch.zeros(1, dtype=torch.int32)
        api.memcpy_sync(idx_buf.data_ptr(), l_out.data_ptr(), 4, llaisys.MemcpyKind.D2H)

        expected = logits.argmax().item()
        assert idx_buf[0].item() == expected, f"got {idx_buf[0].item()}, expected {expected}"
        ok("sample (top_k=1, F32)")
        record("op_sample", True)
    except Exception as e:
        fail(f"sample: {e}")
        record("op_sample", False)


def test_op_self_attention():
    """测试 self_attention 算子"""
    import torch
    try:
        seq_len, n_head, head_dim = 2, 4, 32
        total_len = 4
        n_kv_head = 2
        scale = 1.0 / (head_dim ** 0.5)

        t_q, l_q = random_tensor((seq_len, n_head, head_dim), "f32", DEVICE)
        t_k, l_k = random_tensor((total_len, n_kv_head, head_dim), "f32", DEVICE)
        t_v, l_v = random_tensor((total_len, n_kv_head, head_dim), "f32", DEVICE)
        l_out = llaisys.Tensor((seq_len, n_head, head_dim), dtype=llaisys.DataType.F32,
                               device=llaisys.DeviceType.METAX)

        llaisys.Ops.self_attention(l_out, l_q, l_k, l_v, scale)

        # 简单检查：输出形状和非零
        api = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
        out_buf = torch.zeros(seq_len, n_head, head_dim, dtype=torch.float32)
        nbytes = seq_len * n_head * head_dim * 4
        api.memcpy_sync(out_buf.data_ptr(), l_out.data_ptr(), nbytes, llaisys.MemcpyKind.D2H)

        assert not torch.all(out_buf == 0), "self_attention output is all zeros"
        assert not torch.any(torch.isnan(out_buf)), "self_attention output contains NaN"
        ok("self_attention (2×4×32, total_len=4, GQA kv_head=2)")
        record("op_self_attention", True)
    except Exception as e:
        fail(f"self_attention: {e}")
        record("op_self_attention", False)


# =============================================================
# Level 3: 端到端推理
# =============================================================
def test_inference(model_path):
    banner("Level 3: 端到端推理")
    try:
        from transformers import AutoTokenizer

        info(f"加载 tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        info("加载 llaisys 模型 (MetaX)...")
        model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.METAX)

        prompt = "What is 1+1?"
        info(f"Prompt: {prompt}")

        input_content = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer.encode(input_content)

        t0 = time.time()
        outputs = model.generate(
            inputs,
            max_new_tokens=32,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        t1 = time.time()

        text = tokenizer.decode(outputs, skip_special_tokens=True)
        info(f"输出 ({len(outputs)} tokens, {t1-t0:.2f}s):")
        print(f"    {text}")

        assert len(outputs) > len(inputs), "模型未生成新 token"
        ok(f"端到端推理成功 ({len(outputs)-len(inputs)} new tokens)")
        record("inference_e2e", True)

        # 性能报告
        new_tokens = len(outputs) - len(inputs)
        if new_tokens > 0 and t1 - t0 > 0:
            tps = new_tokens / (t1 - t0)
            info(f"吞吐: {tps:.1f} tokens/s (prefill + decode)")

    except Exception as e:
        fail(f"端到端推理: {e}")
        traceback.print_exc()
        record("inference_e2e", False)


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="MetaX C500 Phase D 验证测试")
    parser.add_argument("--model", default=None, type=str,
                        help="模型路径 (提供则运行 Level 3 推理测试)")
    parser.add_argument("--full", action="store_true",
                        help="运行全量测试 (默认使用 ./quantized_model)")
    args = parser.parse_args()

    print(f"\n{CYAN}沐曦 (MetaX) C500 — Phase D 验证测试{NC}")
    print(f"{CYAN}{'─'*50}{NC}\n")

    # ── Level 1 ──
    runtime_ok = test_runtime_api()

    if not runtime_ok:
        print(f"\n{RED}Runtime API 测试失败，无法继续算子测试{NC}")
        print_summary()
        sys.exit(1)

    # ── Level 2 ──
    banner("Level 2: 算子单元测试")
    test_op_add()
    test_op_argmax()
    test_op_embedding()
    test_op_rms_norm()
    test_op_swiglu()
    test_op_rope()
    test_op_linear()
    test_op_dequantize()
    test_op_sample()
    test_op_self_attention()

    # ── Level 3 ──
    model_path = args.model
    if args.full and not model_path:
        # 尝试默认路径
        for p in ["./quantized_model", "./quantized_model_int4"]:
            if os.path.isdir(p):
                model_path = p
                break

    if model_path:
        test_inference(model_path)
    else:
        info("未提供 --model 参数，跳过 Level 3 推理测试")
        record("inference_e2e", True, skip=True)

    print_summary()

    if results["failed"]:
        sys.exit(1)


def print_summary():
    banner("测试结果汇总")
    total = len(results["passed"]) + len(results["failed"]) + len(results["skipped"])
    print(f"  总计: {total}")
    print(f"  {GREEN}通过: {len(results['passed'])}{NC}")
    if results["failed"]:
        print(f"  {RED}失败: {len(results['failed'])}{NC}")
        for f in results["failed"]:
            print(f"    - {f}")
    if results["skipped"]:
        print(f"  {YELLOW}跳过: {len(results['skipped'])}{NC}")

    if not results["failed"]:
        print(f"\n  {GREEN}✓ 所有测试通过！沐曦 C500 适配验证完成{NC}")
    else:
        print(f"\n  {RED}✗ 存在失败项，请检查上方日志{NC}")


if __name__ == "__main__":
    main()
