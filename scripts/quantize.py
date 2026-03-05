#!/usr/bin/env python3
"""Weight-Only 对称量化工具 (INT8 / INT4)

用法:
    # INT8 per-channel 量化 (~2x 压缩)
    python scripts/quantize.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output ./quantized_model

    # INT4 per-group 量化 (~4x 压缩)
    python scripts/quantize.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output ./quantized_model_int4 --bits 4 --group-size 128

原理:
    INT8 per-channel:
        scale = max(|W_row|) / 127
        W_int8 = round(W / scale).clamp(-128, 127)
        保存: W_int8 (int8, [out, in]) + scale (float32, [out])

    INT4 per-group:
        对每组 group_size 个连续元素:
        scale = max(|group|) / 7
        W_int4 = round(W / scale).clamp(-8, 7)
        打包: 两个 INT4 值存入一个 uint8 字节
        byte = ((val1 + 8) << 4) | ((val0 + 8) & 0x0F)
        保存: W_packed (uint8, [out, in/2]) + scale (float32, [out, in/group_size])
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import safetensors
import safetensors.torch
from huggingface_hub import snapshot_download


# 需要量化的权重后缀 (都是 Linear 层的权重矩阵)
QUANTIZE_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "lm_head.weight",
]

# 不量化的权重 (embedding, norm, bias 等)
SKIP_SUFFIXES = [
    "embed_tokens.weight",
    "layernorm.weight",
    "norm.weight",
    ".bias",
]


def should_quantize(name: str) -> bool:
    """判断是否对该权重做 INT8 量化"""
    for s in QUANTIZE_SUFFIXES:
        if name.endswith(s):
            return True
    return False


def quantize_per_channel_symmetric(weight: torch.Tensor):
    """Per-channel 对称量化
    
    Args:
        weight: FP32 / FP16 tensor, shape [out_features, in_features]
    
    Returns:
        w_int8: INT8 tensor, shape [out_features, in_features]
        scale: FP32 tensor, shape [out_features]
    """
    w = weight.float()   # 确保 FP32 计算
    
    # Per-channel: 每一行的最大绝对值
    row_max = w.abs().amax(dim=-1)  # shape [out_features]
    
    # scale = max(|w|) / 127, 防止除零
    scale = row_max / 127.0
    scale = scale.clamp(min=1e-10)
    
    # 量化
    w_scaled = w / scale.unsqueeze(-1)
    w_int8 = w_scaled.round().clamp(-128, 127).to(torch.int8)
    
    return w_int8, scale.float()


def quantize_per_group_symmetric_int4(weight: torch.Tensor, group_size: int = 128):
    """Per-group 对称 INT4 量化, 两个 INT4 值打包进一个 uint8

    Args:
        weight: FP32 / FP16 tensor, shape [out_features, in_features]
        group_size: 每组元素个数 (必须能整除 in_features, 且为偶数)

    Returns:
        packed: uint8 tensor, shape [out_features, in_features // 2]
        scale:  FP32 tensor, shape [out_features, num_groups]
    """
    w = weight.float()                     # [rows, cols]
    rows, cols = w.shape

    assert cols % group_size == 0, f"in_features {cols} 不能被 group_size {group_size} 整除"
    assert group_size % 2 == 0, f"group_size {group_size} 必须为偶数"
    num_groups = cols // group_size

    # Reshape 为 [rows, num_groups, group_size]
    w_grouped = w.reshape(rows, num_groups, group_size)

    # Per-group: 每组最大绝对值
    group_max = w_grouped.abs().amax(dim=-1)    # [rows, num_groups]

    # scale = max(|group|) / 7, 防止除零
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)              # [rows, num_groups]

    # 量化到 [-8, 7]
    w_scaled = w_grouped / scale.unsqueeze(-1)  # [rows, num_groups, group_size]
    w_int4 = w_scaled.round().clamp(-8, 7).to(torch.int8)  # 用 int8 暂存

    # 展平回 [rows, cols]
    w_int4 = w_int4.reshape(rows, cols)

    # 打包: 两个 INT4 值存入一个 uint8
    # byte = ((val[2k+1] + 8) << 4) | ((val[2k] + 8) & 0x0F)
    # 偶数列 → 低 nibble, 奇数列 → 高 nibble
    w_even = (w_int4[:, 0::2] + 8).to(torch.uint8)  # [rows, cols/2], 范围 [0, 15]
    w_odd  = (w_int4[:, 1::2] + 8).to(torch.uint8)  # [rows, cols/2], 范围 [0, 15]
    packed = (w_odd << 4) | (w_even & 0x0F)          # [rows, cols/2], uint8

    return packed, scale.float()


def quantize_model(model_path: str, output_dir: str, bits: int = 8, group_size: int = 128):
    """读取 safetensors 模型并量化 (INT8 per-channel 或 INT4 per-group)"""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert bits in (4, 8), f"bits 必须为 4 或 8, 当前: {bits}"
    if bits == 4:
        print(f"==> INT4 per-group 量化, group_size={group_size}")
    else:
        print(f"==> INT8 per-channel 量化")
    
    # 如果是 HuggingFace model ID, 先下载
    if not model_path.exists():
        print(f"Downloading model {model_path} ...")
        model_path = Path(snapshot_download(str(model_path)))
    
    # 收集 safetensors 文件
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No .safetensors found in {model_path}", file=sys.stderr)
        sys.exit(1)
    
    quantized_tensors = {}
    stats = {"quantized": 0, "skipped": 0, "total_bytes_fp32": 0, "total_bytes_quant": 0}
    
    for file in st_files:
        print(f"\nProcessing: {file.name}")
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                
                if should_quantize(name):
                    orig_bytes = tensor.numel() * tensor.element_size()

                    if bits == 4:
                        # INT4 per-group
                        packed, scale = quantize_per_group_symmetric_int4(tensor, group_size)
                        quantized_tensors[name] = packed      # uint8, [rows, cols/2]
                        quantized_tensors[name + ".scale"] = scale  # float32, [rows, num_groups]
                        quant_bytes = packed.numel() * 1 + scale.numel() * 4
                        label = "int4"
                    else:
                        # INT8 per-channel
                        w_int8, scale = quantize_per_channel_symmetric(tensor)
                        quantized_tensors[name] = w_int8      # int8, [rows, cols]
                        quantized_tensors[name + ".scale"] = scale  # float32, [rows]
                        quant_bytes = w_int8.numel() * 1 + scale.numel() * 4
                        label = "int8"
                    
                    ratio = orig_bytes / quant_bytes
                    stats["quantized"] += 1
                    stats["total_bytes_fp32"] += orig_bytes
                    stats["total_bytes_quant"] += quant_bytes
                    
                    print(f"  [QUANT] {name}: {list(tensor.shape)} "
                          f"{tensor.dtype} → {label}, "
                          f"compress {ratio:.1f}x")
                else:
                    # 不量化, 保持 FP32
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                    quantized_tensors[name] = tensor
                    stats["skipped"] += 1
                    print(f"  [KEEP]  {name}: {list(tensor.shape)} → float32")
    
    # 保存量化后的模型
    suffix = "int4" if bits == 4 else "int8"
    output_file = output_dir / f"model_{suffix}.safetensors"
    print(f"\nSaving quantized model to {output_file} ...")
    safetensors.torch.save_file(quantized_tensors, str(output_file))
    
    # 复制 config 和 tokenizer 文件
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")
    
    # 保存量化元信息
    if bits == 4:
        quant_method = f"per_group_symmetric_int4_g{group_size}"
    else:
        quant_method = "per_channel_symmetric_int8"

    quant_config = {
        "quant_method": quant_method,
        "bits": bits,
        "group_size": group_size if bits == 4 else -1,
        "quantized_suffixes": QUANTIZE_SUFFIXES,
        "stats": stats,
    }
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    
    # 打印统计
    if stats["total_bytes_fp32"] > 0:
        overall_ratio = stats["total_bytes_fp32"] / stats["total_bytes_quant"]
    else:
        overall_ratio = 1.0
    
    print(f"\n{'='*50}")
    print(f"Quantization complete!")
    print(f"  Method:            INT{bits} {'per-group (g=' + str(group_size) + ')' if bits == 4 else 'per-channel'}")
    print(f"  Quantized tensors: {stats['quantized']}")
    print(f"  Skipped tensors:   {stats['skipped']}")
    print(f"  Original size:     {stats['total_bytes_fp32'] / 1e6:.1f} MB (weight only)")
    print(f"  Quantized size:    {stats['total_bytes_quant'] / 1e6:.1f} MB (weight only)")
    print(f"  Compression ratio: {overall_ratio:.2f}x")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight-Only Quantization (INT8/INT4)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8],
                        help="Quantization bits (4 or 8, default: 8)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for INT4 quantization (default: 128)")
    args = parser.parse_args()
    
    quantize_model(args.model, args.output, bits=args.bits, group_size=args.group_size)
