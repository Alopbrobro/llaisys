#!/usr/bin/env python3
"""INT8 Weight-Only 对称量化工具

用法:
    python scripts/quantize.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output ./quantized_model

原理:
    对每个 Linear 权重矩阵 W (shape [out, in]):
    1. 按 per-channel (每一行) 计算 scale = max(|W_row|) / 127
    2. 量化: W_int8 = round(W / scale).clamp(-128, 127)
    3. 保存: W_int8 (int8) + scale (float32, shape [out])
    
    推理时 dequantize: W_fp32 ≈ W_int8 * scale (per-channel broadcast)
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


def quantize_model(model_path: str, output_dir: str):
    """读取 safetensors 模型并量化"""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    stats = {"quantized": 0, "skipped": 0, "total_params_fp32": 0, "total_params_int8": 0}
    
    for file in st_files:
        print(f"\nProcessing: {file.name}")
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                
                if should_quantize(name):
                    # 量化
                    w_int8, scale = quantize_per_channel_symmetric(tensor)
                    quantized_tensors[name] = w_int8
                    quantized_tensors[name + ".scale"] = scale
                    
                    # 统计
                    orig_bytes = tensor.numel() * tensor.element_size()
                    quant_bytes = w_int8.numel() * 1 + scale.numel() * 4
                    ratio = orig_bytes / quant_bytes
                    
                    stats["quantized"] += 1
                    stats["total_params_fp32"] += tensor.numel() * tensor.element_size()
                    stats["total_params_int8"] += quant_bytes
                    
                    print(f"  [QUANT] {name}: {list(tensor.shape)} "
                          f"{tensor.dtype} → int8, "
                          f"compress {ratio:.1f}x")
                else:
                    # 不量化, 保持 FP32
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                    quantized_tensors[name] = tensor
                    stats["skipped"] += 1
                    print(f"  [KEEP]  {name}: {list(tensor.shape)} → float32")
    
    # 保存量化后的模型
    output_file = output_dir / "model_int8.safetensors"
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
    quant_config = {
        "quant_method": "per_channel_symmetric_int8",
        "bits": 8,
        "group_size": -1,  # per-channel, 不是 group
        "quantized_suffixes": QUANTIZE_SUFFIXES,
        "stats": stats,
    }
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    
    # 打印统计
    if stats["total_params_fp32"] > 0:
        overall_ratio = stats["total_params_fp32"] / stats["total_params_int8"]
    else:
        overall_ratio = 1.0
    
    print(f"\n{'='*50}")
    print(f"Quantization complete!")
    print(f"  Quantized tensors: {stats['quantized']}")
    print(f"  Skipped tensors:   {stats['skipped']}")
    print(f"  Original size:     {stats['total_params_fp32'] / 1e6:.1f} MB (weight only)")
    print(f"  Quantized size:    {stats['total_params_int8'] / 1e6:.1f} MB (weight only)")
    print(f"  Compression ratio: {overall_ratio:.2f}x")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT8 Weight-Only Quantization")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    args = parser.parse_args()
    
    quantize_model(args.model, args.output)
