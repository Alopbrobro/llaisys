import ctypes
import numpy as np
import torch
from typing import Sequence, Optional
from ..libllaisys import DeviceType

# 引入底层接口定义
from ..libllaisys.qwen2 import (
    LlaisysQwen2Meta, 
    model_create, 
    model_destroy, 
    load_weight, 
    model_infer,
    model_infer_sample,
    model_reset_cache,
    # Phase 4: KV-Cache 高级接口
    cache_save,
    cache_restore,
    cache_truncate,
    cache_get_pos,
    cache_snapshot_destroy,
    # Phase 4: 前缀树 KV-Cache 池
    pool_create,
    pool_destroy,
    pool_insert,
    pool_lookup,
    pool_clear,
)

from pathlib import Path
import safetensors
import os

# =========================================================================
# GPTQ/AWQ format constants
# =========================================================================

# GPTQ quantized weight suffixes (without the .qweight etc.)
GPTQ_LINEAR_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# =========================================================================
# 强制覆盖函数签名 (确保指针类型正确)
# =========================================================================
model_create.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
model_create.restype = ctypes.c_void_p

model_destroy.argtypes = [ctypes.c_void_p]
model_destroy.restype = None

load_weight.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
load_weight.restype = None

# 重点：Infer 的签名
model_infer.argtypes = [
    ctypes.c_void_p,                # model
    ctypes.POINTER(ctypes.c_int64), # token_ids (int64_t*)
    ctypes.c_size_t                 # ntoken (size_t)
]
model_infer.restype = ctypes.c_int64

# InferSample 的签名
model_infer_sample.argtypes = [
    ctypes.c_void_p,                # model
    ctypes.POINTER(ctypes.c_int64), # token_ids (int64_t*)
    ctypes.c_size_t,                # ntoken (size_t)
    ctypes.c_float,                 # temperature
    ctypes.c_int,                   # top_k
    ctypes.c_float,                 # top_p
]
model_infer_sample.restype = ctypes.c_int64
# =========================================================================

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # 初始化配置
        meta = LlaisysQwen2Meta()
        meta.dtype = 13 
        meta.nlayer = 28
        meta.hs = 1536
        meta.nh = 12
        meta.nkvh = 2
        meta.dh = 128
        meta.di = 8960
        meta.maxseq = 4096
        meta.voc = 151936
        meta.epsilon = 1e-6
        meta.theta = 10000.0
        meta.end_token = 151643

        print("Creating Qwen2 Model instance...")
        self.model_handle = model_create(ctypes.byref(meta), device.value, None, 0)

        # 加载权重
        model_path = Path(model_path)
        print(f"Loading weights from {model_path}...")
        self._load_weights(model_path)
        print("Model loaded successfully.")

    def _get_dtype_enum(self, dtype_str):
        if "float32" in dtype_str: return 13
        if "float16" in dtype_str: return 12
        if "bfloat16" in dtype_str: return 19
        if "int64" in dtype_str: return 6
        if "int32" in dtype_str: return 5
        if "int8" in dtype_str: return 3
        return 0

    # =====================================================================
    # GPTQ/AWQ format helpers
    # =====================================================================

    @staticmethod
    def _unpack_int32_to_int4(packed_int32, bits=4):
        """Unpack int32-packed values to individual uint8 values.

        GPTQ packing: 8 × 4-bit values per int32 (for bits=4).
        qweight shape [rows_packed, cols] → [rows_packed * pack_factor, cols]
        qzeros  shape [num_groups, cols_packed] → [num_groups, cols_packed * pack_factor]
        """
        pack_factor = 32 // bits  # 8 for 4-bit
        mask = (1 << bits) - 1    # 0xF for 4-bit

        results = []
        for i in range(pack_factor):
            results.append(((packed_int32 >> (i * bits)) & mask).to(torch.int32))
        # Interleave along the packed axis
        return results, pack_factor

    @staticmethod
    def _convert_gptq_layer(qweight, qzeros, scales, bits=4, group_size=128):
        """Convert one GPTQ/AWQ linear layer to our symmetric INT4 format.

        Args:
            qweight: [in_features // pack_factor, out_features] int32
            qzeros:  [num_groups, out_features // pack_factor] int32
            scales:  [num_groups, out_features] float16/float32
            bits:    quantization bits (4)
            group_size: elements per group

        Returns:
            packed:  [out_features, in_features // 2] uint8 (our format)
            scale:   [out_features, num_groups] float32 (our format)
        """
        pack_factor = 32 // bits  # 8 for 4-bit
        mask = (1 << bits) - 1    # 0xF

        in_packed, out_features = qweight.shape
        in_features = in_packed * pack_factor
        num_groups = scales.shape[0]

        # --- Step 1: Unpack qweight [in_packed, out] → [in, out] ---
        w_unpacked = torch.zeros(in_features, out_features, dtype=torch.int32)
        for i in range(pack_factor):
            w_unpacked[i::pack_factor, :] = (qweight >> (i * bits)) & mask

        # --- Step 2: Unpack qzeros [ngroup, out_packed] → [ngroup, out] ---
        z_unpacked = torch.zeros(num_groups, out_features, dtype=torch.int32)
        for i in range(pack_factor):
            z_unpacked[:, i::pack_factor] = (qzeros >> (i * bits)) & mask

        # --- Step 3: Dequantize to FP32 ---
        # w_float[row][col] = (w_unpacked[row][col] - (z[group][col] + 1)) * s[group][col]
        # Note: AutoGPTQ stores zero_point - 1, so we add 1 back
        scales_f32 = scales.float()
        w_float = torch.zeros(in_features, out_features, dtype=torch.float32)
        for g in range(num_groups):
            r_start = g * group_size
            r_end = min(r_start + group_size, in_features)
            w_slice = w_unpacked[r_start:r_end, :].float()  # [gs, out]
            z_row = (z_unpacked[g, :].float() + 1.0).unsqueeze(0)  # [1, out], +1 correction
            s_row = scales_f32[g, :].unsqueeze(0)            # [1, out]
            w_float[r_start:r_end, :] = (w_slice - z_row) * s_row

        # --- Step 4: Transpose to our [out, in] layout ---
        w_float = w_float.T.contiguous()  # [out_features, in_features]

        # --- Step 5: Re-quantize to our symmetric INT4 per-group ---
        rows, cols = w_float.shape
        assert cols % group_size == 0, f"in_features {cols} not divisible by group_size {group_size}"
        ngroups = cols // group_size

        w_grouped = w_float.reshape(rows, ngroups, group_size)
        group_max = w_grouped.abs().amax(dim=-1).clamp(min=1e-10)  # [rows, ngroups]
        our_scale = group_max / 7.0

        w_q = (w_grouped / our_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
        w_q = w_q.reshape(rows, cols)

        # Pack two INT4 per byte
        w_even = (w_q[:, 0::2] + 8).to(torch.uint8)  # [rows, cols/2]
        w_odd  = (w_q[:, 1::2] + 8).to(torch.uint8)
        packed = (w_odd << 4) | (w_even & 0x0F)       # [rows, cols/2] uint8

        return packed, our_scale.float()

    def _call_load_weight(self, name, tensor, dtype_enum):
        """Helper to call the C load_weight function with proper ctypes."""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        c_name = name.encode('utf-8')
        data_ptr = tensor.data_ptr()
        ndim = len(tensor.shape)
        shape_array = (ctypes.c_int64 * ndim)(*tensor.shape)
        load_weight(self.model_handle, c_name, ctypes.c_void_p(data_ptr), ndim, shape_array, dtype_enum)

    def _load_weights(self, model_path):
        import json
        
        # ─── Detect quantization format ───
        quant_config_path = model_path / "quant_config.json"       # Our native format
        gptq_config_path = model_path / "quantize_config.json"     # GPTQ / AWQ standalone
        config_json_path = model_path / "config.json"              # HF config (may embed GPTQ config)

        # GPTQ/AWQ: standalone quantize_config.json OR embedded in config.json
        gcfg = None
        if gptq_config_path.exists():
            with open(gptq_config_path) as f:
                gcfg = json.load(f)
        elif config_json_path.exists():
            with open(config_json_path) as f:
                main_cfg = json.load(f)
            if "quantization_config" in main_cfg:
                qc = main_cfg["quantization_config"]
                if qc.get("quant_method") in ("gptq", "awq"):
                    gcfg = qc

        if gcfg is not None:
            quant_method = gcfg.get("quant_method", "gptq")
            bits = gcfg.get("bits", 4)
            group_size = gcfg.get("group_size", 128)
            desc_act = gcfg.get("desc_act", False)
            if desc_act:
                print("WARNING: desc_act=True 暂不支持, 视为 False (可能影响精度)")
            print(f"Detected {quant_method.upper()} model: {bits}-bit, group_size={group_size}")
            self._load_weights_gptq(model_path, bits, group_size)
            return

        # ─── Our native format (INT8 / INT4 / FP32) ───
        is_quantized = quant_config_path.exists()
        quant_bits = 0
        if is_quantized:
            with open(quant_config_path) as f:
                qcfg = json.load(f)
            quant_bits = qcfg.get("bits", 8)
            print(f"Detected quantized model: {qcfg.get('quant_method', 'unknown')} (bits={quant_bits})")
        
        files = sorted(list(model_path.glob("*.safetensors")))
        if not files:
            print(f"Warning: No .safetensors files found in {model_path}")
            
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    tensor = data_.get_tensor(name_)
                    
                    c_name = name_.encode('utf-8')
                    
                    # 确保内存连续
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    
                    # 根据 tensor dtype 决定如何传给 C++
                    if tensor.dtype == torch.int8:
                        # INT8 量化权重 — 原样传递
                        dtype_enum = 3  # LLAISYS_DTYPE_I8
                    elif tensor.dtype == torch.uint8:
                        # INT4 packed 量化权重 — 原样传递  (U8)
                        dtype_enum = 7  # LLAISYS_DTYPE_U8
                    elif name_.endswith(".scale"):
                        # scale 向量/矩阵 — 确保 FP32
                        if tensor.dtype != torch.float32:
                            tensor = tensor.to(torch.float32)
                        dtype_enum = 13  # LLAISYS_DTYPE_F32
                    else:
                        # 普通权重 — 转 FP32
                        if tensor.dtype != torch.float32:
                            tensor = tensor.to(torch.float32)
                        dtype_enum = 13  # LLAISYS_DTYPE_F32
                        
                    data_ptr = tensor.data_ptr()
                    ndim = len(tensor.shape)
                    shape_array = (ctypes.c_int64 * ndim)(*tensor.shape)
                    
                    load_weight(self.model_handle, c_name, ctypes.c_void_p(data_ptr), ndim, shape_array, dtype_enum)

    def _load_weights_gptq(self, model_path, bits=4, group_size=128):
        """Load GPTQ/AWQ format model, converting to our symmetric INT4 at load time.
        
        GPTQ tensors per linear layer:
          .qweight → [in_features // 8, out_features] int32
          .qzeros  → [num_groups, out_features // 8] int32
          .scales  → [num_groups, out_features] float16
          .g_idx   → [in_features] int32 (optional, ignored)
        
        Non-linear tensors (embed, norm, bias) are loaded as FP32.
        """
        files = sorted(list(model_path.glob("*.safetensors")))
        if not files:
            print(f"Warning: No .safetensors files found in {model_path}")
            return

        # First pass: collect all tensors into dict
        all_tensors = {}
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    all_tensors[name] = f.get_tensor(name)

        processed = set()
        converted_count = 0

        for name in sorted(all_tensors.keys()):
            if name in processed:
                continue
            tensor = all_tensors[name]

            if name.endswith(".qweight"):
                # ─── GPTQ quantized linear layer ───
                base = name[:-len(".qweight")]
                qweight = tensor
                qzeros = all_tensors.get(base + ".qzeros")
                scales = all_tensors.get(base + ".scales")

                if qzeros is None or scales is None:
                    print(f"  WARNING: Missing qzeros/scales for {base}, loading as FP32")
                    continue

                # Convert GPTQ → our symmetric INT4
                packed, our_scale = self._convert_gptq_layer(
                    qweight, qzeros, scales, bits=bits, group_size=group_size)

                # Map to our weight naming convention
                weight_name = base + ".weight"
                scale_name = base + ".weight.scale"

                self._call_load_weight(weight_name, packed, dtype_enum=7)   # U8
                self._call_load_weight(scale_name, our_scale, dtype_enum=13) # FP32

                processed.update([name, base + ".qzeros", base + ".scales"])
                if base + ".g_idx" in all_tensors:
                    processed.add(base + ".g_idx")
                if base + ".bias" in all_tensors:
                    # GPTQ linear may have bias — load as FP32
                    bias = all_tensors[base + ".bias"]
                    if bias.dtype != torch.float32:
                        bias = bias.to(torch.float32)
                    self._call_load_weight(base + ".bias", bias, dtype_enum=13)
                    processed.add(base + ".bias")

                converted_count += 1
                print(f"  [GPTQ→INT4] {weight_name}: {list(packed.shape)} + scale {list(our_scale.shape)}")

            elif name.endswith((".qzeros", ".scales", ".g_idx")):
                # Handled together with .qweight
                continue

            else:
                # ─── Regular tensor (embedding, norm, bias, lm_head) ───
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(torch.float32)

                # lm_head.weight in GPTQ is sometimes still quantized
                if name == "lm_head.weight" and "lm_head.qweight" in all_tensors:
                    continue  # handled above

                self._call_load_weight(name, tensor, dtype_enum=13)
                processed.add(name)
                print(f"  [KEEP]  {name}: {list(tensor.shape)} → float32")

        print(f"  Converted {converted_count} GPTQ layers to symmetric INT4")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if not inputs:
            return []

        output_ids = list(inputs)
        max_tokens = max_new_tokens if max_new_tokens is not None else 20
        
        use_sampling = (top_k != 1) and (temperature > 0.0)
        
        # --- 1. Prefill ---
        input_len = len(inputs)
        input_np = np.array(inputs, dtype=np.int64)
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        if use_sampling:
            next_token = model_infer_sample(
                self.model_handle, input_ptr, ctypes.c_size_t(input_len),
                ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
        else:
            next_token = model_infer(self.model_handle, input_ptr, ctypes.c_size_t(input_len))

        output_ids.append(int(next_token))
        current_token = next_token
        
        # --- 2. Decoding ---
        for _ in range(max_tokens - 1):
            token_np = np.array([current_token], dtype=np.int64)
            token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            
            if use_sampling:
                next_token = model_infer_sample(
                    self.model_handle, token_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
            else:
                next_token = model_infer(self.model_handle, token_ptr, ctypes.c_size_t(1))

            output_ids.append(int(next_token))
            current_token = next_token
            
            if next_token == 151643: # EOS
                break

        return output_ids

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generator that yields one token at a time."""
        if not inputs:
            return

        max_tokens = max_new_tokens if max_new_tokens is not None else 512
        use_sampling = (top_k != 1) and (temperature > 0.0)
        
        # Prefill
        input_len = len(inputs)
        input_np = np.array(inputs, dtype=np.int64)
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        if use_sampling:
            next_token = model_infer_sample(
                self.model_handle, input_ptr, ctypes.c_size_t(input_len),
                ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
        else:
            next_token = model_infer(self.model_handle, input_ptr, ctypes.c_size_t(input_len))

        next_token = int(next_token)
        yield next_token
        current_token = next_token
        
        # Decode
        for _ in range(max_tokens - 1):
            if current_token == 151643:  # EOS
                break
            token_np = np.array([current_token], dtype=np.int64)
            token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            
            if use_sampling:
                next_token = model_infer_sample(
                    self.model_handle, token_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k), ctypes.c_float(top_p))
            else:
                next_token = model_infer(self.model_handle, token_ptr, ctypes.c_size_t(1))

            next_token = int(next_token)
            yield next_token
            current_token = next_token

    def __del__(self):
        if hasattr(self, 'model_handle') and self.model_handle:
            model_destroy(self.model_handle)

    def reset_cache(self):
        """Reset the KV-cache position without reloading weights."""
        model_reset_cache(self.model_handle)

    # ==========================================
    # Phase 4: KV-Cache 高级接口
    # ==========================================

    def save_cache(self):
        """保存当前 KV-Cache 快照 (深拷贝到 CPU).
        
        Returns:
            int: 快照句柄 (C++ 指针), 如果 cache 为空则返回 None.
        """
        handle = cache_save(self.model_handle)
        if not handle:
            return None
        return handle

    def restore_cache(self, snapshot_handle):
        """从快照恢复 KV-Cache.
        
        Args:
            snapshot_handle: save_cache() 返回的快照句柄.
        """
        if snapshot_handle:
            cache_restore(self.model_handle, snapshot_handle)

    def truncate_cache(self, pos: int):
        """截断 KV-Cache 到指定位置.
        
        Args:
            pos: 目标位置 (0 = 清空, pos <= current_pos).
        """
        cache_truncate(self.model_handle, ctypes.c_int64(pos))

    def get_cache_pos(self) -> int:
        """获取当前 KV-Cache 位置 (已处理的 token 数)."""
        return int(cache_get_pos(self.model_handle))

    @staticmethod
    def destroy_snapshot(snapshot_handle):
        """释放快照内存.
        
        Args:
            snapshot_handle: save_cache() 返回的快照句柄.
        """
        if snapshot_handle:
            cache_snapshot_destroy(snapshot_handle)

    # ==========================================
    # Phase 4: 前缀树 KV-Cache 池
    # ==========================================

    def create_cache_pool(self):
        """创建 KV-Cache 前缀树池.
        
        Returns:
            int: 池句柄.
        """
        return pool_create()

    @staticmethod
    def destroy_cache_pool(pool_handle):
        """销毁 KV-Cache 前缀树池."""
        if pool_handle:
            pool_destroy(pool_handle)

    @staticmethod
    def cache_pool_insert(pool_handle, tokens: Sequence[int], snapshot_handle):
        """向前缀树池插入快照 (池获取所有权).
        
        Args:
            pool_handle: create_cache_pool() 返回的池句柄.
            tokens: token 序列 (前缀).
            snapshot_handle: save_cache() 返回的快照句柄, 插入后调用者不再拥有.
        """
        if not pool_handle or not snapshot_handle or not tokens:
            return
        token_np = np.array(tokens, dtype=np.int64)
        token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        pool_insert(pool_handle, token_ptr, ctypes.c_size_t(len(tokens)), snapshot_handle)

    @staticmethod
    def cache_pool_lookup(pool_handle, tokens: Sequence[int]):
        """在前缀树池中查找最长前缀匹配.
        
        Args:
            pool_handle: 池句柄.
            tokens: 要匹配的 token 序列.
            
        Returns:
            tuple: (snapshot_handle or None, match_len: int)
        """
        if not pool_handle or not tokens:
            return None, 0
        token_np = np.array(tokens, dtype=np.int64)
        token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        match_len = ctypes.c_size_t(0)
        snap = pool_lookup(pool_handle, token_ptr, ctypes.c_size_t(len(tokens)), ctypes.byref(match_len))
        if not snap:
            return None, 0
        return snap, int(match_len.value)

    @staticmethod
    def cache_pool_clear(pool_handle):
        """清空前缀树池."""
        if pool_handle:
            pool_clear(pool_handle)