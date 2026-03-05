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

    def _load_weights(self, model_path):
        import json
        
        # 检测是否是量化模型 (有 quant_config.json)
        quant_config_path = model_path / "quant_config.json"
        is_quantized = quant_config_path.exists()
        if is_quantized:
            with open(quant_config_path) as f:
                qcfg = json.load(f)
            print(f"Detected quantized model: {qcfg.get('quant_method', 'unknown')}")
        
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
                    elif name_.endswith(".scale"):
                        # scale 向量 — 确保 FP32
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