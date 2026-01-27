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
    model_infer
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
        return 0

    def _load_weights(self, model_path):
        files = sorted(list(model_path.glob("*.safetensors")))
        if not files:
            print(f"Warning: No .safetensors files found in {model_path}")
            
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    tensor = data_.get_tensor(name_)
                    # 1. 强制转为 Float32 以匹配 C++ 计算类型
                    if tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    
                    c_name = name_.encode('utf-8')
                    # 2. 确保内存连续
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        
                    data_ptr = tensor.data_ptr()
                    ndim = len(tensor.shape)
                    shape_array = (ctypes.c_int64 * ndim)(*tensor.shape)
                    dtype = 13 # F32
                    
                    load_weight(self.model_handle, c_name, ctypes.c_void_p(data_ptr), ndim, shape_array, dtype)

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
            
        output_ids = []
        max_tokens = max_new_tokens if max_new_tokens is not None else 20
        
        # --- 1. Prefill ---
        input_len = len(inputs)
        
        # 【终极修复】使用 Numpy 数组来传递数据
        # 1. 创建 numpy int64 数组，这是内存中最标准的 C int64_t 数组形式
        input_np = np.array(inputs, dtype=np.int64)
        
        # 2. 确保内存连续 (Contiguous)，否则 C++ 指针移动会出错
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
            
        # 3. 获取底层数据指针 (void*) 并转为 int64_t*
        # 这一步绕过了 ctypes 所有的自动转换猜测，直接传内存地址
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        # 调用推理
        next_token = model_infer(self.model_handle, input_ptr, ctypes.c_size_t(input_len))
        
        output_ids.append(next_token)
        current_token = next_token
        
        # --- 2. Decoding ---
        for _ in range(max_tokens - 1):
            # 同样对单个 token 使用 numpy 处理
            token_np = np.array([current_token], dtype=np.int64)
            token_ptr = token_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            
            next_token = model_infer(self.model_handle, token_ptr, ctypes.c_size_t(1))
            
            output_ids.append(next_token)
            current_token = next_token
            
            if next_token == 151643: # EOS
                break

        return output_ids

    def __del__(self):
        if hasattr(self, 'model_handle') and self.model_handle:
            model_destroy(self.model_handle)