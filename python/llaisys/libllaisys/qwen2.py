import ctypes
from . import LIB_LLAISYS

# 1. 定义 C 结构体 (对应 qwen2.h 中的 LlaisysQwen2Meta)
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

# 2. 配置函数签名
def _setup_functions():
    lib = LIB_LLAISYS
    
    # Create
    if hasattr(lib, 'llaisysQwen2ModelCreate'):
        lib.llaisysQwen2ModelCreate.argtypes = [
            ctypes.POINTER(LlaisysQwen2Meta), 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_int), 
            ctypes.c_int
        ]
        lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

    # Destroy
    if hasattr(lib, 'llaisysQwen2ModelDestroy'):
        lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2ModelDestroy.restype = None

    # Load Weight
    if hasattr(lib, 'llaisysQwen2LoadWeightByName'):
        lib.llaisysQwen2LoadWeightByName.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.c_char_p,                # name
            ctypes.c_void_p,                # data
            ctypes.c_int,                   # ndim
            ctypes.POINTER(ctypes.c_int64), # shape
            ctypes.c_int                    # dtype
        ]
        lib.llaisysQwen2LoadWeightByName.restype = None

    # Infer
    if hasattr(lib, 'llaisysQwen2ModelInfer'):
        lib.llaisysQwen2ModelInfer.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.POINTER(ctypes.c_int64), # token_ids
            ctypes.c_size_t                 # ntoken
        ]
        lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

# 执行配置
_setup_functions()

# 3. 导出函数 (方便外部调用)
model_create = LIB_LLAISYS.llaisysQwen2ModelCreate
model_destroy = LIB_LLAISYS.llaisysQwen2ModelDestroy
load_weight = LIB_LLAISYS.llaisysQwen2LoadWeightByName
model_infer = LIB_LLAISYS.llaisysQwen2ModelInfer