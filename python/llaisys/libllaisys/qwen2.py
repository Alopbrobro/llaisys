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

    # InferSample
    if hasattr(lib, 'llaisysQwen2ModelInferSample'):
        lib.llaisysQwen2ModelInferSample.argtypes = [
            ctypes.c_void_p,                # model
            ctypes.POINTER(ctypes.c_int64), # token_ids
            ctypes.c_size_t,                # ntoken
            ctypes.c_float,                 # temperature
            ctypes.c_int,                   # top_k
            ctypes.c_float,                 # top_p
        ]
        lib.llaisysQwen2ModelInferSample.restype = ctypes.c_int64

    # ResetCache
    if hasattr(lib, 'llaisysQwen2ResetCache'):
        lib.llaisysQwen2ResetCache.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2ResetCache.restype = None

    # ── Phase 4: KV-Cache 高级接口 ──

    # SaveCache
    if hasattr(lib, 'llaisysQwen2SaveCache'):
        lib.llaisysQwen2SaveCache.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2SaveCache.restype = ctypes.c_void_p

    # RestoreCache
    if hasattr(lib, 'llaisysQwen2RestoreCache'):
        lib.llaisysQwen2RestoreCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llaisysQwen2RestoreCache.restype = None

    # TruncateCache
    if hasattr(lib, 'llaisysQwen2TruncateCache'):
        lib.llaisysQwen2TruncateCache.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lib.llaisysQwen2TruncateCache.restype = None

    # GetCachePos
    if hasattr(lib, 'llaisysQwen2GetCachePos'):
        lib.llaisysQwen2GetCachePos.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2GetCachePos.restype = ctypes.c_int64

    # DestroyCacheSnapshot
    if hasattr(lib, 'llaisysQwen2DestroyCacheSnapshot'):
        lib.llaisysQwen2DestroyCacheSnapshot.argtypes = [ctypes.c_void_p]
        lib.llaisysQwen2DestroyCacheSnapshot.restype = None

    # ── Phase 4: 前缀树 KV-Cache 池 ──

    # KVCachePoolCreate
    if hasattr(lib, 'llaisysKVCachePoolCreate'):
        lib.llaisysKVCachePoolCreate.argtypes = []
        lib.llaisysKVCachePoolCreate.restype = ctypes.c_void_p

    # KVCachePoolDestroy
    if hasattr(lib, 'llaisysKVCachePoolDestroy'):
        lib.llaisysKVCachePoolDestroy.argtypes = [ctypes.c_void_p]
        lib.llaisysKVCachePoolDestroy.restype = None

    # KVCachePoolInsert
    if hasattr(lib, 'llaisysKVCachePoolInsert'):
        lib.llaisysKVCachePoolInsert.argtypes = [
            ctypes.c_void_p,                # pool
            ctypes.POINTER(ctypes.c_int64), # tokens
            ctypes.c_size_t,                # len
            ctypes.c_void_p,                # snapshot
        ]
        lib.llaisysKVCachePoolInsert.restype = None

    # KVCachePoolLookup
    if hasattr(lib, 'llaisysKVCachePoolLookup'):
        lib.llaisysKVCachePoolLookup.argtypes = [
            ctypes.c_void_p,                 # pool
            ctypes.POINTER(ctypes.c_int64),  # tokens
            ctypes.c_size_t,                 # len
            ctypes.POINTER(ctypes.c_size_t), # match_len (output)
        ]
        lib.llaisysKVCachePoolLookup.restype = ctypes.c_void_p

    # KVCachePoolClear
    if hasattr(lib, 'llaisysKVCachePoolClear'):
        lib.llaisysKVCachePoolClear.argtypes = [ctypes.c_void_p]
        lib.llaisysKVCachePoolClear.restype = None

# 执行配置
_setup_functions()

# 3. 导出函数 (方便外部调用)
model_create = LIB_LLAISYS.llaisysQwen2ModelCreate
model_destroy = LIB_LLAISYS.llaisysQwen2ModelDestroy
load_weight = LIB_LLAISYS.llaisysQwen2LoadWeightByName
model_infer = LIB_LLAISYS.llaisysQwen2ModelInfer
model_infer_sample = LIB_LLAISYS.llaisysQwen2ModelInferSample
model_reset_cache = LIB_LLAISYS.llaisysQwen2ResetCache

# Phase 4 导出
cache_save = LIB_LLAISYS.llaisysQwen2SaveCache
cache_restore = LIB_LLAISYS.llaisysQwen2RestoreCache
cache_truncate = LIB_LLAISYS.llaisysQwen2TruncateCache
cache_get_pos = LIB_LLAISYS.llaisysQwen2GetCachePos
cache_snapshot_destroy = LIB_LLAISYS.llaisysQwen2DestroyCacheSnapshot

pool_create = LIB_LLAISYS.llaisysKVCachePoolCreate
pool_destroy = LIB_LLAISYS.llaisysKVCachePoolDestroy
pool_insert = LIB_LLAISYS.llaisysKVCachePoolInsert
pool_lookup = LIB_LLAISYS.llaisysKVCachePoolLookup
pool_clear = LIB_LLAISYS.llaisysKVCachePoolClear