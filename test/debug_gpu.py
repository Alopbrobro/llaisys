#!/usr/bin/env python3
"""
Debug GPU correctness: test each operator individually to find the bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import llaisys
import torch
import numpy as np
from transformers import AutoModelForCausalLM

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
hf_model.eval()

device_name = "nvidia"
device_type = llaisys.DeviceType.NVIDIA
api = llaisys.RuntimeAPI(device_type)

def to_gpu(np_data, dtype=llaisys.DataType.F32):
    """Create a GPU tensor from numpy data."""
    shape = list(np_data.shape)
    t = llaisys.Tensor(shape, dtype=dtype, device=device_type, device_id=0)
    host_data = np.ascontiguousarray(np_data)
    # Use H2D via a host tensor
    host_t = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.CPU, device_id=0)
    import ctypes
    ctypes.memmove(host_t.data_ptr(), host_data.ctypes.data, host_data.nbytes)
    api.memcpy_sync(t.data_ptr(), host_t.data_ptr(), host_data.nbytes, llaisys.MemcpyKind.H2D)
    return t

def from_gpu(t):
    """Read a GPU tensor back to numpy."""
    shape = t.shape()
    dtype_size = 4  # F32
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= dtype_size
    out = np.zeros([s for s in shape], dtype=np.float32)
    host_t = llaisys.Tensor([s for s in shape], dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU, device_id=0)
    api.memcpy_sync(host_t.data_ptr(), t.data_ptr(), nbytes, llaisys.MemcpyKind.D2H)
    import ctypes
    ctypes.memmove(out.ctypes.data, host_t.data_ptr(), nbytes)
    return out

def from_gpu_i32(t):
    """Read a GPU I32 tensor back to numpy."""
    shape = t.shape()
    nbytes = 4
    for s in shape:
        nbytes *= s
    out = np.zeros([s for s in shape], dtype=np.int32)
    host_t = llaisys.Tensor([s for s in shape], dtype=llaisys.DataType.I32, device=llaisys.DeviceType.CPU, device_id=0)
    api.memcpy_sync(host_t.data_ptr(), t.data_ptr(), nbytes, llaisys.MemcpyKind.D2H)
    import ctypes
    ctypes.memmove(out.ctypes.data, host_t.data_ptr(), nbytes)
    return out

# ============ Test 1: Embedding ============
print("=" * 60)
print("Test 1: Embedding")
embed_weight = hf_model.model.embed_tokens.weight.detach().float().numpy()
embed_w_gpu = to_gpu(embed_weight)

token_id = 15191  # "Who"
idx_np = np.array([token_id], dtype=np.int64)
idx_gpu = to_gpu(idx_np, dtype=llaisys.DataType.I64)

embed_dim = embed_weight.shape[1]
out_gpu = llaisys.Tensor([1, embed_dim], dtype=llaisys.DataType.F32, device=device_type, device_id=0)

llaisys.Ops.embedding(out_gpu, idx_gpu, embed_w_gpu)

result = from_gpu(out_gpu)
expected = embed_weight[token_id]
print(f"  Max abs error: {np.max(np.abs(result[0] - expected)):.6e}")
print(f"  Match: {np.allclose(result[0], expected, atol=1e-5)}")

# ============ Test 2: RMS Norm ============
print("=" * 60)
print("Test 2: RMS Norm")
norm_weight = hf_model.model.layers[0].input_layernorm.weight.detach().float().numpy()
norm_w_gpu = to_gpu(norm_weight)
eps = 1e-6

# input is the embedding output
hidden_gpu = llaisys.Tensor([1, embed_dim], dtype=llaisys.DataType.F32, device=device_type, device_id=0)
api.memcpy_sync(hidden_gpu.data_ptr(), out_gpu.data_ptr(), embed_dim * 4, llaisys.MemcpyKind.D2D)

norm_out_gpu = llaisys.Tensor([1, embed_dim], dtype=llaisys.DataType.F32, device=device_type, device_id=0)
llaisys.Ops.rms_norm(norm_out_gpu, hidden_gpu, norm_w_gpu, eps)

norm_result = from_gpu(norm_out_gpu)

# Torch reference
hidden_torch = torch.tensor(expected).unsqueeze(0)
norm_w_torch = torch.tensor(norm_weight)
variance = hidden_torch.pow(2).mean(-1, keepdim=True)
norm_expected = (hidden_torch * torch.rsqrt(variance + eps) * norm_w_torch).numpy()
print(f"  Max abs error: {np.max(np.abs(norm_result - norm_expected)):.6e}")
print(f"  Match: {np.allclose(norm_result, norm_expected, atol=1e-4)}")

# ============ Test 3: Linear (Q projection) ============
print("=" * 60)
print("Test 3: Linear (Q projection, layer 0)")
q_w = hf_model.model.layers[0].self_attn.q_proj.weight.detach().float().numpy()
q_b = hf_model.model.layers[0].self_attn.q_proj.bias.detach().float().numpy()
q_w_gpu = to_gpu(q_w)
q_b_gpu = to_gpu(q_b)

q_dim = q_w.shape[0]
q_out_gpu = llaisys.Tensor([1, q_dim], dtype=llaisys.DataType.F32, device=device_type, device_id=0)

llaisys.Ops.linear(q_out_gpu, norm_out_gpu, q_w_gpu, q_b_gpu)

q_result = from_gpu(q_out_gpu)

# Torch reference: Y = X @ W^T + b
q_expected = (norm_expected @ q_w.T + q_b).astype(np.float32)
print(f"  Max abs error: {np.max(np.abs(q_result - q_expected)):.6e}")
print(f"  Match: {np.allclose(q_result, q_expected, atol=1e-3)}")

# ============ Test 4: RoPE ============
print("=" * 60)
print("Test 4: RoPE")
nh = 12  # num_heads for 1.5B
dh = 128  # head_dim
theta = 10000.0

# Reshape Q to [1, nh, dh]
q_3d_np = q_result.reshape(1, nh, dh)
q_3d_gpu = to_gpu(q_3d_np)
q_rope_out = llaisys.Tensor([1, nh, dh], dtype=llaisys.DataType.F32, device=device_type, device_id=0)

pos_np = np.array([0], dtype=np.int64)
pos_gpu = to_gpu(pos_np, dtype=llaisys.DataType.I64)

llaisys.Ops.rope(q_rope_out, q_3d_gpu, pos_gpu, theta)

rope_result = from_gpu(q_rope_out)

# Torch reference for RoPE at position 0
# At position 0, angle = 0 for all dims, so cos=1 sin=0, output should equal input
print(f"  At pos=0, max diff from input: {np.max(np.abs(rope_result - q_3d_np)):.6e}")
# cos(0)=1, sin(0)=0, so a'=a*1-b*0=a, b'=b*1+a*0=b
print(f"  RoPE at pos=0 should be identity: {np.allclose(rope_result, q_3d_np, atol=1e-5)}")

# Test at pos=1
pos_np1 = np.array([1], dtype=np.int64)
pos_gpu1 = to_gpu(pos_np1, dtype=llaisys.DataType.I64)
llaisys.Ops.rope(q_rope_out, q_3d_gpu, pos_gpu1, theta)
rope_result_pos1 = from_gpu(q_rope_out)

# Manual rope at pos=1
half = dh // 2
rope_expected = np.zeros_like(q_3d_np)
for h in range(nh):
    for l in range(half):
        angle = 1.0 / (theta ** (2.0 * l / dh))
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        a = q_3d_np[0, h, l]
        b = q_3d_np[0, h, l + half]
        rope_expected[0, h, l] = a * cos_val - b * sin_val
        rope_expected[0, h, l + half] = b * cos_val + a * sin_val

print(f"  At pos=1, max abs error: {np.max(np.abs(rope_result_pos1 - rope_expected)):.6e}")
print(f"  RoPE at pos=1 match: {np.allclose(rope_result_pos1, rope_expected, atol=1e-4)}")

# ============ Test 5: Self Attention (single token, single head) ============
print("=" * 60)
print("Test 5: Self Attention (1 token)")

nkvh = 2  # kv heads for 1.5B
# Compute K and V projections
k_w = hf_model.model.layers[0].self_attn.k_proj.weight.detach().float().numpy()
k_b = hf_model.model.layers[0].self_attn.k_proj.bias.detach().float().numpy()
v_w = hf_model.model.layers[0].self_attn.v_proj.weight.detach().float().numpy()
v_b = hf_model.model.layers[0].self_attn.v_proj.bias.detach().float().numpy()

kv_dim = k_w.shape[0]
k_proj = (norm_expected @ k_w.T + k_b).reshape(1, nkvh, dh).astype(np.float32)
v_proj = (norm_expected @ v_w.T + v_b).reshape(1, nkvh, dh).astype(np.float32)

# Apply RoPE to K at pos 0 (identity)
# k_proj stays the same at pos 0

q_attn = rope_result.copy()  # Q after RoPE at pos=0 (same as input at pos 0)
k_attn = k_proj.copy()
v_attn = v_proj.copy()

# GPU tensors
q_attn_gpu = to_gpu(q_attn)
k_attn_gpu = to_gpu(k_attn)
v_attn_gpu = to_gpu(v_attn)
attn_out_gpu = llaisys.Tensor([1, nh, dh], dtype=llaisys.DataType.F32, device=device_type, device_id=0)

scale = 1.0 / np.sqrt(dh)
llaisys.Ops.self_attention(attn_out_gpu, q_attn_gpu, k_attn_gpu, v_attn_gpu, scale)
attn_result = from_gpu(attn_out_gpu)

# Manual reference: for single token, attention scores = scale * Q * K^T
# With causal mask (j <= current_pos=0), only j=0 is kept
# Softmax of single element = 1.0
# Output = 1.0 * V
group_size = nh // nkvh
attn_expected = np.zeros_like(q_attn)
for h in range(nh):
    kv_h = h // group_size
    # score = scale * dot(q[h], k[kv_h]) — but softmax of 1 element = 1
    # so output[h] = v[kv_h]
    attn_expected[0, h, :] = v_attn[0, kv_h, :]

print(f"  Max abs error: {np.max(np.abs(attn_result - attn_expected)):.6e}")
print(f"  Match (single token attn = V): {np.allclose(attn_result, attn_expected, atol=1e-4)}")
if not np.allclose(attn_result, attn_expected, atol=1e-4):
    # Print first few values for debugging
    print(f"  Expected head 0: {attn_expected[0,0,:5]}")
    print(f"  Got head 0:      {attn_result[0,0,:5]}")
    print(f"  Expected head 6: {attn_expected[0,6,:5]}")
    print(f"  Got head 6:      {attn_result[0,6,:5]}")

# ============ Test 6: Add ============
print("=" * 60)
print("Test 6: Add")
a_np = np.random.randn(1, 1536).astype(np.float32)
b_np = np.random.randn(1, 1536).astype(np.float32)
a_gpu = to_gpu(a_np)
b_gpu = to_gpu(b_np)
c_gpu = llaisys.Tensor([1, 1536], dtype=llaisys.DataType.F32, device=device_type, device_id=0)
llaisys.Ops.add(c_gpu, a_gpu, b_gpu)
c_result = from_gpu(c_gpu)
c_expected = a_np + b_np
print(f"  Max abs error: {np.max(np.abs(c_result - c_expected)):.6e}")
print(f"  Match: {np.allclose(c_result, c_expected, atol=1e-5)}")

# Test in-place add (c = a + b where c is same as a)
print("Test 6b: In-place Add (c=a, c=a+b)")
api.memcpy_sync(c_gpu.data_ptr(), a_gpu.data_ptr(), 1536*4, llaisys.MemcpyKind.D2D)
llaisys.Ops.add(c_gpu, c_gpu, b_gpu)
c_result2 = from_gpu(c_gpu)
print(f"  Max abs error: {np.max(np.abs(c_result2 - c_expected)):.6e}")
print(f"  Match: {np.allclose(c_result2, c_expected, atol=1e-5)}")

# ============ Test 7: SwiGLU ============
print("=" * 60)
print("Test 7: SwiGLU")
gate_np = np.random.randn(1, 4096).astype(np.float32)
up_np = np.random.randn(1, 4096).astype(np.float32)
gate_gpu = to_gpu(gate_np)
up_gpu = to_gpu(up_np)
swiglu_out_gpu = llaisys.Tensor([1, 4096], dtype=llaisys.DataType.F32, device=device_type, device_id=0)
llaisys.Ops.swiglu(swiglu_out_gpu, gate_gpu, up_gpu)
swiglu_result = from_gpu(swiglu_out_gpu)
silu_gate = gate_np / (1.0 + np.exp(-gate_np))
swiglu_expected = up_np * silu_gate
print(f"  Max abs error: {np.max(np.abs(swiglu_result - swiglu_expected)):.6e}")
print(f"  Match: {np.allclose(swiglu_result, swiglu_expected, atol=1e-4)}")

# ============ Test 8: Argmax ============
print("=" * 60)
print("Test 8: Argmax")
vals_np = np.random.randn(1, 151936).astype(np.float32)
vals_np[0, 42] = 100.0  # make 42 the argmax
vals_gpu = to_gpu(vals_np)
idx_out_gpu = llaisys.Tensor([1], dtype=llaisys.DataType.I32, device=device_type, device_id=0)
val_out_gpu = llaisys.Tensor([1], dtype=llaisys.DataType.F32, device=device_type, device_id=0)
llaisys.Ops.argmax(idx_out_gpu, val_out_gpu, vals_gpu)
idx_result = from_gpu_i32(idx_out_gpu)
val_result = from_gpu(val_out_gpu)
print(f"  Expected idx: 42, Got: {idx_result[0]}")
print(f"  Expected val: 100.0, Got: {val_result[0]:.4f}")
print(f"  Match: {idx_result[0] == 42 and abs(val_result[0] - 100.0) < 0.01}")

print("=" * 60)
print("All individual operator tests done!")
