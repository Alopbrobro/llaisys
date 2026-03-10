// Phase B smoke test: 验证 TP 权重分片加载
// 创建 tp_size=2 的模型，加载假权重，检查切片后的维度

#include "llaisys/models/qwen2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

// 生成长度为 numel 的全 1.0 FP32 数据
static std::vector<float> make_fp32(size_t numel) {
    std::vector<float> v(numel, 1.0f);
    return v;
}

// 通过 Weights 结构体的 handle 获取 tensor 形状 (重新 cast 回 Tensor*)
// 由于 llaisysTensor_t = Tensor*, 我们直接读 shape
#include "src/tensor/tensor.hpp"
static std::vector<size_t> get_shape(llaisysTensor_t t) {
    if (!t) return {};
    auto* ptr = reinterpret_cast<llaisys::Tensor*>(t);
    return ptr->shape();
}

int main() {
    // ── 配置: 模拟 Qwen2-style 模型 ──
    // nh=16, nkvh=2, dh=8, hs=128, di=32, nlayer=2
    // tp_size=2 → local_nh=8, local_nkvh=1, local_di=16
    LlaisysQwen2Meta meta = {};
    meta.dtype = LLAISYS_DTYPE_F32;
    meta.nlayer = 2;
    meta.hs = 128;     // hidden_size = nh * dh
    meta.nh = 16;      // num_heads
    meta.nkvh = 2;     // num_kv_heads (GQA)
    meta.dh = 8;       // head_dim = hs / nh
    meta.di = 32;      // intermediate_size
    meta.maxseq = 64;
    meta.voc = 100;
    meta.epsilon = 1e-6f;
    meta.theta = 10000.0f;
    meta.end_token = 0;

    int tp_size = 2;
    int ok = 1;

    printf("[tp-shard-smoke] Creating TP models (tp_size=%d) ...\n", tp_size);

    for (int tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
        auto* model = llaisysQwen2ModelCreateTP(&meta, LLAISYS_DEVICE_CPU, 0, tp_size, tp_rank);
        assert(model != nullptr);
        assert(llaisysQwen2GetTpSize(model) == tp_size);
        assert(llaisysQwen2GetTpRank(model) == tp_rank);

        printf("  rank=%d: model created\n", tp_rank);

        // ── 加载顶层权重 (不切分) ──
        {
            // embed_tokens: [voc, hs]
            auto d = make_fp32(meta.voc * meta.hs);
            int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.embed_tokens.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        {
            // model.norm.weight: [hs]
            auto d = make_fp32(meta.hs);
            int64_t sh[] = {(int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.norm.weight",
                d.data(), 1, sh, LLAISYS_DTYPE_F32);
        }
        {
            // lm_head.weight: [voc, hs]
            auto d = make_fp32(meta.voc * meta.hs);
            int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "lm_head.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }

        // ── 加载 layer 0 权重 ──
        // Q weight: [nh*dh, hs] = [128, 128] → column parallel → [64, 128]
        {
            size_t full_rows = meta.nh * meta.dh;  // 128
            auto d = make_fp32(full_rows * meta.hs);
            int64_t sh[] = {(int64_t)full_rows, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.q_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // Q bias: [nh*dh] = [128] → column parallel → [64]
        {
            size_t full = meta.nh * meta.dh;
            auto d = make_fp32(full);
            int64_t sh[] = {(int64_t)full};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.q_proj.bias",
                d.data(), 1, sh, LLAISYS_DTYPE_F32);
        }
        // K weight: [nkvh*dh, hs] = [16, 128] → column parallel → [8, 128]
        {
            size_t full_rows = meta.nkvh * meta.dh;  // 16
            auto d = make_fp32(full_rows * meta.hs);
            int64_t sh[] = {(int64_t)full_rows, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.k_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // V weight: same as K
        {
            size_t full_rows = meta.nkvh * meta.dh;
            auto d = make_fp32(full_rows * meta.hs);
            int64_t sh[] = {(int64_t)full_rows, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.v_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // O weight: [hs, nh*dh] = [128, 128] → row parallel → [128, 64]
        {
            auto d = make_fp32(meta.hs * meta.nh * meta.dh);
            int64_t sh[] = {(int64_t)meta.hs, (int64_t)(meta.nh * meta.dh)};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.o_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // gate weight: [di, hs] = [32, 128] → column parallel → [16, 128]
        {
            auto d = make_fp32(meta.di * meta.hs);
            int64_t sh[] = {(int64_t)meta.di, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.mlp.gate_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // up weight: [di, hs] = [32, 128] → column parallel → [16, 128]
        {
            auto d = make_fp32(meta.di * meta.hs);
            int64_t sh[] = {(int64_t)meta.di, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.mlp.up_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // down weight: [hs, di] = [128, 32] → row parallel → [128, 16]
        {
            auto d = make_fp32(meta.hs * meta.di);
            int64_t sh[] = {(int64_t)meta.hs, (int64_t)meta.di};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.mlp.down_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        // norm weights (不切分)
        {
            auto d = make_fp32(meta.hs);
            int64_t sh[] = {(int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.input_layernorm.weight",
                d.data(), 1, sh, LLAISYS_DTYPE_F32);
            llaisysQwen2LoadWeightByName(model, "model.layers.0.post_attention_layernorm.weight",
                d.data(), 1, sh, LLAISYS_DTYPE_F32);
        }

        // ── 验证维度 ──
        auto* w = llaisysQwen2ModelWeights(model);
        size_t local_nh = meta.nh / tp_size;
        size_t local_nkvh = meta.nkvh / tp_size;
        size_t local_di = meta.di / tp_size;

        // Q weight: 应为 [local_nh*dh, hs]
        {
            auto s = get_shape(w->attn_q_w[0]);
            if (s.size() != 2 || s[0] != local_nh * meta.dh || s[1] != meta.hs) {
                printf("  FAIL: rank=%d Q weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], local_nh * meta.dh, meta.hs);
                ok = 0;
            }
        }
        // Q bias: [local_nh*dh]
        {
            auto s = get_shape(w->attn_q_b[0]);
            if (s.size() != 1 || s[0] != local_nh * meta.dh) {
                printf("  FAIL: rank=%d Q bias shape=[%zu] expected=[%zu]\n",
                    tp_rank, s[0], local_nh * meta.dh);
                ok = 0;
            }
        }
        // K weight: [local_nkvh*dh, hs]
        {
            auto s = get_shape(w->attn_k_w[0]);
            if (s.size() != 2 || s[0] != local_nkvh * meta.dh || s[1] != meta.hs) {
                printf("  FAIL: rank=%d K weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], local_nkvh * meta.dh, meta.hs);
                ok = 0;
            }
        }
        // O weight: [hs, local_nh*dh] (row parallel, split dim 1)
        {
            auto s = get_shape(w->attn_o_w[0]);
            if (s.size() != 2 || s[0] != meta.hs || s[1] != local_nh * meta.dh) {
                printf("  FAIL: rank=%d O weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], meta.hs, local_nh * meta.dh);
                ok = 0;
            }
        }
        // gate weight: [local_di, hs]
        {
            auto s = get_shape(w->mlp_gate_w[0]);
            if (s.size() != 2 || s[0] != local_di || s[1] != meta.hs) {
                printf("  FAIL: rank=%d gate weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], local_di, meta.hs);
                ok = 0;
            }
        }
        // down weight: [hs, local_di] (row parallel)
        {
            auto s = get_shape(w->mlp_down_w[0]);
            if (s.size() != 2 || s[0] != meta.hs || s[1] != local_di) {
                printf("  FAIL: rank=%d down weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], meta.hs, local_di);
                ok = 0;
            }
        }
        // embed (不切分): [voc, hs]
        {
            auto s = get_shape(w->in_embed);
            if (s.size() != 2 || s[0] != meta.voc || s[1] != meta.hs) {
                printf("  FAIL: rank=%d embed shape=[%zu,%zu] expected=[%zu,%zu]\n",
                    tp_rank, s[0], s[1], meta.voc, meta.hs);
                ok = 0;
            }
        }
        // norm (不切分): [hs]
        {
            auto s = get_shape(w->attn_norm_w[0]);
            if (s.size() != 1 || s[0] != meta.hs) {
                printf("  FAIL: rank=%d attn_norm shape=[%zu] expected=[%zu]\n",
                    tp_rank, s[0], meta.hs);
                ok = 0;
            }
        }

        printf("  rank=%d: weight shapes verified\n", tp_rank);

        llaisysQwen2ModelDestroy(model);
    }

    // ── 回归测试: tp_size=1 ──
    {
        auto* model = llaisysQwen2ModelCreate(&meta, LLAISYS_DEVICE_CPU, nullptr, 0);
        assert(model != nullptr);
        assert(llaisysQwen2GetTpSize(model) == 1);
        assert(llaisysQwen2GetTpRank(model) == 0);

        // Q weight [128, 128] should stay [128, 128]
        {
            size_t full_rows = meta.nh * meta.dh;
            auto d = make_fp32(full_rows * meta.hs);
            int64_t sh[] = {(int64_t)full_rows, (int64_t)meta.hs};
            llaisysQwen2LoadWeightByName(model, "model.layers.0.self_attn.q_proj.weight",
                d.data(), 2, sh, LLAISYS_DTYPE_F32);
        }
        auto* w = llaisysQwen2ModelWeights(model);
        auto s = get_shape(w->attn_q_w[0]);
        if (s.size() != 2 || s[0] != meta.nh * meta.dh || s[1] != meta.hs) {
            printf("  FAIL: tp=1 regression Q weight shape=[%zu,%zu] expected=[%zu,%zu]\n",
                s[0], s[1], meta.nh * meta.dh, meta.hs);
            ok = 0;
        }
        printf("  tp_size=1 regression: OK\n");
        llaisysQwen2ModelDestroy(model);
    }

    if (ok) {
        printf("[tp-shard-smoke] ALL PASSED\n");
    } else {
        printf("[tp-shard-smoke] SOME TESTS FAILED\n");
        return 1;
    }
    return 0;
}
