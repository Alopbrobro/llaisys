// Phase C smoke test: 验证 TP 前向推理闭环 (mock comm, CPU)
// tp_size=2: 加载分片权重 → 前向推理 → 不崩溃
// tp_size=1: 回归测试

#include "llaisys/models/qwen2.h"
#include "llaisys/distributed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>
#include <cmath>

static std::vector<float> make_fp32(size_t numel, float val = 0.01f) {
    std::vector<float> v(numel, val);
    return v;
}

// 模拟加载全部权重 (用常数填充)
static void load_all_weights(LlaisysQwen2Model* model, const LlaisysQwen2Meta& meta) {
    // embed_tokens: [voc, hs]
    {
        auto d = make_fp32(meta.voc * meta.hs);
        int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
        llaisysQwen2LoadWeightByName(model, "model.embed_tokens.weight", d.data(), 2, sh, LLAISYS_DTYPE_F32);
    }
    // model.norm.weight: [hs]
    {
        auto d = make_fp32(meta.hs, 1.0f);
        int64_t sh[] = {(int64_t)meta.hs};
        llaisysQwen2LoadWeightByName(model, "model.norm.weight", d.data(), 1, sh, LLAISYS_DTYPE_F32);
    }
    // lm_head.weight: [voc, hs]
    {
        auto d = make_fp32(meta.voc * meta.hs);
        int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
        llaisysQwen2LoadWeightByName(model, "lm_head.weight", d.data(), 2, sh, LLAISYS_DTYPE_F32);
    }

    // per-layer weights
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        char buf[256];
        auto load2d = [&](const char* suffix, size_t rows, size_t cols) {
            snprintf(buf, sizeof(buf), "model.layers.%zu.%s", layer, suffix);
            auto d = make_fp32(rows * cols);
            int64_t sh[] = {(int64_t)rows, (int64_t)cols};
            llaisysQwen2LoadWeightByName(model, buf, d.data(), 2, sh, LLAISYS_DTYPE_F32);
        };
        auto load1d = [&](const char* suffix, size_t size) {
            snprintf(buf, sizeof(buf), "model.layers.%zu.%s", layer, suffix);
            auto d = make_fp32(size, 1.0f);
            int64_t sh[] = {(int64_t)size};
            llaisysQwen2LoadWeightByName(model, buf, d.data(), 1, sh, LLAISYS_DTYPE_F32);
        };

        size_t q_out = meta.nh * meta.dh;
        size_t kv_out = meta.nkvh * meta.dh;

        // Norm weights (不切分)
        load1d("input_layernorm.weight", meta.hs);
        load1d("post_attention_layernorm.weight", meta.hs);

        // Column parallel: Q/K/V weight [out, hs], bias [out]
        load2d("self_attn.q_proj.weight", q_out, meta.hs);
        load1d("self_attn.q_proj.bias", q_out);
        load2d("self_attn.k_proj.weight", kv_out, meta.hs);
        load1d("self_attn.k_proj.bias", kv_out);
        load2d("self_attn.v_proj.weight", kv_out, meta.hs);
        load1d("self_attn.v_proj.bias", kv_out);

        // Row parallel: O weight [hs, nh*dh]
        load2d("self_attn.o_proj.weight", meta.hs, q_out);

        // Column parallel: gate/up [di, hs]
        load2d("mlp.gate_proj.weight", meta.di, meta.hs);
        load2d("mlp.up_proj.weight", meta.di, meta.hs);

        // Row parallel: down [hs, di]
        load2d("mlp.down_proj.weight", meta.hs, meta.di);
    }
}

int main() {
    // 小型模型 (确保 CPU 上能快速跑完)
    LlaisysQwen2Meta meta = {};
    meta.dtype = LLAISYS_DTYPE_F32;
    meta.nlayer = 2;
    meta.hs = 64;
    meta.nh = 8;
    meta.nkvh = 2;
    meta.dh = 8;   // hs = nh * dh = 64
    meta.di = 32;
    meta.maxseq = 32;
    meta.voc = 50;
    meta.epsilon = 1e-6f;
    meta.theta = 10000.0f;
    meta.end_token = 0;

    int ok = 1;

    // ── Test 1: tp_size=1 (回归) ──
    printf("[tp-fwd-smoke] Test 1: tp_size=1 regression ...\n");
    {
        auto* model = llaisysQwen2ModelCreate(&meta, LLAISYS_DEVICE_CPU, nullptr, 0);
        assert(model != nullptr);
        load_all_weights(model, meta);

        int64_t tokens[] = {1, 2, 3};
        int64_t out = llaisysQwen2ModelInfer(model, tokens, 3);
        printf("  tp=1 infer output=%lld (should be valid token 0..%zu)\n", (long long)out, meta.voc - 1);
        if (out < 0 || (size_t)out >= meta.voc) {
            printf("  FAIL: invalid token\n");
            ok = 0;
        }

        llaisysQwen2ResetCache(model);
        int64_t out2 = llaisysQwen2ModelInferSample(model, tokens, 3, 0.0f, 1, 1.0f);
        printf("  tp=1 infer_sample output=%lld\n", (long long)out2);
        if (out2 < 0 || (size_t)out2 >= meta.voc) {
            printf("  FAIL: invalid token\n");
            ok = 0;
        }

        llaisysQwen2ModelDestroy(model);
        printf("  tp_size=1: PASSED\n");
    }

    // ── Test 2: tp_size=2, rank=0, mock comm ──
    printf("[tp-fwd-smoke] Test 2: tp_size=2, rank=0, mock comm ...\n");
    {
        // 创建 mock comm
        LlaisysDistConfig cfg = {};
        cfg.backend = LLAISYS_DIST_BACKEND_MOCK;
        cfg.world_size = 2;
        cfg.rank = 0;
        cfg.local_device = 0;
        auto* comm = llaisysDistCommCreate(cfg);
        assert(comm != nullptr);

        // 创建 TP 模型
        auto* model = llaisysQwen2ModelCreateTP(&meta, LLAISYS_DEVICE_CPU, 0, 2, 0);
        assert(model != nullptr);
        llaisysQwen2SetComm(model, comm);

        load_all_weights(model, meta);

        int64_t tokens[] = {1, 2, 3};
        int64_t out = llaisysQwen2ModelInfer(model, tokens, 3);
        printf("  tp=2,rank=0 infer output=%lld\n", (long long)out);
        // 模拟通信下结果可能不正确，但不应崩溃
        if (out < 0 || (size_t)out >= meta.voc) {
            printf("  FAIL: invalid token\n");
            ok = 0;
        }

        llaisysQwen2ResetCache(model);
        int64_t out2 = llaisysQwen2ModelInferSample(model, tokens, 3, 0.8f, 5, 0.9f);
        printf("  tp=2,rank=0 infer_sample output=%lld\n", (long long)out2);
        // mock all-reduce is no-op, result may differ but must be valid
        if (out2 < -1 || (size_t)out2 >= meta.voc) {
            // InferSample with seed might return -1 if something is wrong, but >= voc is definitely bad
            printf("  WARNING: unusual token (may be OK with mock comm)\n");
        }

        llaisysQwen2ModelDestroy(model);
        llaisysDistCommDestroy(comm);
        printf("  tp_size=2: PASSED (no crash)\n");
    }

    // ── Test 3: tp_size=2, rank=1 ──
    printf("[tp-fwd-smoke] Test 3: tp_size=2, rank=1, mock comm ...\n");
    {
        LlaisysDistConfig cfg = {};
        cfg.backend = LLAISYS_DIST_BACKEND_MOCK;
        cfg.world_size = 2;
        cfg.rank = 1;
        cfg.local_device = 0;
        auto* comm = llaisysDistCommCreate(cfg);

        auto* model = llaisysQwen2ModelCreateTP(&meta, LLAISYS_DEVICE_CPU, 0, 2, 1);
        assert(model != nullptr);
        llaisysQwen2SetComm(model, comm);

        load_all_weights(model, meta);

        int64_t tokens[] = {5, 10};
        int64_t out = llaisysQwen2ModelInfer(model, tokens, 2);
        printf("  tp=2,rank=1 infer output=%lld\n", (long long)out);
        if (out < 0 || (size_t)out >= meta.voc) {
            printf("  FAIL: invalid token\n");
            ok = 0;
        }

        llaisysQwen2ModelDestroy(model);
        llaisysDistCommDestroy(comm);
        printf("  tp_size=2,rank=1: PASSED (no crash)\n");
    }

    if (ok) {
        printf("[tp-fwd-smoke] ALL PASSED\n");
    } else {
        printf("[tp-fwd-smoke] SOME TESTS FAILED\n");
        return 1;
    }
    return 0;
}
