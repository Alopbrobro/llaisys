// Phase D smoke test: 验证 TP 下 KV-Cache save/restore 和 Attention 分片
// 1. tp=1: infer → save → infer → restore → infer (一致性)
// 2. tp=2: infer → save → restore → infer (不崩溃)
// 3. tp mismatch: save(tp=2,rank=0) → restore(tp=1) → 应拒绝

#include "llaisys/models/qwen2.h"
#include "llaisys/distributed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

static std::vector<float> make_fp32(size_t numel, float val = 0.01f) {
    return std::vector<float>(numel, val);
}

static void load_all_weights(LlaisysQwen2Model* model, const LlaisysQwen2Meta& meta) {
    // embed_tokens
    { auto d = make_fp32(meta.voc * meta.hs); int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
      llaisysQwen2LoadWeightByName(model, "model.embed_tokens.weight", d.data(), 2, sh, LLAISYS_DTYPE_F32); }
    // norm
    { auto d = make_fp32(meta.hs, 1.0f); int64_t sh[] = {(int64_t)meta.hs};
      llaisysQwen2LoadWeightByName(model, "model.norm.weight", d.data(), 1, sh, LLAISYS_DTYPE_F32); }
    // lm_head
    { auto d = make_fp32(meta.voc * meta.hs); int64_t sh[] = {(int64_t)meta.voc, (int64_t)meta.hs};
      llaisysQwen2LoadWeightByName(model, "lm_head.weight", d.data(), 2, sh, LLAISYS_DTYPE_F32); }

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
        load1d("input_layernorm.weight", meta.hs);
        load1d("post_attention_layernorm.weight", meta.hs);
        load2d("self_attn.q_proj.weight", q_out, meta.hs);
        load1d("self_attn.q_proj.bias", q_out);
        load2d("self_attn.k_proj.weight", kv_out, meta.hs);
        load1d("self_attn.k_proj.bias", kv_out);
        load2d("self_attn.v_proj.weight", kv_out, meta.hs);
        load1d("self_attn.v_proj.bias", kv_out);
        load2d("self_attn.o_proj.weight", meta.hs, q_out);
        load2d("mlp.gate_proj.weight", meta.di, meta.hs);
        load2d("mlp.up_proj.weight", meta.di, meta.hs);
        load2d("mlp.down_proj.weight", meta.hs, meta.di);
    }
}

int main() {
    LlaisysQwen2Meta meta = {};
    meta.dtype = LLAISYS_DTYPE_F32;
    meta.nlayer = 2;
    meta.hs = 64;
    meta.nh = 8;
    meta.nkvh = 2;
    meta.dh = 8;
    meta.di = 32;
    meta.maxseq = 32;
    meta.voc = 50;
    meta.epsilon = 1e-6f;
    meta.theta = 10000.0f;
    meta.end_token = 0;

    int ok = 1;

    // ── Test 1: tp=1 cache save/restore consistency ──
    printf("[tp-cache-smoke] Test 1: tp=1 cache save/restore ...\n");
    {
        auto* model = llaisysQwen2ModelCreate(&meta, LLAISYS_DEVICE_CPU, nullptr, 0);
        load_all_weights(model, meta);

        // Prefill 3 tokens
        int64_t tokens[] = {1, 2, 3};
        llaisysQwen2ModelInfer(model, tokens, 3);
        assert(llaisysQwen2GetCachePos(model) == 3);
        auto* snap = llaisysQwen2SaveCache(model);
        assert(snap != nullptr);

        // Generate 1 more token
        int64_t tok4[] = {10};
        int64_t out_a = llaisysQwen2ModelInfer(model, tok4, 1);
        assert(llaisysQwen2GetCachePos(model) == 4);

        // Restore cache to pos=3
        llaisysQwen2RestoreCache(model, snap);
        assert(llaisysQwen2GetCachePos(model) == 3);

        // Generate again from same state → should produce same token
        int64_t out_b = llaisysQwen2ModelInfer(model, tok4, 1);
        if (out_a != out_b) {
            printf("  FAIL: restore inconsistency (out_a=%lld, out_b=%lld)\n",
                   (long long)out_a, (long long)out_b);
            ok = 0;
        } else {
            printf("  tp=1 save→restore→infer: consistent (token=%lld)\n", (long long)out_a);
        }

        llaisysQwen2DestroyCacheSnapshot(snap);
        llaisysQwen2ModelDestroy(model);
    }

    // ── Test 2: tp=2 cache save/restore (no crash) ──
    printf("[tp-cache-smoke] Test 2: tp=2 cache save/restore ...\n");
    {
        LlaisysDistConfig cfg = {};
        cfg.backend = LLAISYS_DIST_BACKEND_MOCK;
        cfg.world_size = 2;
        cfg.rank = 0;
        cfg.local_device = 0;
        auto* comm = llaisysDistCommCreate(cfg);

        auto* model = llaisysQwen2ModelCreateTP(&meta, LLAISYS_DEVICE_CPU, 0, 2, 0);
        llaisysQwen2SetComm(model, comm);
        load_all_weights(model, meta);

        int64_t tokens[] = {1, 2, 3};
        llaisysQwen2ModelInfer(model, tokens, 3);
        assert(llaisysQwen2GetCachePos(model) == 3);

        auto* snap = llaisysQwen2SaveCache(model);
        assert(snap != nullptr);

        // Generate one more
        int64_t tok4[] = {5};
        llaisysQwen2ModelInfer(model, tok4, 1);
        assert(llaisysQwen2GetCachePos(model) == 4);

        // Restore
        llaisysQwen2RestoreCache(model, snap);
        assert(llaisysQwen2GetCachePos(model) == 3);

        // Continue from restored state
        int64_t out = llaisysQwen2ModelInfer(model, tok4, 1);
        printf("  tp=2,rank=0 cache restore→infer: token=%lld\n", (long long)out);
        if (out < 0 || (size_t)out >= meta.voc) {
            printf("  FAIL: invalid token\n");
            ok = 0;
        }

        llaisysQwen2DestroyCacheSnapshot(snap);
        llaisysQwen2ModelDestroy(model);
        llaisysDistCommDestroy(comm);
        printf("  Test 2: PASSED\n");
    }

    // ── Test 3: TP mismatch rejection ──
    printf("[tp-cache-smoke] Test 3: TP mismatch rejection ...\n");
    {
        // Create tp=2 model, save cache
        LlaisysDistConfig cfg = {};
        cfg.backend = LLAISYS_DIST_BACKEND_MOCK;
        cfg.world_size = 2;
        cfg.rank = 0;
        cfg.local_device = 0;
        auto* comm = llaisysDistCommCreate(cfg);

        auto* model_tp2 = llaisysQwen2ModelCreateTP(&meta, LLAISYS_DEVICE_CPU, 0, 2, 0);
        llaisysQwen2SetComm(model_tp2, comm);
        load_all_weights(model_tp2, meta);

        int64_t tokens[] = {1, 2};
        llaisysQwen2ModelInfer(model_tp2, tokens, 2);
        auto* snap = llaisysQwen2SaveCache(model_tp2);
        assert(snap != nullptr);

        // Create tp=1 model, try to restore tp=2 snapshot → should be rejected
        auto* model_tp1 = llaisysQwen2ModelCreate(&meta, LLAISYS_DEVICE_CPU, nullptr, 0);
        load_all_weights(model_tp1, meta);

        llaisysQwen2RestoreCache(model_tp1, snap);
        // pos should remain 0 (restore was rejected)
        int64_t pos = llaisysQwen2GetCachePos(model_tp1);
        if (pos != 0) {
            printf("  FAIL: TP mismatch should have been rejected (pos=%lld)\n", (long long)pos);
            ok = 0;
        } else {
            printf("  TP mismatch correctly rejected\n");
        }

        llaisysQwen2DestroyCacheSnapshot(snap);
        llaisysQwen2ModelDestroy(model_tp2);
        llaisysQwen2ModelDestroy(model_tp1);
        llaisysDistCommDestroy(comm);
        printf("  Test 3: PASSED\n");
    }

    if (ok) {
        printf("[tp-cache-smoke] ALL PASSED\n");
    } else {
        printf("[tp-cache-smoke] SOME TESTS FAILED\n");
        return 1;
    }
    return 0;
}
