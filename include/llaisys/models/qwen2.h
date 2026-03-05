#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h" // 确保包含基础类型定义 (如 llaisysDataType_t, __export, __C)

__C {
    // 模型元数据结构体
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    // 权重结构体 (用于内部管理或调试)
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // model.norm.weight
        llaisysTensor_t *attn_norm_w; // input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    // 不透明的模型句柄
    struct LlaisysQwen2Model;

    // 不透明的 KV-Cache 快照句柄
    struct LlaisysQwen2CacheSnapshot;

    // 不透明的 KV-Cache 前缀树池句柄
    struct LlaisysKVCachePool;

    // 创建模型实例
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const struct LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    // 销毁模型实例
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    // 获取权重结构体指针 (可选)
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // 按名称加载权重 (Python 包装器使用此接口)
    __export void llaisysQwen2LoadWeightByName(struct LlaisysQwen2Model * model, const char * name, void * data, int ndim, int64_t * shape, int dtype);

    // 执行推理
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    // 执行推理 (带采样参数)
    __export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken,
                                                  float temperature, int top_k, float top_p);

    // 重置 KV-Cache 位置 (不重新加载权重)
    __export void llaisysQwen2ResetCache(struct LlaisysQwen2Model * model);

    // ==========================================
    // Phase 4: KV-Cache 高级接口
    // ==========================================

    // 保存当前 KV-Cache 快照 (深拷贝到 CPU)
    __export struct LlaisysQwen2CacheSnapshot *llaisysQwen2SaveCache(struct LlaisysQwen2Model * model);

    // 从快照恢复 KV-Cache (从 CPU 拷贝回设备)
    __export void llaisysQwen2RestoreCache(struct LlaisysQwen2Model * model, struct LlaisysQwen2CacheSnapshot * snapshot);

    // 截断 KV-Cache 到指定位置 (pos 必须 <= current_pos)
    __export void llaisysQwen2TruncateCache(struct LlaisysQwen2Model * model, int64_t pos);

    // 获取当前 KV-Cache 位置
    __export int64_t llaisysQwen2GetCachePos(struct LlaisysQwen2Model * model);

    // 销毁 KV-Cache 快照
    __export void llaisysQwen2DestroyCacheSnapshot(struct LlaisysQwen2CacheSnapshot * snapshot);

    // ==========================================
    // Phase 4: 前缀树 KV-Cache 池
    // ==========================================

    // 创建 KV-Cache 前缀树池
    __export struct LlaisysKVCachePool *llaisysKVCachePoolCreate(void);

    // 销毁 KV-Cache 前缀树池 (释放所有存储的快照)
    __export void llaisysKVCachePoolDestroy(struct LlaisysKVCachePool * pool);

    // 向池中插入快照 (池获取快照所有权, 调用者不再拥有)
    __export void llaisysKVCachePoolInsert(struct LlaisysKVCachePool * pool, int64_t * tokens, size_t len, struct LlaisysQwen2CacheSnapshot * snapshot);

    // 查找最长前缀匹配, 返回对应快照 (不转移所有权), match_len 输出匹配长度
    __export struct LlaisysQwen2CacheSnapshot *llaisysKVCachePoolLookup(struct LlaisysKVCachePool * pool, int64_t * tokens, size_t len, size_t * match_len);

    // 清空池中所有条目
    __export void llaisysKVCachePoolClear(struct LlaisysKVCachePool * pool);
}
#endif // LLAISYS_MODELS_QWEN2_H