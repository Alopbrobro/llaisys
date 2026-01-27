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
}
#endif // LLAISYS_MODELS_QWEN2_H