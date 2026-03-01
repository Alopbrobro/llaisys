#include "llaisys/models/qwen2.h"
#include "../../ops/op.hpp"        
#include "../../utils/types.hpp"   
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>

using namespace llaisys;

// ==========================================
// 1. 辅助工具函数
// ==========================================

static inline tensor_t TO_CPP_TENSOR(llaisysTensor_t t) {
    if (!t) return nullptr;
    return tensor_t(reinterpret_cast<Tensor*>(t), [](Tensor*){});
}

// ==========================================
// 2. 模型结构体定义
// ==========================================

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device_type;
    int device_id;
    
    std::vector<tensor_t> resources;
    std::vector<std::vector<tensor_t>> kv_caches;

    // 缓冲区声明
    tensor_t input_ids_buf;
    tensor_t pos_ids_buf;
    
    tensor_t hidden_states; 
    tensor_t residual;      
    tensor_t norm_out;      
    
    tensor_t q, k, v;       
    tensor_t attn_out;      
    
    tensor_t gate, up, mlp_act; 
    tensor_t logits;        
    tensor_t next_token;    
    tensor_t max_val;       

    int64_t current_pos = 0;

    // 设备感知内存操作辅助函数
    void memcpyH2D(tensor_t dst, const void* host_src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst->data(), host_src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_async(
                dst->data(), host_src, bytes, LLAISYS_MEMCPY_H2D, nullptr);
        }
    }

    void memcpyD2H(void* host_dst, tensor_t src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(host_dst, src->data(), bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_sync(
                host_dst, src->data(), bytes, LLAISYS_MEMCPY_D2H);
        }
    }

    void memcpyOnDevice(void* dst, const void* src, size_t bytes) {
        if (device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, bytes);
        } else {
            core::context().setDevice(device_type, device_id);
            core::context().runtime().api()->memcpy_async(
                dst, src, bytes, LLAISYS_MEMCPY_D2D, nullptr);
        }
    }

    LlaisysQwen2Model(const LlaisysQwen2Meta* m, llaisysDeviceType_t dev, int dev_id)
        : meta(*m), device_type(dev), device_id(dev_id >= 0 ? dev_id : 0) {
        weights.in_embed = nullptr;
        weights.out_embed = nullptr;
        weights.out_norm_w = nullptr;
        
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer]();

        init_cache();
        init_buffers();
    }

    ~LlaisysQwen2Model() {
        delete[] weights.attn_norm_w; delete[] weights.attn_q_w; delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;    delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;    delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;  delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;    delete[] weights.mlp_down_w;
    }

    void init_cache() {
        std::vector<size_t> shape = {meta.maxseq, meta.nkvh, meta.dh};
        for (size_t i = 0; i < meta.nlayer; ++i) {
            auto k_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
            auto v_c = Tensor::create(shape, LLAISYS_DTYPE_F32, device_type, device_id);
            kv_caches.push_back({k_c, v_c});
        }
    }

    void init_buffers() {
        input_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        pos_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        
        hidden_states = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32, device_type, device_id);
        residual = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32, device_type, device_id);
        norm_out = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32, device_type, device_id);

        size_t q_dim = meta.nh * meta.dh;
        size_t kv_dim = meta.nkvh * meta.dh;
        
        q = Tensor::create({1, q_dim}, LLAISYS_DTYPE_F32, device_type, device_id);
        k = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32, device_type, device_id);
        v = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32, device_type, device_id);
        
        attn_out = Tensor::create({1, meta.nh, meta.dh}, LLAISYS_DTYPE_F32, device_type, device_id);
        
        gate = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32, device_type, device_id);
        up = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32, device_type, device_id);
        mlp_act = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32, device_type, device_id);
        
        logits = Tensor::create({1, meta.voc}, LLAISYS_DTYPE_F32, device_type, device_id);
        next_token = Tensor::create({1}, LLAISYS_DTYPE_I32, device_type, device_id);
        max_val = Tensor::create({1}, LLAISYS_DTYPE_F32, device_type, device_id);
    }
};

// ==========================================
// 3. C 接口实现
// ==========================================

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta) return nullptr;
    int dev_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
    return new LlaisysQwen2Model(meta, device, dev_id);
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) return -1;

    if (ntoken > 1) {
        model->current_pos = 0;
    }

    int64_t output_token = 0;

    for (size_t t = 0; t < ntoken; ++t) {
        int64_t token = token_ids[t];
        int64_t pos = model->current_pos;
        
        // H2D: 将 token 和 position 从 CPU 写入设备缓冲区
        model->memcpyH2D(model->input_ids_buf, &token, sizeof(int64_t));
        model->memcpyH2D(model->pos_ids_buf, &pos, sizeof(int64_t));

        // 1. Embedding
        ops::embedding(model->hidden_states, model->input_ids_buf, TO_CPP_TENSOR(model->weights.in_embed));

        // 2. Transformer Layers
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            // Save residual via pointer swap (zero-cost)
            std::swap(model->residual, model->hidden_states);

            // A. Pre-Norm (read from residual which now holds input)
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.attn_norm_w[i]), model->meta.epsilon);

            // B. QKV Linear
            ops::linear(model->q, model->norm_out, TO_CPP_TENSOR(model->weights.attn_q_w[i]), TO_CPP_TENSOR(model->weights.attn_q_b[i]));
            ops::linear(model->k, model->norm_out, TO_CPP_TENSOR(model->weights.attn_k_w[i]), TO_CPP_TENSOR(model->weights.attn_k_b[i]));
            ops::linear(model->v, model->norm_out, TO_CPP_TENSOR(model->weights.attn_v_w[i]), TO_CPP_TENSOR(model->weights.attn_v_b[i]));

            // C. RoPE
            auto q_3d = model->q->reshape({1, model->meta.nh, model->meta.dh});
            auto k_3d = model->k->reshape({1, model->meta.nkvh, model->meta.dh});
            auto v_3d = model->v->reshape({1, model->meta.nkvh, model->meta.dh});

            ops::rope(q_3d, q_3d, model->pos_ids_buf, model->meta.theta);
            ops::rope(k_3d, k_3d, model->pos_ids_buf, model->meta.theta);

            // D. Update KV Cache
            if (model->current_pos < (int64_t)model->meta.maxseq) {
                size_t bytes = model->meta.nkvh * model->meta.dh * 4;
                char* k_dst = (char*)model->kv_caches[i][0]->data() + model->current_pos * bytes;
                char* v_dst = (char*)model->kv_caches[i][1]->data() + model->current_pos * bytes;
                model->memcpyOnDevice(k_dst, k_3d->data(), bytes);
                model->memcpyOnDevice(v_dst, v_3d->data(), bytes);
            }

            // E. Self Attention
            float scale = 1.0f / std::sqrt((float)model->meta.dh);
            auto k_slice = model->kv_caches[i][0]->slice(0, 0, model->current_pos + 1);
            auto v_slice = model->kv_caches[i][1]->slice(0, 0, model->current_pos + 1);
            
            ops::self_attention(model->attn_out, q_3d, k_slice, v_slice, scale);

            // F. Output Projection
            auto attn_flat = model->attn_out->reshape({1, model->meta.hs});
            ops::linear(model->hidden_states, attn_flat, TO_CPP_TENSOR(model->weights.attn_o_w[i]), nullptr);

            // G. Residual Add 1
            ops::add(model->hidden_states, model->hidden_states, model->residual);

            // H. MLP Block
            std::swap(model->residual, model->hidden_states);
            ops::rms_norm(model->norm_out, model->residual, TO_CPP_TENSOR(model->weights.mlp_norm_w[i]), model->meta.epsilon);

            ops::linear(model->gate, model->norm_out, TO_CPP_TENSOR(model->weights.mlp_gate_w[i]), nullptr);
            ops::linear(model->up, model->norm_out, TO_CPP_TENSOR(model->weights.mlp_up_w[i]), nullptr);
            
            ops::swiglu(model->mlp_act, model->gate, model->up);
            
            ops::linear(model->hidden_states, model->mlp_act, TO_CPP_TENSOR(model->weights.mlp_down_w[i]), nullptr);

            // I. Residual Add 2
            ops::add(model->hidden_states, model->hidden_states, model->residual);
        }

        // 4. Final Norm
        ops::rms_norm(model->hidden_states, model->hidden_states, TO_CPP_TENSOR(model->weights.out_norm_w), model->meta.epsilon);

        // 5. LM Head
        ops::linear(model->logits, model->hidden_states, TO_CPP_TENSOR(model->weights.out_embed), nullptr);

        // 6. Argmax
        auto logits_2d = model->logits->reshape({1, model->meta.voc});
        ops::argmax(model->next_token, model->max_val, logits_2d);

        // D2H: 从设备读取 argmax 结果
        int32_t host_token;
        model->memcpyD2H(&host_token, model->next_token, sizeof(int32_t));
        output_token = host_token;
        
        model->current_pos++;
    }
    return output_token;
}

__export void llaisysQwen2LoadWeightByName(struct LlaisysQwen2Model* model, const char* name, void* data, int ndim, int64_t* shape, int dtype) {
    if (!model) return;

    std::vector<size_t> shape_vec;
    size_t numel = 1;
    for (int i = 0; i < ndim; ++i) {
        shape_vec.push_back((size_t)shape[i]);
        numel *= shape[i];
    }
    
    // 在目标设备上创建 tensor，通过 load() 完成 H2D 传输
    auto tensor = llaisys::Tensor::create(shape_vec, (llaisysDataType_t)dtype, model->device_type, model->device_id);
    tensor->load(data);

    model->resources.push_back(tensor);
    llaisysTensor_t t_handle = reinterpret_cast<llaisysTensor_t>(tensor.get()); 
    
    std::string key(name);
    if (key == "model.embed_tokens.weight") model->weights.in_embed = t_handle;
    else if (key == "model.norm.weight") model->weights.out_norm_w = t_handle;
    else if (key == "lm_head.weight") model->weights.out_embed = t_handle;
    else if (key.find("model.layers.") == 0) {
        size_t first_dot = 13;
        size_t second_dot = key.find('.', first_dot);
        int layer_idx = std::stoi(key.substr(first_dot, second_dot - first_dot));
        std::string suffix = key.substr(second_dot + 1);
        
        if (suffix == "input_layernorm.weight") model->weights.attn_norm_w[layer_idx] = t_handle;
        else if (suffix == "post_attention_layernorm.weight") model->weights.mlp_norm_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.q_proj.weight") model->weights.attn_q_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.weight") model->weights.attn_k_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.weight") model->weights.attn_v_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.o_proj.weight") model->weights.attn_o_w[layer_idx] = t_handle;
        else if (suffix == "self_attn.q_proj.bias") model->weights.attn_q_b[layer_idx] = t_handle;
        else if (suffix == "self_attn.k_proj.bias") model->weights.attn_k_b[layer_idx] = t_handle;
        else if (suffix == "self_attn.v_proj.bias") model->weights.attn_v_b[layer_idx] = t_handle;
        else if (suffix == "mlp.gate_proj.weight") model->weights.mlp_gate_w[layer_idx] = t_handle;
        else if (suffix == "mlp.up_proj.weight") model->weights.mlp_up_w[layer_idx] = t_handle;
        else if (suffix == "mlp.down_proj.weight") model->weights.mlp_down_w[layer_idx] = t_handle;
    }
}

} // extern "C"