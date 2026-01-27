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

static void element_wise_add(tensor_t a, tensor_t b) {
    if (!a || !b) return;
    float* a_ptr = reinterpret_cast<float*>(a->data());
    const float* b_ptr = reinterpret_cast<const float*>(b->data());
    size_t size = a->numel();
    for (size_t i = 0; i < size; ++i) {
        a_ptr[i] += b_ptr[i];
    }
}

// ==========================================
// 2. 模型结构体定义
// ==========================================

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    std::vector<tensor_t> resources;
    std::vector<std::vector<tensor_t>> kv_caches;

    // 缓冲区声明
    tensor_t input_ids_buf;
    tensor_t pos_ids_buf;
    
    // 【修复】2D 缓冲区 [1, dim]
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

    LlaisysQwen2Model(const LlaisysQwen2Meta* m) : meta(*m) {
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
            auto k_c = Tensor::create(shape, LLAISYS_DTYPE_F32);
            auto v_c = Tensor::create(shape, LLAISYS_DTYPE_F32);
            kv_caches.push_back({k_c, v_c});
        }
    }

    void init_buffers() {
        input_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64);
        pos_ids_buf = Tensor::create({1}, LLAISYS_DTYPE_I64);
        
        // 【核心修复】这里全部改为 2D {1, dim}，而不是 {1, 1, dim}
        // 这样 Linear 算子读取 shape[1] 时才是正确的特征维度
        hidden_states = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32);
        residual = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32);
        norm_out = Tensor::create({1, meta.hs}, LLAISYS_DTYPE_F32);

        size_t q_dim = meta.nh * meta.dh;
        size_t kv_dim = meta.nkvh * meta.dh;
        
        q = Tensor::create({1, q_dim}, LLAISYS_DTYPE_F32);
        k = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32);
        v = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32);
        
        // attn_out 本来就是 Attention 的输出，必须是 3D，稍后 reshape
        attn_out = Tensor::create({1, meta.nh, meta.dh}, LLAISYS_DTYPE_F32);
        
        gate = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32);
        up = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32);
        mlp_act = Tensor::create({1, meta.di}, LLAISYS_DTYPE_F32);
        
        logits = Tensor::create({1, meta.voc}, LLAISYS_DTYPE_F32);
        next_token = Tensor::create({1}, LLAISYS_DTYPE_I32);
        max_val = Tensor::create({1}, LLAISYS_DTYPE_F32);
    }
};

// ==========================================
// 3. C 接口实现
// ==========================================

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta) return nullptr;
    return new LlaisysQwen2Model(meta);
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
        
        ((int64_t*)model->input_ids_buf->data())[0] = token;
        ((int64_t*)model->pos_ids_buf->data())[0] = model->current_pos;

        // 1. Embedding
        ops::embedding(model->hidden_states, model->input_ids_buf, TO_CPP_TENSOR(model->weights.in_embed));

        // 2. Transformer Layers
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            std::memcpy(model->residual->data(), model->hidden_states->data(), model->hidden_states->numel() * 4);

            // A. Pre-Norm
            ops::rms_norm(model->norm_out, model->hidden_states, TO_CPP_TENSOR(model->weights.attn_norm_w[i]), model->meta.epsilon);

            // B. QKV Linear (输入 2D, 输出 2D)
            ops::linear(model->q, model->norm_out, TO_CPP_TENSOR(model->weights.attn_q_w[i]), TO_CPP_TENSOR(model->weights.attn_q_b[i]));
            ops::linear(model->k, model->norm_out, TO_CPP_TENSOR(model->weights.attn_k_w[i]), TO_CPP_TENSOR(model->weights.attn_k_b[i]));
            ops::linear(model->v, model->norm_out, TO_CPP_TENSOR(model->weights.attn_v_w[i]), TO_CPP_TENSOR(model->weights.attn_v_b[i]));

            // C. RoPE
            // 【关键】Reshape 2D -> 3D 以适配 RoPE 算子
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
                std::memcpy(k_dst, k_3d->data(), bytes);
                std::memcpy(v_dst, v_3d->data(), bytes);
            }

            // E. Self Attention
            float scale = 1.0f / std::sqrt((float)model->meta.dh);
            auto k_slice = model->kv_caches[i][0]->slice(0, 0, model->current_pos + 1);
            auto v_slice = model->kv_caches[i][1]->slice(0, 0, model->current_pos + 1);
            
            // 计算 Attention，输出为 3D
            ops::self_attention(model->attn_out, q_3d, k_slice, v_slice, scale);

            // F. Output Projection
            // 【关键】Reshape 3D -> 2D 以适配 Output Linear 算子
            auto attn_flat = model->attn_out->reshape({1, model->meta.hs});
            ops::linear(model->hidden_states, attn_flat, TO_CPP_TENSOR(model->weights.attn_o_w[i]), nullptr);

            // G. Residual Add 1
            element_wise_add(model->hidden_states, model->residual);

            // H. MLP Block
            std::memcpy(model->residual->data(), model->hidden_states->data(), model->hidden_states->numel() * 4); 
            ops::rms_norm(model->norm_out, model->hidden_states, TO_CPP_TENSOR(model->weights.mlp_norm_w[i]), model->meta.epsilon);

            ops::linear(model->gate, model->norm_out, TO_CPP_TENSOR(model->weights.mlp_gate_w[i]), nullptr);
            ops::linear(model->up, model->norm_out, TO_CPP_TENSOR(model->weights.mlp_up_w[i]), nullptr);
            
            ops::swiglu(model->mlp_act, model->gate, model->up);
            
            ops::linear(model->hidden_states, model->mlp_act, TO_CPP_TENSOR(model->weights.mlp_down_w[i]), nullptr);

            // I. Residual Add 2
            element_wise_add(model->hidden_states, model->residual);
        }

        // 4. Final Norm
        ops::rms_norm(model->hidden_states, model->hidden_states, TO_CPP_TENSOR(model->weights.out_norm_w), model->meta.epsilon);

        // 5. LM Head
        ops::linear(model->logits, model->hidden_states, TO_CPP_TENSOR(model->weights.out_embed), nullptr);

        // 6. Argmax
        auto logits_2d = model->logits->reshape({1, model->meta.voc});
        ops::argmax(model->next_token, model->max_val, logits_2d);

        output_token = ((int*)model->next_token->data())[0];
        
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
    
    // 即使 Python 端传了 Float32 数据，我们也创建 F32 Tensor
    auto tensor = llaisys::Tensor::create(shape_vec, (llaisysDataType_t)dtype);
    size_t element_size = llaisys::utils::dsize((llaisysDataType_t)dtype);
    std::memcpy(tensor->data(), data, numel * element_size);

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