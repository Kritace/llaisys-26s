#pragma once
#include "../../tensor/tensor.hpp"
#include "llaisys.h"
#include <vector>

namespace llaisys::models {

struct Qwen2LayerWeights {
    tensor_t attn_norm_w;   // input_layernorm.weight
    tensor_t attn_q_w;      // self_attn.q_proj.weight
    tensor_t attn_q_b;      // self_attn.q_proj.bias
    tensor_t attn_k_w;      // self_attn.k_proj.weight
    tensor_t attn_k_b;      // self_attn.k_proj.bias
    tensor_t attn_v_w;      // self_attn.v_proj.weight
    tensor_t attn_v_b;      // self_attn.v_proj.bias
    tensor_t attn_o_w;      // self_attn.o_proj.weight
    
    tensor_t mlp_norm_w;    // post_attention_layernorm.weight
    tensor_t mlp_gate_w;    // mlp.gate_proj.weight
    tensor_t mlp_up_w;      // mlp.up_proj.weight
    tensor_t mlp_down_w;    // mlp.down_proj.weight
};

struct Qwen2Weights {
    tensor_t in_embed;      // model.embed_tokens.weight  [vocab_size, hidden_size]
    tensor_t out_embed;     // lm_head.weight             [vocab_size, hidden_size]
    tensor_t out_norm_w;    // model.norm.weight          [hidden_size]
    std::vector<Qwen2LayerWeights> layers;  // 每层的权重
};

struct Qwen2KVCache {
    std::vector<tensor_t> k_cache;  // 每层的 K 缓存
    std::vector<tensor_t> v_cache;  // 每层的 V 缓存
    size_t step;                     // 当前已生成的 token 数
};

struct Qwen2Config {
    size_t nlayer;   // 层数
    size_t hs;       // hidden_size
    size_t nh;       // num_attention_heads
    size_t nkvh;     // num_key_value_heads
    size_t dh;       // head_dim = hidden_size / num_attention_heads
    size_t di;       // intermediate_size
    size_t voc;      // vocab_size
    float epsilon;   // rms norm eps
    float theta;     // rope theta
};

class Qwen2Model {
public:
    Qwen2Model(const Qwen2Config &config, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    // 加载权重
    void load_weights(const Qwen2Weights &weights);

    // 推理一步: 输入当前 token → 输出下一个 token 的 logits
    tensor_t forward(tensor_t input_ids, Qwen2KVCache &cache);

    // 获取配置
    const Qwen2Config &config() const { return config_; }

private:
    // 单层前向传播
    tensor_t forward_layer(tensor_t hidden, const Qwen2LayerWeights &w, 
                           Qwen2KVCache &cache, size_t layer_idx);

    Qwen2Config config_;
    Qwen2Weights weights_;
    llaisysDeviceType_t device_;
    int device_id_;
};

} // namespace llaisys::models