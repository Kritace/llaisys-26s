#include "qwen2.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../core/llaisys_core.hpp"   // core::context()：设备感知的 memcpy

#include <cmath>

namespace llaisys::models {

tensor_t Qwen2Model::forward_layer(
    tensor_t hidden, const Qwen2LayerWeights &w,
    Qwen2KVCache &cache, size_t layer_idx) 
{
    // 1. Pre-attention RMS Norm
    tensor_t normed = Tensor::create(hidden->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    ops::rms_norm(normed, hidden, w.attn_norm_w, config_.epsilon);

    // 2. QKV projections
    tensor_t q = Tensor::create({1, config_.nh * config_.dh}, LLAISYS_DTYPE_F32, device_, device_id_);
    tensor_t k = Tensor::create({1, config_.nkvh * config_.dh}, LLAISYS_DTYPE_F32, device_, device_id_);
    tensor_t v = Tensor::create({1, config_.nkvh * config_.dh}, LLAISYS_DTYPE_F32, device_, device_id_);
    
    ops::linear(q, normed, w.attn_q_w, w.attn_q_b);
    ops::linear(k, normed, w.attn_k_w, w.attn_k_b);
    ops::linear(v, normed, w.attn_v_w, w.attn_v_b);

    // 3. Reshape Q, K, V to [1, n_heads, head_dim] for RoPE
    //    q: [1, nh*dh] → [1, nh, dh]
    //    k: [1, nkvh*dh] → [1, nkvh, dh]
    //    v: [1, nkvh*dh] → [1, nkvh, dh]
    tensor_t q_3d = q->view({1, config_.nh, config_.dh});
    tensor_t k_3d = k->view({1, config_.nkvh, config_.dh});
    tensor_t v_3d = v->view({1, config_.nkvh, config_.dh});

    // 4. Apply RoPE (只对 Q 和 K 做)
    // position ids: 当前步数 cache.step
    tensor_t pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
    std::int64_t pos = static_cast<std::int64_t>(cache.step);
    pos_ids->load(&pos);
    
    tensor_t q_rope = Tensor::create(q_3d->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    tensor_t k_rope = Tensor::create(k_3d->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    ops::rope(q_rope, q_3d, pos_ids, config_.theta);
    ops::rope(k_rope, k_3d, pos_ids, config_.theta);

    // 5. KV Cache
    // 5a. 首次调用时初始化所有层的缓存（max_seq=4096）
    if (cache.k_cache.empty()) {
        cache.k_cache.resize(config_.nlayer);
        cache.v_cache.resize(config_.nlayer);
        size_t max_seq = 4096; // max token

        // 逐层创建KV cache
        for (size_t i = 0; i < config_.nlayer; i++) {
            // k_cache[layer] 可以存 max_seq 个 token
            // 每个 token 有 nkvh 个 K/V 头，每个头 dh 维
            cache.k_cache[i] = Tensor::create(
                {max_seq, config_.nkvh, config_.dh},
                LLAISYS_DTYPE_F32, device_, device_id_);
            cache.v_cache[i] = Tensor::create(
                {max_seq, config_.nkvh, config_.dh},
                LLAISYS_DTYPE_F32, device_, device_id_);
        }
    }

    // 5b. 将当前步的 K_rope 和 V 写入缓存（device→device 拷贝）
    //     slice(0, step, step+1) → [1, nkvh, dh]，指向缓存中 step 位置
    //     注意：load() 是 host→device（H2D），不能用于 device→device，
    //     否则会把显存指针当主机源读 → GPU 下段错误/写坏。
    tensor_t k_dst = cache.k_cache[layer_idx]->slice(0, cache.step, cache.step + 1);
    tensor_t v_dst = cache.v_cache[layer_idx]->slice(0, cache.step, cache.step + 1);
    const size_t kv_bytes = k_dst->numel() * k_dst->elementSize();
    llaisys::core::context().runtime().api()->memcpy_sync(
        k_dst->data(), k_rope->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    llaisys::core::context().runtime().api()->memcpy_sync(
        v_dst->data(), v_3d->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    // 5c. 取出完整缓存 [0, step+1) 给 attention 用
    tensor_t K_all = cache.k_cache[layer_idx]->slice(0, 0, cache.step + 1);
    tensor_t V_all = cache.v_cache[layer_idx]->slice(0, 0, cache.step + 1);

    // 6. Self-Attention
    //     缩放因子 scale = 1/sqrt(dh)
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dh));
    tensor_t attn_val = Tensor::create(q_rope->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    ops::self_attention(attn_val, q_rope, K_all, V_all, scale);
    // attn_val: [1, nh, dh] = [1, 12, 128]

    // 7. Output projection
    //    attn_val: [1, 12, 128] → view [1, 1536] → linear o_proj → [1, 1536]
    tensor_t attn_2d = attn_val->view({1, config_.nh * config_.dh});
    tensor_t attn_out = Tensor::create({1, config_.hs}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::linear(attn_out, attn_2d, w.attn_o_w, nullptr);

    // 8. Residual add: hidden = hidden + attn_out
    ops::add(hidden, hidden, attn_out);

    // 9. MLP: RMS Norm → SwiGLU → Down proj → Residual
    //    9a. Post-attention RMS Norm
    tensor_t mlp_normed = Tensor::create(hidden->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    ops::rms_norm(mlp_normed, hidden, w.mlp_norm_w, config_.epsilon);

    //    9b. SwiGLU: gate = linear(mlp_normed, gate_w) → [1, di]
    //                up   = linear(mlp_normed, up_w)   → [1, di]
    //                swiglu(out, gate, up)             → [1, di]
    //                down = linear(out, down_w)        → [1, hs]
    tensor_t gate = Tensor::create({1, config_.di}, LLAISYS_DTYPE_F32, device_, device_id_);
    tensor_t up   = Tensor::create({1, config_.di}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::linear(gate, mlp_normed, w.mlp_gate_w, nullptr);
    ops::linear(up,   mlp_normed, w.mlp_up_w,   nullptr);

    tensor_t mlp_mid = Tensor::create({1, config_.di}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::swiglu(mlp_mid, gate, up);

    tensor_t mlp_out = Tensor::create({1, config_.hs}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::linear(mlp_out, mlp_mid, w.mlp_down_w, nullptr);
    
    //    9c. Residual add: hidden = hidden + mlp_out
    ops::add(hidden, hidden, mlp_out);

    return hidden;
}

Qwen2Model::Qwen2Model(const Qwen2Config &config,
                       llaisysDeviceType_t device, int device_id)
    : config_(config), device_(device), device_id_(device_id) {}

void Qwen2Model::load_weights(const Qwen2Weights &weights) {
    weights_ = weights;
}

tensor_t Qwen2Model::forward(tensor_t input_ids, Qwen2KVCache &cache) {
    // 1. Embedding: token ID → hidden vector [1, hs]
    tensor_t hidden = Tensor::create({1, config_.hs}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::embedding(hidden, input_ids, weights_.in_embed);

    // 2. 循环 28 层 Transformer
    for (size_t i = 0; i < config_.nlayer; i++) {
        hidden = forward_layer(hidden, weights_.layers[i], cache, i);
    }

    // 3. Final RMS Norm
    tensor_t normed = Tensor::create(hidden->shape(), LLAISYS_DTYPE_F32, device_, device_id_);
    ops::rms_norm(normed, hidden, weights_.out_norm_w, config_.epsilon);

    // 4. LM Head: 投影到词表大小 [1, voc]
    tensor_t logits = Tensor::create({1, config_.voc}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::linear(logits, normed, weights_.out_embed, nullptr);

    // 5. Argmax: 取概率最大的 token ID
    tensor_t next_token = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
    tensor_t max_val   = Tensor::create({1}, LLAISYS_DTYPE_F32, device_, device_id_);
    ops::argmax(next_token, max_val, logits);

    return next_token;
}

} // namespace llaisys::models