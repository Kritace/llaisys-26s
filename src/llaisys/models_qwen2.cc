#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"
#include "../core/llaisys_core.hpp"   // core::context()：设备感知的 memcpy

#include <vector>
#include <cstring>

/**
 * C++ 使用 tensor_t: std::shared_ptr<Tensor>
 * LlaisysTensor 是一个包含 tensor_t 的结构体
 * C 使用 llaisysTensor_t: LlaisysTensor*
 */

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *cpp_model; // cpp model ptr
    llaisys::models::Qwen2Weights cpp_weights;
    llaisys::models::Qwen2KVCache cache; // KV Cache
    LlaisysQwen2Weights c_weights;
    llaisysDeviceType_t device;
    int device_id;

    // 每层权重指针数组（C 权重结构中的指针指向这里的 data()）
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

// 辅助：llaisysTensor_t → C++ tensor_t（提取共享指针）
static llaisys::tensor_t c2cpp(llaisysTensor_t t) {
    if (!t) return nullptr;
    return ((LlaisysTensor*)t)->tensor;
}

// 辅助：将 C 权重结构同步到 C++ 权重，并载入模型
static void sync_weights(LlaisysQwen2Model *m) {
    size_t n = m->cpp_model->config().nlayer;

    m->cpp_weights.in_embed   = c2cpp(m->c_weights.in_embed);
    m->cpp_weights.out_embed  = c2cpp(m->c_weights.out_embed);
    m->cpp_weights.out_norm_w = c2cpp(m->c_weights.out_norm_w);

    m->cpp_weights.layers.resize(n);

    // 置换每一层的权重
    for (size_t i = 0; i < n; i++) {
        auto &l = m->cpp_weights.layers[i];
        l.attn_norm_w = c2cpp(m->attn_norm_w[i]);
        l.attn_q_w    = c2cpp(m->attn_q_w[i]);
        l.attn_q_b    = c2cpp(m->attn_q_b[i]);
        l.attn_k_w    = c2cpp(m->attn_k_w[i]);
        l.attn_k_b    = c2cpp(m->attn_k_b[i]);
        l.attn_v_w    = c2cpp(m->attn_v_w[i]);
        l.attn_v_b    = c2cpp(m->attn_v_b[i]);
        l.attn_o_w    = c2cpp(m->attn_o_w[i]);
        l.mlp_norm_w  = c2cpp(m->mlp_norm_w[i]);
        l.mlp_gate_w  = c2cpp(m->mlp_gate_w[i]);
        l.mlp_up_w    = c2cpp(m->mlp_up_w[i]);
        l.mlp_down_w  = c2cpp(m->mlp_down_w[i]);
    }

    m->cpp_model->load_weights(m->cpp_weights);
}


// C API 实现
__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids, int ndevice)
{
    auto *m = new LlaisysQwen2Model();
    m->device = device;
    m->device_id = (ndevice > 0) ? device_ids[0] : 0;

    // 元信息 → C++ Config
    llaisys::models::Qwen2Config config;
    config.nlayer  = meta->nlayer;
    config.hs      = meta->hs;
    config.nh      = meta->nh;
    config.nkvh    = meta->nkvh;
    config.dh      = meta->dh;
    config.di      = meta->di;
    config.voc     = meta->voc;
    config.epsilon = meta->epsilon;
    config.theta   = meta->theta;

    // 创建 C++ 模型
    m->cpp_model = new llaisys::models::Qwen2Model(config, device, m->device_id);

    // 初始化 C 权重结构

    // 全局权重先置 nullptr
    m->c_weights.in_embed   = nullptr;
    m->c_weights.out_embed  = nullptr;
    m->c_weights.out_norm_w = nullptr;

    // 每层权重数组：分配 nlayer 个空位
    size_t n = meta->nlayer;
    m->attn_norm_w.resize(n, nullptr);
    m->attn_q_w.resize(n, nullptr);
    m->attn_q_b.resize(n, nullptr);
    m->attn_k_w.resize(n, nullptr);
    m->attn_k_b.resize(n, nullptr);
    m->attn_v_w.resize(n, nullptr);
    m->attn_v_b.resize(n, nullptr);
    m->attn_o_w.resize(n, nullptr);
    m->mlp_norm_w.resize(n, nullptr);
    m->mlp_gate_w.resize(n, nullptr);
    m->mlp_up_w.resize(n, nullptr);
    m->mlp_down_w.resize(n, nullptr);

    // C 权重结构中的指针 → 指向 cpp vector 的首地址
    m->c_weights.attn_norm_w = m->attn_norm_w.data();
    m->c_weights.attn_q_w    = m->attn_q_w.data();
    m->c_weights.attn_q_b    = m->attn_q_b.data();
    m->c_weights.attn_k_w    = m->attn_k_w.data();
    m->c_weights.attn_k_b    = m->attn_k_b.data();
    m->c_weights.attn_v_w    = m->attn_v_w.data();
    m->c_weights.attn_v_b    = m->attn_v_b.data();
    m->c_weights.attn_o_w    = m->attn_o_w.data();
    m->c_weights.mlp_norm_w  = m->mlp_norm_w.data();
    m->c_weights.mlp_gate_w  = m->mlp_gate_w.data();
    m->c_weights.mlp_up_w    = m->mlp_up_w.data();
    m->c_weights.mlp_down_w  = m->mlp_down_w.data();

    // 初始化缓存
    m->cache.step = 0;

    // C++ 权重先置空
    m->cpp_weights.in_embed   = nullptr;
    m->cpp_weights.out_embed  = nullptr;
    m->cpp_weights.out_norm_w = nullptr;

    return m;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete model->cpp_model;
        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model)
{
    return &model->c_weights;
}

/**
*@brief 推理入口
*@return next token id
*/
int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids, size_t ntoken)
{
    // 同步 C 权重 → C++ 权重
    sync_weights(model);

    int64_t result = 0;

    // 逐个 token 推理
    for (size_t i = 0; i < ntoken; i++) {
        // 创建 int64 输入张量，载入当前 token ID
        auto input = llaisys::Tensor::create(
            {1}, LLAISYS_DTYPE_I64, model->device, model->device_id);

        input->load(&token_ids[i]);

        // forward：输出下一个 token 的预测
        auto next = model->cpp_model->forward(input, model->cache);

        // 读取结果，memcpy_sync 在 CPU 下就是 std::memcpy（忽略 kind），在 GPU 下是 cudaMemcpy(D2H)。
        // 若直接 std::memcpy, GPU 下会把显存指针当主机源读，导致段错误
        llaisys::core::context().runtime().api()->memcpy_sync(
            &result, next->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

        model->cache.step++;
    }

    return result;
}

} // __C
