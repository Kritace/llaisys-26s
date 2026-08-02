from ctypes import (
    Structure,
    POINTER,
    c_void_p,
    c_size_t,
    c_float,
    c_int64,
    c_int,
)
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    # 对应 C 头文件中的 LlaisysQwen2Meta
    _fields_ = [
        ("dtype", llaisysDataType_t), # 数据类型
        ("nlayer", c_size_t),         # 层数
        ("hs", c_size_t),             # hidden size
        ("nh", c_size_t),             # query heads
        ("nkvh", c_size_t),           # KV heads
        ("dh", c_size_t),             # head dim
        ("di", c_size_t),             # intermediate size
        ("maxseq", c_size_t),         # 最大序列长度
        ("voc", c_size_t),            # vocab size
        ("epsilon", c_float),         # rms norm eps
        ("theta", c_float),           # rope theta
        ("end_token", c_int64),       # 结束 token ID
    ]


class LlaisysQwen2Weights(Structure):
    # 对应 C 头文件中的 LlaisysQwen2Weights
    _fields_ = [
        ("in_embed", llaisysTensor_t),   # model.embed_tokens.weight
        ("out_embed", llaisysTensor_t),  # lm_head.weight
        ("out_norm_w", llaisysTensor_t), # model.norm.weight
        # 每层权重：指向 nlayer 个张量的数组
        ("attn_norm_w", POINTER(llaisysTensor_t)), # input_layernorm.weight
        ("attn_q_w", POINTER(llaisysTensor_t)),    # q_proj.weight
        ("attn_q_b", POINTER(llaisysTensor_t)),    # q_proj.bias
        ("attn_k_w", POINTER(llaisysTensor_t)),    # k_proj.weight
        ("attn_k_b", POINTER(llaisysTensor_t)),    # k_proj.bias
        ("attn_v_w", POINTER(llaisysTensor_t)),    # v_proj.weight
        ("attn_v_b", POINTER(llaisysTensor_t)),    # v_proj.bias
        ("attn_o_w", POINTER(llaisysTensor_t)),    # o_proj.weight
        ("mlp_norm_w", POINTER(llaisysTensor_t)),  # post_attention_layernorm.weight
        ("mlp_gate_w", POINTER(llaisysTensor_t)),  # gate_proj.weight
        ("mlp_up_w", POINTER(llaisysTensor_t)),    # up_proj.weight
        ("mlp_down_w", POINTER(llaisysTensor_t)),  # down_proj.weight
    ]


# 模型句柄（opaque 指针）
LlaisysQwen2Model_t = c_void_p


def load_models(lib):
    # llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)
    # 设置参数类型
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    # 设置返回类型
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_t

    # llaisysQwen2ModelDestroy(model)
    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights(model)
    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights) # 权重结构指针

    # llaisysQwen2ModelInfer(model, token_ids, ntoken)
    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
