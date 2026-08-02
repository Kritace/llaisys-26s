from typing import Sequence
from pathlib import Path
import ctypes
import json

import numpy as np
import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
)
from ..tensor import Tensor


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # 1. 读取 config.json 获取模型参数 
        with open(model_path / "config.json") as f:
            config = json.load(f)

        # 2. 构建元信息结构体并赋值
        meta = LlaisysQwen2Meta() # meta 是 python -> C 的接口
        meta.dtype = int(DataType.F32)
        meta.nlayer = config["num_hidden_layers"]      # 28
        meta.hs = config["hidden_size"]                # 1536
        meta.nh = config["num_attention_heads"]        # 12
        meta.nkvh = config["num_key_value_heads"]      # 2
        meta.dh = meta.hs // meta.nh                   # 128
        meta.di = config["intermediate_size"]          # 8960
        meta.maxseq = 4096
        meta.voc = config["vocab_size"]                # 151936
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        self._end_token = meta.end_token

        # 3. 通过 C API 创建模型
        device_ids = (ctypes.c_int * 1)(0)

        # 模型句柄
        self._model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), device_ids, 1
        )

        # 4. 获取 C 权重结构 
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model_ptr)
        weights = weights_ptr.contents

        # 保存 device，加载权重时使用
        self._device = int(device)

        # 5. 加载权重 
        # 重要：保持所有 Tensor 对象的引用，防止被垃圾回收销毁底层 C 张量！
        self._weight_tensors = []

        # safetensors 名称 → C 权重结构字段 的映射
        field_map = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(str(file), framework="pt", device="cpu")
            for name in data.keys():
                # 读取张量并转为 float32
                tensor_data = data.get_tensor(name).float().numpy()

                # 创建 C 张量并加载数据
                t = Tensor(
                    shape=tensor_data.shape,
                    dtype=DataType.F32,
                    device=DeviceType(self._device),
                )
                self._weight_tensors.append(t)  # 保持引用
                t.load(tensor_data.ctypes.data_as(ctypes.c_void_p))

                # 按名称分配到权重结构的对应字段
                if name == "model.embed_tokens.weight":
                    weights.in_embed = t.lib_tensor()
                elif name == "lm_head.weight":
                    weights.out_embed = t.lib_tensor()
                elif name == "model.norm.weight":
                    weights.out_norm_w = t.lib_tensor()
                elif name.startswith("model.layers."):
                    parts = name.split(".")
                    layer_idx = int(parts[2])
                    sub = ".".join(parts[3:])
                    field = field_map[sub]
                    # 对应字段是数组：weights.attn_q_w[layer_idx] = ...
                    getattr(weights, field)[layer_idx] = t.lib_tensor()
                else:
                    print(f"[WARN] Unrecognized weight name: {name}")

    # 析构函数
    def __del__(self):
        if hasattr(self, "_model_ptr") and self._model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_ptr)
            self._model_ptr = None

    # 文本生成
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        # 生成的完整序列（包含 prompt）
        generated = list(inputs)

        # Pre-fill: 处理所有 prompt token 
        # llaisysQwen2ModelInfer 会逐个处理输入 token 并更新 KV Cache，
        # 返回最后一个 token 的预测结果（即第一个生成 token）
        prompt_arr = (ctypes.c_int64 * len(inputs))(*inputs)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model_ptr, prompt_arr, len(inputs)
        )

        #  Decode: 循环生成新 token 
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if next_token == self._end_token:
                break
            token_arr = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model_ptr, token_arr, 1
            )

        return generated
