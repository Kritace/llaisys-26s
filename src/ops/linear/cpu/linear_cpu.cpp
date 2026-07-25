#include "linear_cpu.hpp"
#include "../../add/cpu/add_cpu.hpp"

#include "../../../utils.hpp"
#include <iostream>

// 线性变换操作：计算 out = in * weight^T + bias, bias是可选的。
template <typename T>
void linear_(const T *in, const T *weight, T *out, size_t in_rows, size_t in_cols, size_t out_cols, const T *bias = nullptr) {

    for (size_t i = 0; i < in_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < in_cols; ++k) {
                float a = llaisys::utils::cast<float>(in[i * in_cols + k]);
                float b = llaisys::utils::cast<float>(weight[j * in_cols + k]);
                acc += a * b;
            }
            if (bias) acc += llaisys::utils::cast<float>(bias[j]);
            out[i * out_cols + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, llaisysDataType_t type, size_t /*numel*/) {
    const size_t in_rows = in->shape()[0];
    const size_t in_cols = in->shape()[1];
    const size_t out_cols = weight->shape()[0];

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<const float *>(in->data()),
                       reinterpret_cast<const float *>(weight->data()),
                       reinterpret_cast<float *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const float *>(bias->data()) : nullptr);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                       reinterpret_cast<llaisys::bf16_t *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                       reinterpret_cast<llaisys::fp16_t *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu