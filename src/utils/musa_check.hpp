#pragma once

#include <musa_runtime.h>

#include <iostream>
#include <stdexcept>

// MUSA 错误检查宏：调用 MUSA Runtime 函数，失败则打印错误码并抛异常。
// 与 cuda_check.hpp 的 CHECK_CUDA 对应，但针对 MUSA 的错误码（musaError_t）。
// 备注：MUSA 兼容 CUDA，错误码枚举基本同名（musaSuccess 等）。
#define CHECK_MUSA(call)                                                       \
    do {                                                                       \
        musaError_t e___ = (call);                                             \
        if (e___ != musaSuccess) {                                             \
            std::cerr << "[ERROR] MUSA error code: " << (int)e___              \
                      << " at " << __FILE__ << ":" << __LINE__ << "." << std::endl; \
            throw std::runtime_error("MUSA error");                            \
        }                                                                      \
    } while (0)
