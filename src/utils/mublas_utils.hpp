#pragma once

#include <mublas.h>

#include <iostream>
#include <stdexcept>

// mublas 错误检查宏：与 CHECK_CUBLAS 对应，针对 MUSA BLAS 的状态码。
#define CHECK_MUBLAS(call)                                                     \
    do {                                                                       \
        mublasStatus_t st___ = (call);                                         \
        if (st___ != MUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "[ERROR] mublas error code: " << (int)st___           \
                      << " at " << __FILE__ << ":" << __LINE__ << "." << std::endl; \
            throw std::runtime_error("mublas error");                          \
        }                                                                      \
    } while (0)

// 进程内只创建一次的 mublas 句柄（Meyers singleton）。
// 对应 cublas_utils.hpp 的 get_cublas_handle()；linear / self_attention 共用。
inline mublasHandle_t get_mublas_handle() {
    static mublasHandle_t handle = [] {
        mublasHandle_t h;
        if (mublasCreate(&h) != MUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("mublas create failed");
        }
        return h;
    }();
    return handle;
}
