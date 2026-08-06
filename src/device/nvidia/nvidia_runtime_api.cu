#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "../../utils/cuda_check.hpp"

namespace llaisys::device::nvidia {

namespace runtime_api {

// 可用的GPU数量
int getDeviceCount() {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

// 设置当前使用的GPU设备
void setDevice(int device) {
    CHECK_CUDA(cudaSetDevice(device));
}

// 同步设备，等待所有设备上的任务完成
void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 创建一个新的CUDA流，返回一个指向流的指针
llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return (llaisysStream_t)stream;
}

// 销毁一个CUDA流，释放相关资源
void destroyStream(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamDestroy((cudaStream_t)stream));
}

// 同步CUDA流，等待流中的所有任务完成
void streamSynchronize(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)stream));
}

// 在设备上分配内存，返回指向分配内存的指针
void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

// 释放设备内存
void freeDevice(void *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

// 在主机上分配内存（页锁定），返回指向分配内存的指针
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // 页锁定 (pinned) host 内存，用于 H2D/D2H 异步拷贝
    CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}

// 释放主机内存（页锁定）
void freeHost(void *ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

// 将 llaisysMemcpyKind_t 转换为 cudaMemcpyKind
static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    }
    throw std::invalid_argument("Invalid memcpy kind");
}

// 同步内存拷贝
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_CUDA(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

// 异步内存拷贝
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), (cudaStream_t)stream));
}

// 定义一个静态的 LlaisysRuntimeAPI 实例，包含所有的函数指针
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
