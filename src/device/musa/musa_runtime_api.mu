#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

#include <musa_runtime.h>

#include "../../utils/musa_check.hpp"

namespace llaisys::device::musa {

namespace runtime_api {

// 可用的 GPU 数量
int getDeviceCount() {
    int count = 0;
    CHECK_MUSA(musaGetDeviceCount(&count));
    return count;
}

// 设置当前使用的 GPU 设备
void setDevice(int device) {
    CHECK_MUSA(musaSetDevice(device));
}

// 同步设备，等待所有任务完成
void deviceSynchronize() {
    CHECK_MUSA(musaDeviceSynchronize());
}

// 创建一个新的 MUSA 流
llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    CHECK_MUSA(musaStreamCreateWithFlags(&stream, musaStreamNonBlocking));
    return (llaisysStream_t)stream;
}

// 销毁一个 MUSA 流
void destroyStream(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamDestroy((musaStream_t)stream));
}

// 同步 MUSA 流
void streamSynchronize(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamSynchronize((musaStream_t)stream));
}

// 在设备上分配内存
void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_MUSA(musaMalloc(&ptr, size));
    return ptr;
}

// 释放设备内存
void freeDevice(void *ptr) {
    CHECK_MUSA(musaFree(ptr));
}

// 在主机上分配页锁定内存
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CHECK_MUSA(musaMallocHost(&ptr, size));
    return ptr;
}

// 释放主机页锁定内存
void freeHost(void *ptr) {
    CHECK_MUSA(musaFreeHost(ptr));
}

// 将 llaisysMemcpyKind_t 转换为 musaMemcpyKind
static musaMemcpyKind toMusaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    }
    throw std::invalid_argument("Invalid memcpy kind");
}

// 同步内存拷贝
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_MUSA(musaMemcpy(dst, src, size, toMusaMemcpyKind(kind)));
}

// 异步内存拷贝
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_MUSA(musaMemcpyAsync(dst, src, size, toMusaMemcpyKind(kind), (musaStream_t)stream));
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
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::musa
