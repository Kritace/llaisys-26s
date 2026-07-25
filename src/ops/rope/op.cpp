#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");
    
    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out, in, pos_ids, out->dtype(), theta);
    }

    llaisys::core::context().setDevice(in->deviceType(), out->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, out->dtype(), theta);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
