#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// 检查张量的形状和步长，判断它在内存中是否连续。
bool Tensor::isContiguous() const {
    // 检查张量是否是连续的。一个张量是连续的，如果它的步长与其形状一致，即每个维度的步长等于该维度之后所有维度的元素数量的乘积。
    ptrdiff_t expected_stride = 1;
    for (size_t i = this->ndim(); i-- > 0;) {
        if (this->_meta.shape[i] != 1) {
            if (this->_meta.strides[i] != expected_stride) {
                return false;
            }
            expected_stride *= static_cast<ptrdiff_t>(this->_meta.shape[i]);
        }
    }
    return true;
}

// 创建一个新张量，改变原始张量维度的顺序。转置可以通过这个函数实现，而无需移动数据。
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查置换顺序的有效性，确保它与张量的维度数量匹配，并且没有重复或超出范围的索引。
    CHECK_ARGUMENT(order.size() == this->ndim(), "Permutation order must match tensor dimensions");

    size_t ndim_ = this->ndim();
    // 使用布尔向量来跟踪每个维度是否已被使用，以确保置换顺序中没有重复的索引。
    std::vector<bool> used(ndim_, false);

    // 遍历置换顺序，检查每个索引是否在有效范围内，并且没有重复使用。如果发现无效索引或重复索引，将抛出异常。
    for (size_t idx = 0; idx < ndim_; idx++) {
        CHECK_ARGUMENT(order[idx] < ndim_, "Permutation indices out of range");
        CHECK_ARGUMENT(!used[order[idx]], "Permutation contains duplicate indices");
        used[order[idx]] = true;
    }

    // 创建新的张量元数据，重新排列形状和步长以反映置换顺序。新张量共享原始张量的存储，但具有不同的形状和步长。
    TensorMeta meta;
    meta.dtype = this->_meta.dtype;
    meta.shape.resize(ndim_);
    meta.strides.resize(ndim_);

    // 遍历置换顺序，将原始张量的形状和步长按照新的顺序复制到新张量的元数据中。
    for (size_t i = 0; i < ndim_; i++) {
        meta.shape[i] = this->_meta.shape[order[i]];
        meta.strides[i] = this->_meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), this->_storage, this->_offset));
}

// 创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输。
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查新形状的元素数量是否与原始张量的元素数量匹配
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    // 检查张量是否是连续的，因为当前实现仅支持连续张量的视图操作
    CHECK_ARGUMENT(new_numel == this->numel(), "Incompatible view shape");
    if (!this->isContiguous()) {
        CHECK_ARGUMENT(false, "View is only supported for contiguous tensors in the current implementation");
    }

    // 计算新的步长，以便新张量可以正确地索引原始张量的数据
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= shape.size(); i++) {
        new_strides[shape.size() - i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[shape.size() - i]);
    }
    
    // 创建新的张量元数据，并返回一个指向新张量的共享指针。新张量共享原始张量的存储，但具有不同的形状和步长。
    TensorMeta meta{this->_meta.dtype, shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), this->_storage, this->_offset));
}

// 创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 检查切片参数的有效性
    CHECK_ARGUMENT(dim < this->ndim(), "Slice dimension out of range");
    CHECK_ARGUMENT(start <= end, "Slice start must be <= end");
    CHECK_ARGUMENT(end <= this->shape()[dim], "Slice end out of range");

    // 创建新的张量元数据，更新形状以反映切片操作
    TensorMeta meta = this->_meta;
    meta.shape[dim] = end - start;

    // 计算新的偏移量，以便新张量指向原始张量的正确位置
    size_t offset_bytes = this->_offset + static_cast<size_t>(start * this->_meta.strides[dim]) * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), this->_storage, offset_bytes));
}

// 将主机（cpu）数据加载到张量（可以在设备上）。查看构造函数了解如何获取当前设备上下文的运行时API，并执行从主机到设备的内存复制。
void Tensor::load(const void *src_) {
    // 将当前设备上下文设置为张量所在的设备类型和设备ID，以确保后续操作在正确的设备上执行。
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();

    // 检查张量的设备类型。如果是CPU设备，直接使用std::memcpy将数据从主机复制到张量的内存中。
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, this->numel() * this->elementSize());
    } else {
        // 如果张量位于非CPU设备（如GPU），则使用运行时API的memcpy_sync函数将数据从主机复制到设备。该函数确保数据传输在继续执行之前完成。
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
