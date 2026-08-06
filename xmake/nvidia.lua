-- NVIDIA CUDA 编译流程
-- 参考 cpu.lua 编写。启用方式: xmake f --nv-gpu=y -cv && xmake
-- 需要 nvcc 在 PATH 中（WSL 里安装 CUDA Toolkit 后自动满足）

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    -- 关闭 RDC（xmake cuda 规则默认 -rdc=true）：
    -- 当前 .cu 均为纯 host 代码，无需 relocatable device code；
    -- 且 devlink 只对 shared/binary 自身含 .cu 的 target 执行，静态库里的 .cu
    -- 不会触发 nvcc -dlink，开着 RDC 会让 __cudaRegisterLinkedBinary 符号悬空。
    -- 后续若加入真正的 kernel，需在同一 .cu 翻译单元内定义，或把 .cu 移入 shared target 并启用 RDC。
    set_values("cuda.rdc", false)
    -- 目标架构：同时支持本地 RTX 30 系 (Ampere, sm_86) 和服务器 RTX 5090 (Blackwell, sm_120)
    -- 注意：sm_120 需要 CUDA >= 12.8。若服务器是其他卡，可改为对应 sm_XX 或 "native"
    add_cugencodes("sm_86", "sm_120")
    -- .cu 的 host 部分需要 -fPIC 才能链接进 libllaisys.so
    add_cuflags("-Xcompiler=-fPIC")
    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

-- 增量合并（无需修改 xmake.lua）：让 llaisys-device 链接 nvidia 设备实现
target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()

-- CUDA 算子：编译 src/ops/*/nvidia/*.cu
target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    -- 与 llaisys-device-nvidia 同理：静态库内的 .cu 关闭 RDC，避免悬空符号
    set_values("cuda.rdc", false)
    add_cugencodes("sm_86", "sm_120")
    add_cuflags("-Xcompiler=-fPIC")
    add_files("../src/ops/*/nvidia/*.cu")
    on_install(function (target) end)
target_end()

-- 增量合并：让 llaisys-ops 链接 nvidia 算子
target("llaisys-ops")
    add_deps("llaisys-ops-nvidia")
target_end()

-- 增量合并：llaisys 共享库链接 cuBLAS（linear/self_attention 用）
-- 用 add_syslinks 让 -lcublas 排在静态库之后链接，
-- 避免默认 --as-needed 因"符号尚未被引用"而把它丢掉
target("llaisys")
    add_syslinks("cublas")
target_end()
