-- NVIDIA CUDA 编译流程
-- 参考 cpu.lua 编写。启用方式: xmake f --nv-gpu=y -cv && xmake
-- 需要 nvcc 在 PATH 中（WSL 里安装 CUDA Toolkit 后自动满足）

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    -- RTX 30 系 (Ampere, sm_86) 目标架构；如需其他卡可改为 "native" 或对应 sm_XX
    add_cugencodes("sm_86")
    -- .cu 的 host 部分需要 -fPIC 才能链接进 libllaisys.so
    add_cuflags("-Xcompiler=-fPIC")
    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

-- 增量合并（无需修改 xmake.lua）：让 llaisys-device 链接 nvidia 设备实现
target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()

-- 增量合并：llaisys shared 库链接 CUDA runtime (cudart)
target("llaisys")
    add_links("cudart")
target_end()

-- CUDA 算子实现后，在此追加（参考 llaisys-ops-cpu）：
-- target("llaisys-ops-nvidia")
--     set_kind("static")
--     add_deps("llaisys-tensor")
--     set_languages("cxx17")
--     set_warnings("all", "error")
--     add_rules("cuda")
--     add_cugencodes("sm_86")
--     add_files("../src/ops/*/nvidia/*.cu")
--     on_install(function (target) end)
-- target_end()
