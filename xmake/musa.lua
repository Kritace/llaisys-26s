-- 摩尔线程 MUSA 链接配置
-- MUSA 源文件(.mu) 由 scripts/build_musa.sh 用 mcc 编译并打包为：
--   build/musa_objs/libllaisys-musa.a （含 device + 8 个算子对象）
-- 本文件只负责让 libllaisys.so 链接这个预编译静态库。
-- 改动 musa 源码后需重跑：bash scripts/build_musa.sh
-- 启用方式: xmake f --musa-gpu=y -y && xmake

target("llaisys")
    -- mublas（linear/self_attention 用）
    add_syslinks("mublas")
    add_linkdirs("/usr/local/musa/lib")
    -- 预编译的 musa 静态库（排在静态库之后链接，避免 --as-needed 丢弃）
    add_syslinks("llaisys-musa")
    add_linkdirs("../build/musa_objs")

    -- 关键：libllaisys-musa.a 由 build_musa.sh 生成，xmake 无法追踪其变化，
    -- 若不强制重链，改 musa 代码后 .so 仍是旧的（表现为 f16 GEMM 一直报 NOT_IMPLEMENTED）。
    -- 构建前删除 .so，强制每次重新链接。
    before_build(function (target)
        os.rm(target:targetfile())
    end)
target_end()
