-- 摩尔线程 MUSA 链接配置
-- MUSA 源文件(.mu) 由 scripts/build_musa.sh 用 mcc 编译并打包为：
--   build/musa_objs/libllaisys-musa.a （含 device + 8 个算子对象）
-- 本文件只负责让 libllaisys.so 链接这个预编译静态库。
-- 改动 musa 源码后需重跑：bash scripts/build_musa.sh
-- 启用方式: xmake f --musa-gpu=y -y && xmake

-- MUSA SDK 路径（默认官方路径，可用环境变量 MUSA_PATH 覆盖）
local musa_path = os.getenv("MUSA_PATH") or "/usr/local/musa"

target("llaisys")
    -- mublas（linear/self_attention 用）
    add_syslinks("mublas")
    add_linkdirs(path.join(musa_path, "lib"))
    -- 运行时自动定位 mublas/musa 动态库（免手动 export LD_LIBRARY_PATH）
    add_rpathdirs(path.join(musa_path, "lib"))
    -- 预编译的 musa 静态库（排在静态库之后链接，避免 --as-needed 丢弃）
    add_syslinks("llaisys-musa")
    add_linkdirs("../build/musa_objs")

    -- 构建前：删除旧 .so 强制重链 + 自动预编译 musa 源（.mu → .a），
    -- 无需手动执行 build_musa.sh，流程与 nvidia 对称。
    before_build(function (target)
        os.rm(target:targetfile())
        os.exec("bash scripts/build_musa.sh")
    end)
target_end()
