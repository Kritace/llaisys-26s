#!/usr/bin/env bash
# 编译所有 MUSA 源文件(.mu) → build/musa_objs/*.o
# 用 mcc（MUSA 4.3.5），编译命令已在服务器验证通过
# 用法：bash scripts/build_musa.sh   （改动 musa 源码后需重跑）
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build/musa_objs
mkdir -p "$OUT"

# 已验证的 mcc 编译参数（-x musa 不需要，.mu 后缀 mcc 自动识别为 musa）
MCC_FLAGS="-c -fPIC --musa-path=/usr/local/musa -m64 -O3 -DNDEBUG \
--offload-arch=mp_10 --offload-arch=mp_21 --offload-arch=mp_22 --offload-arch=mp_31 \
-std=c++17 -Iinclude -Isrc -I/usr/local/musa/include"

echo "=== 编译 MUSA 源文件 ==="
for f in src/device/musa/*.mu src/ops/*/musa/*.mu; do
    base=$(basename "$f" .mu)
    echo "  mcc $f -> $OUT/$base.o"
    mcc $MCC_FLAGS "$f" -o "$OUT/$base.o"
done
echo "=== 完成: $(ls "$OUT"/*.o 2>/dev/null | wc -l) 个 .o ==="

echo "=== 打包 libllaisys-musa.a ==="
ar rcs "$OUT/libllaisys-musa.a" "$OUT"/*.o
ls -lh "$OUT/libllaisys-musa.a"
