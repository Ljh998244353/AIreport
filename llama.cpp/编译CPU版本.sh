#!/bin/bash
# llama.cpp CPU版本一键编译脚本

set -e

echo "=========================================="
echo "编译llama.cpp (CPU版本)"
echo "=========================================="

# 检查是否在llama.cpp目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误: 请在llama.cpp目录下运行此脚本"
    exit 1
fi

# 检查cmake
if ! command -v cmake &> /dev/null; then
    echo "安装cmake..."
    sudo apt update
    sudo apt install -y build-essential cmake
else
    echo "✅ cmake已安装"
fi

# 清理旧的build目录（可选）
if [ -d "build" ]; then
    echo ""
    read -p "检测到build目录，是否清理后重新编译？(y/n，默认n): " clean_choice
    clean_choice=${clean_choice:-n}
    if [ "$clean_choice" == "y" ]; then
        echo "清理build目录..."
        rm -rf build
    fi
fi

# 创建build目录
echo ""
echo "创建build目录..."
mkdir -p build
cd build

# 配置CMake（CPU版本）
echo ""
echo "配置CMake (CPU版本)..."
echo "注意: 禁用CURL功能（-DLLAMA_CURL=OFF）"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CMake配置失败，请检查错误信息"
    echo "如果提示缺少CURL，可以安装: sudo apt install libcurl4-openssl-dev"
    echo "或保持 -DLLAMA_CURL=OFF（推荐，对于量化工具不需要CURL）"
    exit 1
fi

# 编译
echo ""
echo "开始编译（这可能需要几分钟）..."
echo "使用 $(nproc) 个并行任务"
cmake --build . --config Release -j$(nproc)

# 检查编译结果
echo ""
echo "检查编译结果..."
if [ -f "bin/llama-quantize" ] && [ -f "bin/llama-cli" ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "可执行文件位置:"
    ls -lh bin/llama-quantize bin/llama-cli 2>/dev/null | awk '{print "  "$9" ("$5")"}'
    
    # 创建符号链接
    echo ""
    echo "创建符号链接..."
    cd ..
    ln -sf build/bin/llama-cli llama-cli
    ln -sf build/bin/llama-quantize quantize
    echo "✅ 符号链接已创建"
    echo "   可以使用: ./llama-cli 和 ./quantize"
    echo "   注意: 实际文件名是 llama-quantize，已创建为 quantize 方便使用"
else
    echo "❌ 编译失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 安装Python依赖: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo "2. 继续执行阶段二的步骤（转换和量化模型）"


