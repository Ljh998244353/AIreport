#!/bin/bash
# llama.cpp编译脚本
# 使用CMake构建系统

set -e

echo "=========================================="
echo "编译llama.cpp"
echo "=========================================="

# 检查是否在llama.cpp目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误: 请在llama.cpp目录下运行此脚本"
    exit 1
fi

# 安装编译依赖
echo ""
echo "检查编译依赖..."
if ! command -v cmake &> /dev/null; then
    echo "安装cmake..."
    sudo apt update
    sudo apt install -y build-essential cmake
else
    echo "✅ cmake已安装"
fi

# 创建build目录
echo ""
echo "创建build目录..."
mkdir -p build
cd build

# 配置CMake
echo ""
echo "配置CMake..."
echo "选择编译类型:"
echo "1) GPU版本（默认，需要CUDA）"
echo "2) CPU版本"
read -p "请选择 (1/2，默认1): " choice
choice=${choice:-1}

if [ "$choice" == "2" ]; then
    echo "配置CPU版本..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
else
    echo "配置GPU版本（CUDA）..."
    echo "注意: 新版本使用 GGML_CUDA 替代 LLAMA_CUBLAS"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF
    
    # 如果CUDA配置失败，提示用户
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ CUDA配置失败"
        echo "可能的原因："
        echo "1. CUDA未正确安装"
        echo "2. CMake版本过低"
        echo "3. 缺少CUDA开发库"
        echo ""
        echo "建议："
        echo "1. 检查CUDA安装: nvcc --version"
        echo "2. 安装CUDA开发库: sudo apt install nvidia-cuda-toolkit"
        echo "3. 或使用CPU版本: 重新运行脚本选择选项2"
        exit 1
    fi
fi

# 编译
echo ""
echo "开始编译（这可能需要几分钟）..."
cmake --build . --config Release -j$(nproc)

# 检查编译结果
echo ""
echo "检查编译结果..."
if [ -f "bin/quantize" ] && [ -f "bin/llama-cli" ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "可执行文件位置:"
    ls -lh bin/quantize bin/llama-cli | awk '{print "  "$9" ("$5")"}'
    
    # 创建符号链接
    echo ""
    read -p "是否创建符号链接到根目录？(y/n，默认y): " create_link
    create_link=${create_link:-y}
    if [ "$create_link" == "y" ]; then
        cd ..
        ln -sf build/bin/llama-cli llama-cli
        ln -sf build/bin/quantize quantize
        echo "✅ 符号链接已创建"
        echo "   ./llama-cli"
        echo "   ./quantize"
    fi
else
    echo "❌ 编译失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="

