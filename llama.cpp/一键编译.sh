#!/bin/bash
# llama.cpp 智能编译脚本
# 自动检测CUDA并选择最佳编译方式

set -e

echo "=========================================="
echo "llama.cpp 智能编译"
echo "=========================================="

# 检查是否在llama.cpp目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误: 请在llama.cpp目录下运行此脚本"
    exit 1
fi

# 检查CUDA（默认使用GPU版本）
echo ""
echo "检测编译环境（默认使用GPU版本）..."

CUDA_AVAILABLE=false
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✅ 检测到CUDA: $nvcc_version"
    CUDA_AVAILABLE=true
elif [ -d "/usr/local/cuda" ]; then
    echo "✅ 检测到CUDA目录: /usr/local/cuda"
    export CUDAToolkit_ROOT=/usr/local/cuda
    CUDA_AVAILABLE=true
elif nvidia-smi &> /dev/null; then
    echo "⚠️  检测到NVIDIA GPU，但CUDA Toolkit未安装"
    echo "   尝试使用GPU版本编译（如果失败请安装CUDA）"
    CUDA_AVAILABLE=true  # 尝试使用GPU版本
else
    echo "⚠️  未检测到CUDA，但将尝试GPU版本编译"
    echo "   如果失败，请安装CUDA或使用CPU版本脚本"
    CUDA_AVAILABLE=true  # 默认尝试GPU版本
fi

# 检查cmake
if ! command -v cmake &> /dev/null; then
    echo "安装cmake..."
    sudo apt update
    sudo apt install -y build-essential cmake
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

# 配置CMake（默认GPU版本）
echo ""
echo "配置CMake (GPU版本 - CUDA，默认)..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF"
cmake .. $CMAKE_ARGS

# 如果CUDA配置失败，提示用户
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CUDA配置失败"
    echo ""
    echo "可能的原因："
    echo "1. CUDA未正确安装"
    echo "2. CMake版本过低"
    echo "3. 缺少CUDA开发库"
    echo ""
    echo "解决方案："
    echo "1. 安装CUDA: sudo apt install nvidia-cuda-toolkit"
    echo "2. 或使用CPU版本: bash 编译CPU版本.sh"
    echo ""
    read -p "是否切换到CPU版本编译？(y/n，默认n): " use_cpu
    use_cpu=${use_cpu:-n}
    if [ "$use_cpu" == "y" ]; then
        echo ""
        echo "切换到CPU版本编译..."
        cd /home/ljh/AI/llama.cpp
        rm -rf build
        bash 编译CPU版本.sh
        exit $?
    else
        echo ""
        echo "请先安装CUDA后重试，或手动运行: bash 编译CPU版本.sh"
        exit 1
    fi
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
    ls -lh bin/llama-quantize bin/llama-cli | awk '{print "  "$9" ("$5")"}'
    
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

