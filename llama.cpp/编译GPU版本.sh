#!/bin/bash
# llama.cpp GPU版本一键编译脚本
# 使用CUDA加速

set -e

echo "=========================================="
echo "编译llama.cpp (GPU版本 - CUDA)"
echo "=========================================="

# 检查是否在llama.cpp目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误: 请在llama.cpp目录下运行此脚本"
    exit 1
fi

# 检查CUDA
echo ""
echo "检查CUDA环境..."
CUDA_AVAILABLE=false

if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✅ 检测到CUDA: $nvcc_version"
    CUDA_AVAILABLE=true
elif [ -d "/usr/local/cuda" ]; then
    echo "✅ 检测到CUDA目录: /usr/local/cuda"
    echo "⚠️  但nvcc不在PATH中，尝试设置CUDAToolkit_ROOT"
    export CUDAToolkit_ROOT=/usr/local/cuda
    CUDA_AVAILABLE=true
else
    echo "❌ 未检测到CUDA Toolkit"
    echo ""
    echo "选项："
    echo "1. 使用CPU版本编译（推荐，功能相同，只是速度较慢）"
    echo "2. 安装CUDA后重试"
    echo ""
    read -p "是否使用CPU版本编译？(y/n，默认y): " use_cpu
    use_cpu=${use_cpu:-y}
    if [ "$use_cpu" == "y" ]; then
        echo ""
        echo "切换到CPU版本编译..."
        cd /home/ljh/AI/llama.cpp
        bash 编译CPU版本.sh
        exit $?
    else
        echo ""
        echo "请先安装CUDA Toolkit:"
        echo "  sudo apt install nvidia-cuda-toolkit"
        echo "  或访问: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
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

# 配置CMake（GPU版本 - 使用新的参数名GGML_CUDA）
echo ""
echo "配置CMake (GPU版本 - CUDA)..."
echo "使用参数: -DGGML_CUDA=ON"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CMake配置失败（CUDA未找到）"
    echo ""
    echo "自动切换到CPU版本编译..."
    cd /home/ljh/AI/llama.cpp
    rm -rf build
    bash 编译CPU版本.sh
    exit $?
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
    
    # 验证GPU支持
    echo ""
    echo "验证GPU支持..."
    if ./llama-cli --help 2>&1 | grep -q "cuda\|CUDA\|gpu\|GPU"; then
        echo "✅ GPU支持已启用"
    else
        echo "⚠️  无法确认GPU支持，但编译已完成"
    fi
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


