#!/bin/bash
# 设置llama.cpp脚本
# 用于配置手动下载的llama.cpp仓库

set -e

echo "=========================================="
echo "配置llama.cpp"
echo "=========================================="

cd /home/ljh/AI

# 检查llama.cpp目录
if [ ! -d "llama.cpp" ]; then
    if [ -d "llama.cpp-master" ]; then
        echo "重命名 llama.cpp-master -> llama.cpp"
        mv llama.cpp-master llama.cpp
    else
        echo "❌ 错误: 找不到llama.cpp目录"
        echo "请确保已将llama.cpp仓库下载到 llama.cpp 或 llama.cpp-master 目录"
        exit 1
    fi
fi

cd llama.cpp

echo ""
echo "检查llama.cpp目录结构..."

# 检查关键文件
if [ ! -f "Makefile" ]; then
    echo "⚠️  警告: Makefile不存在，可能不是完整的llama.cpp仓库"
fi

# 查找转换脚本
CONVERT_SCRIPT=""
if [ -f "convert-hf-to-gguf.py" ]; then
    CONVERT_SCRIPT="convert-hf-to-gguf.py"
elif [ -f "convert.py" ]; then
    CONVERT_SCRIPT="convert.py"
elif [ -f "convert_hf_to_gguf.py" ]; then
    CONVERT_SCRIPT="convert_hf_to_gguf.py"
else
    echo "⚠️  警告: 未找到转换脚本"
    echo "尝试查找..."
    find . -maxdepth 2 -name "*convert*.py" | head -3
fi

if [ -n "$CONVERT_SCRIPT" ]; then
    echo "✅ 找到转换脚本: $CONVERT_SCRIPT"
fi

# 检查是否已编译
if [ -f "quantize" ] || [ -f "llama-cli" ] || [ -f "llama-server" ]; then
    echo "✅ 检测到已编译的可执行文件"
    echo "可执行文件:"
    ls -lh quantize llama-cli llama-server 2>/dev/null | awk '{print "  "$9" ("$5")"}'
else
    echo "⚠️  未检测到已编译的可执行文件"
    echo ""
    echo "需要编译llama.cpp:"
    echo "  cd llama.cpp"
    echo "  make"
fi

# 检查Python依赖
if [ -f "requirements.txt" ]; then
    echo ""
    echo "检查Python依赖..."
    if python3 -c "import numpy" 2>/dev/null; then
        echo "✅ numpy已安装"
    else
        echo "⚠️  numpy未安装，建议安装:"
        echo "  pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
fi

echo ""
echo "=========================================="
echo "llama.cpp配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 如果未编译，运行: cd llama.cpp && make"
echo "2. 安装Python依赖: cd llama.cpp && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo "3. 继续执行阶段二的步骤"




