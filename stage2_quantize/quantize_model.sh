#!/bin/bash
# 量化GGUF模型
# 注意：此脚本需要在llama.cpp/build/bin/Release目录下运行（Windows）
# 或在llama.cpp目录下运行（Linux/Mac）

# 配置
INPUT_GGUF="../work/outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf"
OUTPUT_GGUF="../work/outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
QUANT_TYPE="Q4_K_M"  # Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q8_0

# 检查输入文件是否存在
if [ ! -f "$INPUT_GGUF" ]; then
    echo "错误: 输入文件不存在: $INPUT_GGUF"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_GGUF")"

# 运行量化
echo "开始量化模型..."
echo "输入: $INPUT_GGUF"
echo "输出: $OUTPUT_GGUF"
echo "量化类型: $QUANT_TYPE"

# Windows (使用quantize.exe)
if [ -f "quantize.exe" ]; then
    ./quantize.exe "$INPUT_GGUF" "$OUTPUT_GGUF" "$QUANT_TYPE"
# Linux/Mac (使用quantize)
elif [ -f "quantize" ]; then
    ./quantize "$INPUT_GGUF" "$OUTPUT_GGUF" "$QUANT_TYPE"
else
    echo "错误: 找不到quantize可执行文件"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "量化完成！"
    echo "输出文件: $OUTPUT_GGUF"
    # 显示文件大小
    if [ -f "$OUTPUT_GGUF" ]; then
        SIZE=$(du -h "$OUTPUT_GGUF" | cut -f1)
        echo "文件大小: $SIZE"
    fi
else
    echo "量化失败！"
    exit 1
fi

