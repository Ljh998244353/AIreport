#!/bin/bash
# 将HuggingFace格式模型转换为GGUF格式
# 注意：此脚本需要在llama.cpp目录下运行

# 配置
HF_MODEL_PATH="../work/outputs/merged_model"
OUTPUT_DIR="../work/outputs/gguf"
OUTPUT_TYPE="f16"  # f16, f32

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行转换脚本（注意：脚本名是 convert_hf_to_gguf.py，使用下划线）
echo "开始转换模型为GGUF格式..."
python convert_hf_to_gguf.py \
    "$HF_MODEL_PATH" \
    --outdir "$OUTPUT_DIR" \
    --outtype "$OUTPUT_TYPE"

echo "转换完成！输出目录: $OUTPUT_DIR"

