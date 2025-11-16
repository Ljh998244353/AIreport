#!/bin/bash
# 快速下载模型脚本 - 使用ModelScope（推荐，国内速度快）

echo "=========================================="
echo "使用ModelScope下载模型（阿里云，国内速度快）"
echo "=========================================="
echo ""

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 运行ModelScope下载脚本
python download_from_modelscope.py

echo ""
echo "如果下载成功，模型可以直接用于训练！"


