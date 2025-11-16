#!/bin/bash
# 快速开始脚本 - 自动化执行所有步骤

set -e  # 遇到错误立即退出

echo "=========================================="
echo "轻量化大模型移动端部署研究 - 快速开始"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    python3 -m venv venv
fi

echo -e "${GREEN}激活虚拟环境...${NC}"
source venv/bin/activate

# 检查依赖
echo -e "${GREEN}检查并安装依赖...${NC}"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# 环境检查
echo -e "${GREEN}运行环境检查...${NC}"
python check_environment.py

echo ""
echo -e "${YELLOW}请查看上面的环境检查报告${NC}"
read -p "按Enter继续，或Ctrl+C退出..."

# 阶段一：数据准备和训练
echo ""
echo "=========================================="
echo "阶段一：数据准备和模型微调"
echo "=========================================="

# 步骤1: 下载数据
echo -e "${GREEN}步骤1: 下载数据集...${NC}"
python stage1_finetune/download_data.py

# 步骤2: 预处理数据
echo -e "${GREEN}步骤2: 预处理数据...${NC}"
python stage1_finetune/preprocess_data.py

# 步骤3: 开始训练
echo -e "${GREEN}步骤3: 开始模型训练...${NC}"
echo -e "${YELLOW}这可能需要30-60分钟，请耐心等待...${NC}"
python stage1_finetune/train_lora.py --config config/training_config.yaml

# 步骤4: 评估模型
echo -e "${GREEN}步骤4: 评估模型...${NC}"
python stage1_finetune/evaluate.py --model_path outputs/checkpoints/final

echo ""
echo -e "${GREEN}阶段一完成！${NC}"
echo ""

# 询问是否继续阶段二
read -p "是否继续阶段二（模型转换与量化）？(y/n): " continue_stage2
if [ "$continue_stage2" != "y" ]; then
    echo "阶段二已跳过，您可以稍后手动执行。"
    exit 0
fi

# 阶段二：模型转换和量化
echo ""
echo "=========================================="
echo "阶段二：模型转换与量化"
echo "=========================================="

# 步骤1: 合并LoRA权重
echo -e "${GREEN}步骤1: 合并LoRA权重...${NC}"
python stage2_quantize/merge_lora.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_path outputs/checkpoints/final \
    --output_path outputs/merged_model

# 检查llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo -e "${YELLOW}llama.cpp未找到，正在克隆...${NC}"
    git clone https://gitee.com/mirrors/llama.cpp.git || git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# 检查是否已编译
if [ ! -f "quantize" ] && [ ! -f "llama-cli" ]; then
    echo -e "${YELLOW}编译llama.cpp...${NC}"
    make
fi

# 步骤2: 转换为GGUF
echo -e "${GREEN}步骤2: 转换为GGUF格式...${NC}"
python convert-hf-to-gguf.py \
    ../outputs/merged_model \
    --outdir ../outputs/gguf \
    --outtype f16

# 步骤3: 量化模型
echo -e "${GREEN}步骤3: 量化模型...${NC}"
./quantize \
    ../outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf \
    ../outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    Q4_K_M

cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "所有步骤完成！"
echo "==========================================${NC}"
echo ""
echo "输出文件："
echo "  - LoRA权重: outputs/checkpoints/final/"
echo "  - 合并模型: outputs/merged_model/"
echo "  - GGUF模型: outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
echo ""


