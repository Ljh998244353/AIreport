# 阶段一：模型选择与高效微调详细指南

## 概述

本阶段将完成：
1. 环境配置
2. 数据准备（ChnSentiCorp中文情感数据集）
3. 使用 LoRA + QLoRA 进行模型微调
4. 模型评估

## 步骤详解

### 步骤1：环境配置

#### 1.1 创建Python虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

#### 1.2 安装依赖包

```bash
# 使用清华大学镜像源（适合中国大陆）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果bitsandbytes安装失败，可以尝试：
# Windows用户可能需要从源码编译或使用预编译版本
# 或者跳过bitsandbytes，使用标准LoRA（不使用4-bit量化）
```

**注意**: 
- `bitsandbytes` 在Windows上可能安装困难，如果无法安装，可以修改训练脚本使用标准LoRA（不使用4-bit）
- 确保PyTorch版本与CUDA版本匹配

#### 1.3 配置HuggingFace镜像（重要！）

由于网络限制，需要配置HuggingFace镜像：

```bash
# 设置环境变量（Windows PowerShell）
$env:HF_ENDPOINT="https://hf-mirror.com"

# 设置环境变量（Windows CMD）
set HF_ENDPOINT=https://hf-mirror.com

# 设置环境变量（Linux/Mac）
export HF_ENDPOINT=https://hf-mirror.com
```

或者在代码中直接使用镜像站URL。

### 步骤2：数据准备

#### 2.1 下载数据集

运行数据下载脚本：

```bash
python stage1_finetune/download_data.py
```

该脚本会：
- 从HuggingFace镜像站下载 ChnSentiCorp 数据集
- 保存到 `data/chn_senti_corp/` 目录

#### 2.2 数据预处理

运行预处理脚本：

```bash
python stage1_finetune/preprocess_data.py
```

该脚本会：
- 清洗数据（去除空值、重复项）
- 将数据转换为模型输入格式
- 划分训练集/验证集/测试集（8:1:1）
- 保存处理后的数据到 `data/processed/`

### 步骤3：模型微调

#### 3.1 配置训练参数

编辑 `config/training_config.yaml` 文件，设置：
- 学习率
- 批大小
- 训练轮次
- LoRA参数（r, alpha, dropout等）

#### 3.2 开始训练

```bash
python stage1_finetune/train_lora.py --config config/training_config.yaml
```

训练过程：
1. 自动从镜像站下载 Qwen2.5-0.5B-Instruct 模型
2. 使用4-bit量化加载模型（QLoRA）
3. 添加LoRA适配器
4. 开始训练，只更新LoRA权重
5. 保存检查点和最终模型到 `outputs/checkpoints/`

**训练时间估算**：
- 使用GPU（RTX 3060 12GB）: 约30-60分钟
- 使用CPU: 可能需要数小时

#### 3.3 监控训练过程

训练脚本会输出：
- 训练损失
- 验证损失
- 学习率变化
- 训练进度条

### 步骤4：模型评估

训练完成后，评估模型性能：

```bash
python stage1_finetune/evaluate.py --model_path outputs/checkpoints/final
```

评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数

## 常见问题

### Q1: bitsandbytes 安装失败怎么办？

**A**: 可以修改 `train_lora.py`，注释掉4-bit量化相关代码，使用标准LoRA。这样会需要更多显存，但功能相同。

### Q2: 显存不足怎么办？

**A**: 
- 减小 `per_device_train_batch_size`
- 减小 `gradient_accumulation_steps`
- 减小 LoRA 的 `r` 参数（如从8改为4）
- 使用梯度检查点：`gradient_checkpointing=True`

### Q3: 模型下载失败怎么办？

**A**: 
- 确保设置了 `HF_ENDPOINT=https://hf-mirror.com`
- 或者手动从 https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct 下载模型到本地

### Q4: 训练中断了怎么办？

**A**: 训练脚本支持断点续训，重新运行相同命令即可从最新检查点继续。

## 输出文件说明

训练完成后，`outputs/checkpoints/final/` 目录包含：
- `adapter_config.json`: LoRA配置
- `adapter_model.bin`: LoRA权重文件（仅几MB）
- `training_args.bin`: 训练参数

这些文件将在阶段二用于合并权重。

