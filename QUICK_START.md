# 快速开始指南

本文档提供完整的、逐步的操作指南，适合在中国大陆网络环境下使用。

## 环境准备

### 1. 安装Python和CUDA

- **Python**: 下载并安装 Python 3.8+ (推荐 3.10)
  - 下载地址: https://www.python.org/downloads/
  - 安装时勾选 "Add Python to PATH"

- **CUDA**: 如果使用GPU训练，需要安装CUDA 11.8+
  - 下载地址: https://developer.nvidia.com/cuda-downloads
  - 验证安装: `nvcc --version`

### 2. 创建项目环境

```powershell
# 进入项目目录
cd C:\Users\ljh\Desktop\my\class\docs\资料\人工智能\课程设计\报告\work

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1
# 如果PowerShell执行策略限制，运行:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 或者使用CMD:
# venv\Scripts\activate.bat
```

### 3. 安装依赖

```powershell
# 使用清华大学镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果bitsandbytes安装失败（Windows常见问题），可以跳过，后续使用标准LoRA
# 或者尝试安装预编译版本
```

### 4. 配置HuggingFace镜像

```powershell
# 设置环境变量（当前会话）
$env:HF_ENDPOINT="https://hf-mirror.com"

# 永久设置（可选）
[System.Environment]::SetEnvironmentVariable('HF_ENDPOINT', 'https://hf-mirror.com', 'User')
```

---

## 阶段一：模型微调

### 步骤1: 下载数据

```powershell
# 确保已激活虚拟环境
python stage1_finetune/download_data.py
```

**预期输出**:
- 数据保存在 `data/chn_senti_corp/` 目录
- 包含 train.json, validation.json, test.json

### 步骤2: 预处理数据

```powershell
python stage1_finetune/preprocess_data.py
```

**预期输出**:
- 处理后的数据保存在 `data/processed/` 目录
- 显示数据统计信息（训练集/验证集/测试集数量）

**注意事项**:
- 首次运行会自动下载 Qwen2.5 tokenizer（约几MB）
- 如果下载失败，检查网络连接和HF_ENDPOINT设置

### 步骤3: 配置训练参数（可选）

编辑 `config/training_config.yaml`，根据需要调整：
- `per_device_train_batch_size`: 根据显存调整（4GB显存建议设为2）
- `learning_rate`: 学习率（默认2e-4）
- `num_train_epochs`: 训练轮次（默认3）

### 步骤4: 开始训练

```powershell
python stage1_finetune/train_lora.py --config config/training_config.yaml
```

**训练过程**:
1. 自动下载 Qwen2.5-0.5B-Instruct 模型（约1GB，首次运行）
2. 使用4-bit量化加载模型（如果bitsandbytes可用）
3. 添加LoRA适配器
4. 开始训练，显示进度条和损失值

**预期时间**:
- GPU (RTX 3060 12GB): 30-60分钟
- GPU (4GB显存): 1-2小时
- CPU: 数小时（不推荐）

**输出文件**:
- `outputs/checkpoints/checkpoint-XXX/`: 训练检查点
- `outputs/checkpoints/final/`: 最终模型（LoRA权重）

### 步骤5: 评估模型

```powershell
python stage1_finetune/evaluate.py --model_path outputs/checkpoints/final
```

**预期输出**:
- 准确率、精确率、召回率、F1分数
- 详细分类报告

---

## 阶段二：模型转换与量化

### 步骤1: 合并LoRA权重

```powershell
python stage2_quantize/merge_lora.py `
    --base_model Qwen/Qwen2.5-0.5B-Instruct `
    --lora_path outputs/checkpoints/final `
    --output_path outputs/merged_model
```

**预期输出**:
- 合并后的完整模型保存在 `outputs/merged_model/`
- 模型大小约1GB（FP16格式）

### 步骤2: 安装llama.cpp

#### 方法一：使用Git克隆（推荐）

```powershell
# 在项目目录外克隆（例如在C盘根目录）
cd C:\
git clone https://gitee.com/mirrors/llama.cpp.git
# 或使用GitHub镜像
# git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
```

#### 方法二：下载ZIP（如果Git不可用）

1. 访问 https://gitee.com/mirrors/llama.cpp
2. 下载ZIP文件并解压

### 步骤3: 编译llama.cpp（Windows）

#### 前置要求
- 安装 Visual Studio 2019+ 或 Build Tools
- 安装 CMake (https://cmake.org/download/)

#### 编译步骤

```powershell
# 在llama.cpp目录下
mkdir build
cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . --config Release
```

**编译完成后**:
- 可执行文件在 `build\bin\Release\` 目录
- 包含 `quantize.exe`, `llama-cli.exe` 等

### 步骤4: 安装Python依赖（用于转换）

```powershell
# 在llama.cpp目录下
cd C:\llama.cpp  # 或你的llama.cpp路径
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤5: 转换为GGUF格式

```powershell
# 在llama.cpp目录下运行
python convert-hf-to-gguf.py `
    C:\Users\ljh\Desktop\my\class\docs\资料\人工智能\课程设计\报告\work\outputs\merged_model `
    --outdir C:\Users\ljh\Desktop\my\class\docs\资料\人工智能\课程设计\报告\work\outputs\gguf `
    --outtype f16
```

**注意**: 
- 将路径替换为你的实际路径
- 如果脚本名不同，查看llama.cpp目录下的实际脚本名

**预期输出**:
- `outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf` (约1GB)

### 步骤6: 量化模型

```powershell
# 在llama.cpp\build\bin\Release目录下运行
cd C:\llama.cpp\build\bin\Release

.\quantize.exe `
    ..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-f16.gguf `
    ..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
    Q4_K_M
```

**注意**: 调整路径为你的实际路径

**预期输出**:
- `outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` (约300MB)

### 步骤7: 测试量化模型（可选）

```powershell
# 安装llama-cpp-python（可选，用于Python测试）
pip install llama-cpp-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行测试脚本
python stage2_quantize/test_quantized_model.py `
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

或使用命令行工具：

```powershell
# 在llama.cpp\build\bin\Release目录下
.\llama-cli.exe -m "..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf" -p "这部电影很好看" -n 50
```

---

## 常见问题解决

### Q1: bitsandbytes安装失败

**解决方案**:
1. 修改 `stage1_finetune/train_lora.py`，将 `use_4bit: false` 设为false
2. 或使用标准LoRA（需要更多显存）

### Q2: 模型下载失败

**解决方案**:
1. 确认设置了 `HF_ENDPOINT=https://hf-mirror.com`
2. 手动从 https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct 下载
3. 使用 `huggingface-cli` 下载到本地

### Q3: 显存不足

**解决方案**:
1. 减小 `per_device_train_batch_size` (如改为1或2)
2. 减小 `gradient_accumulation_steps`
3. 减小LoRA的 `r` 参数（如从8改为4）
4. 使用CPU训练（很慢）

### Q4: convert-hf-to-gguf.py找不到

**解决方案**:
1. 检查llama.cpp版本，确保支持Qwen2.5
2. 脚本名可能是 `convert.py` 或其他
3. 查看llama.cpp文档确认正确的转换方法

### Q5: Windows编译llama.cpp失败

**解决方案**:
1. 确保安装了Visual Studio 2019+和CMake
2. 使用CMake GUI工具配置
3. 参考llama.cpp官方Windows编译文档
4. 或使用预编译版本（如果有）

---

## 项目文件说明

```
work/
├── README.md                    # 项目总览
├── QUICK_START.md              # 本文件：快速开始指南
├── STAGE1_GUIDE.md             # 阶段一详细指南
├── STAGE2_GUIDE.md             # 阶段二详细指南
├── requirements.txt            # Python依赖
├── config/
│   └── training_config.yaml    # 训练配置
├── stage1_finetune/            # 阶段一脚本
│   ├── download_data.py        # 下载数据
│   ├── preprocess_data.py      # 预处理数据
│   ├── train_lora.py           # 训练模型
│   └── evaluate.py             # 评估模型
└── stage2_quantize/            # 阶段二脚本
    ├── merge_lora.py           # 合并权重
    ├── convert_to_gguf.bat     # 转换脚本(Windows)
    ├── quantize_model.bat      # 量化脚本(Windows)
    └── test_quantized_model.py # 测试脚本
```

---

## 预期时间线

- **环境配置**: 30分钟-1小时
- **数据准备**: 10-20分钟
- **模型微调**: 30分钟-2小时（取决于硬件）
- **模型转换**: 10-20分钟
- **模型量化**: 5-10分钟

**总计**: 约2-4小时（首次运行，包含下载时间）

---

## 下一步

完成以上步骤后，你将得到：
1. 微调后的模型（LoRA权重）
2. 合并后的完整模型（HuggingFace格式）
3. 量化后的GGUF模型（可用于移动端）

可以继续研究：
- 移动端部署（Android/iOS）
- 模型性能优化
- 不同量化级别的对比
- 其他下游任务的应用

