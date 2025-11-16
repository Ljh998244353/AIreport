# 阶段二：模型转换与量化详细指南

## 概述

本阶段将完成：
1. 合并LoRA权重到基础模型
2. 将模型转换为GGUF格式
3. 对模型进行量化（4-bit Q4_K_M）
4. 验证量化后的模型

## 前置要求

- 完成阶段一的模型微调
- 安装 llama.cpp 工具链
- 准备至少 10GB 磁盘空间

## 步骤详解

### 步骤1：合并LoRA权重

#### 1.1 运行合并脚本

```bash
python stage2_quantize/merge_lora.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_path outputs/checkpoints/final \
    --output_path outputs/merged_model
```

该脚本会：
- 加载基础模型和LoRA权重
- 合并权重到完整模型
- 保存为 HuggingFace 格式到 `outputs/merged_model/`

**注意**: 合并后的模型大小约为 1GB（FP16格式）

### 步骤2：安装 llama.cpp

#### 2.1 克隆仓库

```bash
# 使用国内镜像（Gitee或GitHub镜像）
git clone https://gitee.com/mirrors/llama.cpp.git
# 或
git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
```

#### 2.2 编译（Windows）

**方法一：使用CMake（推荐）**

```bash
# 安装CMake（如果未安装）
# 下载地址：https://cmake.org/download/

# 创建build目录
mkdir build
cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . --config Release
```

**方法二：使用Visual Studio**

1. 打开 `llama.cpp/llama.cpp.sln`
2. 选择 Release 配置
3. 生成解决方案

编译完成后，可执行文件在 `build/bin/Release/` 或 `build/Release/` 目录。

#### 2.3 编译（Linux/Mac）

```bash
# Linux
make

# Mac (Apple Silicon)
make CC=clang CXX=clang++ LLAMA_METAL=1
```

### 步骤3：转换模型为GGUF格式

#### 3.1 安装Python依赖

```bash
# 在llama.cpp目录下
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3.2 运行转换脚本

```bash
# 确保在llama.cpp目录下
python convert-hf-to-gguf.py \
    ../work/outputs/merged_model \
    --outdir ../work/outputs/gguf \
    --outtype f16
```

参数说明：
- `../work/outputs/merged_model`: 合并后的模型路径（相对于llama.cpp目录）
- `--outdir`: 输出目录
- `--outtype f16`: 输出为FP16格式（后续再量化）

**输出**: `outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf` (约1GB)

### 步骤4：量化模型

#### 4.1 运行量化工具

```bash
# Windows (在llama.cpp/build/bin/Release目录下)
.\quantize.exe ..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-f16.gguf ..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf Q4_K_M

# Linux/Mac
./quantize ../work/outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf ../work/outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf Q4_K_M
```

量化级别说明：
- `Q4_0`: 4-bit量化，最快
- `Q4_K_M`: 4-bit量化，中等质量（推荐）
- `Q4_K_S`: 4-bit量化，较小文件
- `Q5_0`: 5-bit量化，更好质量
- `Q8_0`: 8-bit量化，接近FP16质量

**输出**: `outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` (约300MB)

### 步骤5：验证量化模型

#### 5.1 测试推理

```bash
# Windows
.\llama-cli.exe -m ..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf -p "这部电影很好看" -n 50

# Linux/Mac
./llama-cli -m ../work/outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "这部电影很好看" -n 50
```

#### 5.2 运行评估脚本

```bash
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --test_data data/processed/test.json
```

## 常见问题

### Q1: convert-hf-to-gguf.py 找不到怎么办？

**A**: 
- 确保在 llama.cpp 根目录下运行
- 检查脚本名称，可能是 `convert.py` 或 `convert_hf_to_gguf.py`
- 查看 llama.cpp 最新文档确认正确的脚本名

### Q2: 转换失败，提示模型格式不支持？

**A**: 
- 确保使用的是最新版本的 llama.cpp
- Qwen2.5 模型需要 llama.cpp 的较新版本支持
- 如果仍不支持，可能需要手动修改转换脚本或等待更新

### Q3: 量化后模型性能下降太多？

**A**: 
- 尝试使用 Q5_0 或 Q8_0 量化级别
- 检查原始模型微调质量
- 某些任务对量化更敏感

### Q4: Windows编译失败？

**A**: 
- 确保安装了 Visual Studio 2019+ 或 Build Tools
- 使用CMake GUI工具配置
- 参考 llama.cpp 官方Windows编译文档

## 输出文件说明

最终输出：
- `outputs/merged_model/`: 合并后的完整模型（HuggingFace格式，约1GB）
- `outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf`: FP16格式GGUF（约1GB）
- `outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf`: 量化后的GGUF（约300MB，用于移动端）

## 下一步：移动端部署

量化后的GGUF模型可以：
1. 使用 llama.cpp 在Android/iOS上编译运行
2. 使用 llama.cpp 的绑定库（如llama-cpp-python）在移动应用中使用
3. 转换为其他移动端框架格式（如ONNX、CoreML等）

