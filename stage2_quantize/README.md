# 阶段二：模型量化与评估

本目录包含模型量化、转换和评估的相关脚本。

## 文件说明

- `merge_lora.py` - 合并LoRA权重到基础模型
- `convert_to_gguf.sh` - 转换模型为GGUF格式（Linux）
- `convert_to_gguf.bat` - 转换模型为GGUF格式（Windows）
- `quantize_model.sh` - 量化模型脚本（Linux）
- `quantize_model.bat` - 量化模型脚本（Windows）
- `test_quantized_model.py` - **批量评估量化模型脚本**

## 批量评估脚本使用指南

### 基本用法

```bash
# 激活虚拟环境
cd /home/ljh/AI
source venv/bin/activate

# 测试单个模型
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 常用参数

- `--model_path`: 单个模型路径
- `--model_paths`: 多个模型路径（用于对比）
- `--test_data`: 测试数据路径（默认: `data/processed/test.json`）
- `--max_samples`: 最大测试样本数（默认: 全部）
- `--temperature`: 温度参数（默认: 0.1）
- `--max_tokens`: 最大生成token数（默认: 10）
- `--threads`: CPU线程数（-1表示自动）
- `--gpu_layers`: GPU层数（0表示仅CPU，>0表示使用GPU）
- `--output`: 保存详细结果的JSON文件路径
- `--verbose`: 显示详细信息

### 使用示例

#### 1. 快速测试（100个样本）

```bash
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --max_samples 100
```

#### 2. 完整测试（全部样本）

```bash
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

#### 3. 对比多个量化模型

```bash
python stage2_quantize/test_quantized_model.py \
    --model_paths \
        outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf \
        outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        outputs/gguf/qwen2.5-0.5b-instruct-q8_0.gguf \
    --max_samples 200
```

#### 4. 使用GPU加速

```bash
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --gpu_layers 20 \
    --max_samples 100
```

#### 5. 保存详细结果

```bash
python stage2_quantize/test_quantized_model.py \
    --model_path outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --max_samples 100 \
    --output evaluation_results.json
```

### 评估指标

脚本会输出以下评估指标：

1. **准确率 (Accuracy)**: 整体预测正确的比例
2. **精确率 (Precision)**: 预测为正面的样本中，真正为正面的比例
3. **召回率 (Recall)**: 所有真正的正面样本中，被正确预测的比例
4. **F1分数**: 精确率和召回率的调和平均
5. **混淆矩阵**: TP, FP, TN, FN 的详细统计
6. **推理时间**: 平均每个样本的推理耗时

### 输出格式

评估结果会显示在终端，并可以保存为JSON文件。JSON文件包含：
- 整体评估指标
- 每个样本的详细预测结果
- 混淆矩阵
- 性能统计

### 注意事项

1. **虚拟环境**: 确保已激活虚拟环境（Ubuntu 24.04要求）
2. **llama-cli路径**: 脚本会自动查找llama-cli，如果找不到，请使用 `--llama_cli_path` 指定
3. **测试数据**: 默认使用 `data/processed/test.json`，确保该文件存在
4. **GPU支持**: 如果编译了GPU版本，使用 `--gpu_layers` 参数启用GPU加速

### 常见问题

**Q: 找不到llama-cli？**
A: 确保已编译llama.cpp，脚本会自动查找以下位置：
- `llama.cpp/llama-cli`
- `llama.cpp/build/bin/llama-cli`
- `llama.cpp/build/bin/Release/llama-cli.exe`

**Q: 测试速度太慢？**
A: 
- 使用 `--max_samples` 限制测试样本数
- 使用 `--gpu_layers` 启用GPU加速
- 减少 `--max_tokens` 参数

**Q: 如何对比不同量化级别的模型？**
A: 使用 `--model_paths` 参数指定多个模型路径，脚本会自动对比并输出对比报告。


