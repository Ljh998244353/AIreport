# 轻量化大模型移动端部署研究

## 项目简介

本项目实现从模型选择、高效微调、量化转换到移动端部署的一体化流程。使用 Qwen2.5-0.5B-Instruct 模型，通过 LoRA + QLoRA 进行微调，并转换为 GGUF 格式进行量化部署。

## 技术栈

- **基础模型**: Qwen2.5-0.5B-Instruct (0.49B参数)
- **微调方法**: LoRA + QLoRA (4-bit量化训练)
- **量化格式**: GGUF (4-bit Q4_K_M)
- **推理引擎**: llama.cpp

## 项目结构

```
work/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── stage1_finetune/         # 阶段一：模型微调
│   ├── download_data.py     # 数据下载脚本
│   ├── preprocess_data.py   # 数据预处理脚本
│   ├── train_lora.py        # LoRA微调训练脚本
│   └── evaluate.py          # 模型评估脚本
├── stage2_quantize/         # 阶段二：模型转换与量化
│   ├── merge_lora.py        # LoRA权重合并脚本
│   ├── convert_to_gguf.sh   # 转换为GGUF格式脚本
│   └── quantize_model.sh    # 模型量化脚本
└── config/                  # 配置文件
    └── training_config.yaml # 训练配置
```

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU训练)
- 至少 8GB 显存 (使用QLoRA可降低到4GB)
- 至少 20GB 磁盘空间

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖（使用国内镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 阶段一：模型微调

详细步骤请参考 [STAGE1_GUIDE.md](STAGE1_GUIDE.md)

### 3. 阶段二：模型转换与量化

详细步骤请参考 [STAGE2_GUIDE.md](STAGE2_GUIDE.md)

## 注意事项

1. **网络环境**: 所有脚本已配置使用国内镜像源（HuggingFace镜像、PyPI镜像）
2. **显存优化**: 使用QLoRA技术，4GB显存即可完成训练
3. **模型下载**: 首次运行会自动从HuggingFace镜像站下载模型

## 参考资源

- [Qwen2.5 官方文档](https://qwen.readthedocs.io/)
- [llama.cpp 官方仓库](https://github.com/ggerganov/llama.cpp)
- [PEFT 文档](https://huggingface.co/docs/peft)

