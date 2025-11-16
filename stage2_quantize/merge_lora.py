"""
合并LoRA权重到基础模型
生成完整的微调后模型
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def merge_lora_weights(base_model_name, lora_path, output_path):
    """合并LoRA权重"""
    print(f"加载基础模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="./models/cache",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        cache_dir="./models/cache"
    )
    
    print(f"加载LoRA权重: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("合并LoRA权重到基础模型...")
    merged_model = model.merge_and_unload()
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    print(f"保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(output_path)
    
    print("合并完成！")
    print(f"模型已保存到: {output_path}")
    
    # 显示模型大小
    total_size = 0
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024**3)  # GB
            total_size += size
            print(f"  {file}: {size:.2f} GB")
    print(f"总大小: {total_size:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型名称或路径"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出路径"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(args.base_model, args.lora_path, args.output_path)

if __name__ == "__main__":
    main()

