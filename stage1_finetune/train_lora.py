"""
LoRA微调训练脚本
使用QLoRA（4-bit量化）进行参数高效微调
"""

import os
import json
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置tokenizers并行处理，避免fork警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """加载预处理后的数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def tokenize_function(examples, tokenizer, max_length=512):
    """对数据进行tokenization"""
    # 当batched=True时，examples["prompt"]已经是列表
    texts = examples["prompt"]
    
    # Tokenize - 不设置labels，让DataCollatorForLanguageModeling自动处理
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # 不在tokenize时padding，由DataCollator处理
        return_tensors=None,
    )
    
    # 注意：不在这里设置labels
    # DataCollatorForLanguageModeling(mlm=False)会自动将input_ids作为labels
    
    return tokenized

def setup_model_and_tokenizer(config):
    """设置模型和tokenizer"""
    model_config = config["model"]
    lora_config = config["lora"]
    
    print(f"加载模型: {model_config['name']}")
    print(f"使用镜像站: {os.environ.get('HF_ENDPOINT', '默认')}")
    
    # 确保使用镜像站
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"已设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=True,
        cache_dir="./models/cache",
        resume_download=True  # 支持断点续传
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置4-bit量化（QLoRA）
    if model_config.get("use_4bit", False):
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./models/cache",
                resume_download=True  # 支持断点续传
            )
            
            print("已启用4-bit量化（QLoRA）")
        except Exception as e:
            print(f"4-bit量化加载失败: {e}")
            print("回退到标准加载方式...")
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./models/cache",
                dtype=torch.float16,  # 使用 dtype 替代已弃用的 torch_dtype
                resume_download=True  # 支持断点续传
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models/cache",
            torch_dtype=torch.float16,
            resume_download=True  # 支持断点续传
        )
    
    # 准备模型用于k-bit训练
    if model_config.get("use_4bit", False):
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载数据
    print("加载数据...")
    train_data = load_data(config["data"]["train_file"])
    val_data = load_data(config["data"]["val_file"])
    
    # 转换为datasets格式
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenization
    print("Tokenizing数据...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["data"]["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["data"]["max_length"]),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # 数据整理器 - 用于动态padding
    # 对于因果语言模型，DataCollatorForLanguageModeling会自动将input_ids作为labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 使用因果语言模型，不是掩码语言模型
        pad_to_multiple_of=8,  # 为了效率，padding到8的倍数
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=config["training"].get("eval_strategy", config["training"].get("evaluation_strategy", "steps")),
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        fp16=config["training"].get("fp16", True),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 0),  # 设为0避免fork问题
        report_to=config.get("report_to", "none"),
        seed=config.get("seed", 42),
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_output_dir = os.path.join(config["training"]["output_dir"], "final")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\n训练完成！模型已保存到: {final_output_dir}")

if __name__ == "__main__":
    main()

