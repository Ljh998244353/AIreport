"""
模型评估脚本
评估微调后的模型在测试集上的性能
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_data(data_path):
    """加载测试数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def predict_sentiment(model, tokenizer, text, device="cuda"):
    """预测文本情感"""
    # 创建提示
    prompt = f"""<|im_start|>system
你是一个情感分析助手，需要判断文本的情感倾向。<|im_end|>
<|im_start|>user
请分析以下文本的情感倾向（正面/负面）：
{text}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # 提取标签
    if "正面" in response:
        return 1
    elif "负面" in response:
        return 0
    else:
        # 如果模型没有输出明确的标签，尝试其他方式判断
        return 1 if len(response.strip()) > 0 else 0

def evaluate_model(model_path, test_data_path, base_model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """评估模型"""
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载LoRA权重
    print(f"加载LoRA权重: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    test_data = load_data(test_data_path)
    
    # 预测
    print("开始预测...")
    predictions = []
    true_labels = []
    
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(test_data)}")
        
        text = item["text"]
        true_label = item["label"]
        
        pred_label = predict_sentiment(model, tokenizer, text, device)
        
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )
    
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(true_labels, predictions, target_names=["负面", "正面"]))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径（LoRA权重目录）")
    parser.add_argument("--test_data", type=str, default="data/processed/test.json", help="测试数据路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="基础模型名称")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data, args.base_model)

if __name__ == "__main__":
    main()

