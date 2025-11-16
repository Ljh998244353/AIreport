"""
数据预处理脚本
将ChnSentiCorp数据集转换为模型训练格式
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_data(data_dir="data/chn_senti_corp"):
    """加载原始数据"""
    data = []
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
                data.extend(split_data)
    
    return pd.DataFrame(data)

def create_prompt(text, label=None):
    """创建训练提示模板（Qwen2.5格式）"""
    if label is not None:
        label_text = "正面" if label == 1 else "负面"
        prompt = f"""<|im_start|>system
你是一个情感分析助手，需要判断文本的情感倾向。<|im_end|>
<|im_start|>user
请分析以下文本的情感倾向（正面/负面）：
{text}<|im_end|>
<|im_start|>assistant
{label_text}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>system
你是一个情感分析助手，需要判断文本的情感倾向。<|im_end|>
<|im_start|>user
请分析以下文本的情感倾向（正面/负面）：
{text}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def preprocess_data():
    """预处理数据"""
    print("加载原始数据...")
    df = load_data()
    
    print(f"原始数据量: {len(df)}")
    
    # 数据清洗
    print("清洗数据...")
    # 去除空值
    df = df.dropna(subset=["text", "label"])
    # 去除重复
    df = df.drop_duplicates(subset=["text"])
    # 确保标签为整数
    df["label"] = df["label"].astype(int)
    
    print(f"清洗后数据量: {len(df)}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models/cache"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建提示文本
    print("创建训练提示...")
    df["prompt"] = df.apply(lambda row: create_prompt(row["text"], row["label"]), axis=1)
    
    # 划分数据集：80%训练，10%验证，10%测试
    print("划分数据集...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
    
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 保存处理后的数据
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_data = []
        for _, row in split_df.iterrows():
            output_data.append({
                "text": row["text"],
                "label": row["label"],
                "prompt": row["prompt"]
            })
        
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {split_name} 集到 {output_file}")
    
    # 显示统计信息
    print("\n数据统计:")
    for split_name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        print(f"{split_name}:")
        print(f"  正面: {sum(split_df['label'] == 1)}")
        print(f"  负面: {sum(split_df['label'] == 0)}")
    
    print("\n预处理完成！")

if __name__ == "__main__":
    preprocess_data()

