"""
下载ChnSentiCorp中文情感数据集
使用多种方式确保在中国大陆可访问
"""

import os
import json
import requests
from datasets import load_dataset
from pathlib import Path

# 设置HuggingFace镜像（适合中国大陆）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_from_huggingface_direct():
    """直接从HuggingFace下载数据文件"""
    print("尝试方法1: 从HuggingFace直接下载数据文件...")
    
    try:
        # 尝试使用不同的数据集加载方式
        # 使用trust_remote_code=True可能可以解决某些问题
        dataset = load_dataset(
            "seamew/ChnSentiCorp",
            cache_dir="./data/cache",
            trust_remote_code=True
        )
        return dataset
    except Exception as e:
        print(f"方法1失败: {e}")
        return None

def download_from_github():
    """从GitHub下载数据集"""
    print("\n尝试方法2: 从GitHub下载数据集...")
    
    try:
        # 使用GitHub raw content URL
        urls = {
            "train": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
        }
        
        output_dir = "data/chn_senti_corp"
        os.makedirs(output_dir, exist_ok=True)
        
        import pandas as pd
        
        # 下载训练数据
        print("正在下载数据文件...")
        response = requests.get(urls["train"], timeout=30)
        response.encoding = 'utf-8'
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, "all.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        # 读取CSV并转换为JSON格式
        df = pd.read_csv(csv_path, header=None, names=["label", "text"])
        
        # 划分数据集 (80% train, 10% val, 10% test)
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
        
        # 保存为JSON
        for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            output_data = []
            for _, row in split_df.iterrows():
                output_data.append({
                    "text": str(row["text"]),
                    "label": int(row["label"])
                })
            
            output_file = os.path.join(output_dir, f"{split_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {split_name} 集到 {output_file}")
            print(f"  - 样本数量: {len(output_data)}")
        
        print("\n数据集下载完成！")
        return True
        
    except Exception as e:
        print(f"方法2失败: {e}")
        return False

def create_sample_dataset():
    """创建示例数据集（用于测试）"""
    print("\n尝试方法3: 创建示例数据集...")
    
    # 创建一些示例数据用于测试
    sample_data = {
        "train": [
            {"text": "这部电影真的很棒，演员演技出色，剧情紧凑。", "label": 1},
            {"text": "非常失望，剧情拖沓，演员表演生硬。", "label": 0},
            {"text": "画面精美，音乐动听，值得一看。", "label": 1},
            {"text": "无聊透顶，浪费时间。", "label": 0},
            {"text": "故事情节引人入胜，推荐观看。", "label": 1},
        ] * 100,  # 重复以增加数据量
        "validation": [
            {"text": "整体不错，但有些地方可以改进。", "label": 1},
            {"text": "不太喜欢，感觉一般。", "label": 0},
        ] * 50,
        "test": [
            {"text": "非常精彩，强烈推荐！", "label": 1},
            {"text": "很糟糕，不推荐。", "label": 0},
        ] * 50,
    }
    
    output_dir = "data/chn_senti_corp"
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, data in sample_data.items():
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已创建 {split_name} 集: {output_file} ({len(data)} 样本)")
    
    print("\n注意: 这是示例数据集，仅用于测试。")
    print("建议手动下载完整数据集: https://github.com/SophonPlus/ChineseNlpCorpus")
    return True

def download_chn_senti_corp():
    """下载ChnSentiCorp数据集（多种方法尝试）"""
    print("="*60)
    print("开始下载ChnSentiCorp中文情感数据集")
    print("="*60)
    
    # 方法1: 尝试从HuggingFace加载
    dataset = download_from_huggingface_direct()
    
    if dataset is not None:
        # 创建输出目录
        output_dir = "data/chn_senti_corp"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON格式
        for split in ["train", "validation", "test"]:
            if split in dataset:
                output_file = os.path.join(output_dir, f"{split}.json")
                dataset[split].to_json(output_file, force_ascii=False)
                print(f"已保存 {split} 集到 {output_file}")
                print(f"  - 样本数量: {len(dataset[split])}")
        
        print("\n数据集下载完成！")
        print(f"数据保存在: {output_dir}")
        
        # 显示数据示例
        if "train" in dataset:
            print("\n数据示例:")
            example = dataset["train"][0]
            print(f"  文本: {example.get('text', 'N/A')}")
            print(f"  标签: {example.get('label', 'N/A')}")
        
        return dataset
    
    # 方法2: 从GitHub下载
    if download_from_github():
        return True
    
    # 方法3: 创建示例数据集
    print("\n所有自动下载方法都失败了。")
    print("选项:")
    print("1. 使用示例数据集继续（仅用于测试）")
    print("2. 手动下载数据集")
    print("\n手动下载步骤:")
    print("1. 访问: https://github.com/SophonPlus/ChineseNlpCorpus")
    print("2. 下载 ChnSentiCorp 数据集")
    print("3. 将数据文件放在 data/chn_senti_corp/ 目录下")
    print("4. 运行 preprocess_data.py 进行预处理")
    
    response = input("\n是否创建示例数据集用于测试？(y/n): ").strip().lower()
    if response == 'y':
        create_sample_dataset()
        return True
    else:
        print("请手动下载数据集后重试。")
        return False

if __name__ == "__main__":
    download_chn_senti_corp()

