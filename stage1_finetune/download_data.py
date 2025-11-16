"""
下载ChnSentiCorp中文情感数据集
使用HuggingFace镜像站以确保在中国大陆可访问
"""

import os
from datasets import load_dataset
import json

# 设置HuggingFace镜像（适合中国大陆）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_chn_senti_corp():
    """下载ChnSentiCorp数据集"""
    print("正在从HuggingFace镜像站下载ChnSentiCorp数据集...")
    
    try:
        # 从HuggingFace加载数据集
        dataset = load_dataset("seamew/ChnSentiCorp", cache_dir="./data/cache")
        
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
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n尝试备用方案：手动下载数据集...")
        print("请访问: https://github.com/SophonPlus/ChineseNlpCorpus")
        raise

if __name__ == "__main__":
    download_chn_senti_corp()

