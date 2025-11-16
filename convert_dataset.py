"""
将ChnSentiCorp_htl_all数据集转换为项目所需格式
"""

import os
import json
import csv
import random

def convert_dataset():
    """转换数据集格式"""
    print("=" * 60)
    print("转换ChnSentiCorp数据集")
    print("=" * 60)
    
    # 读取CSV文件
    csv_path = "ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
    print(f"\n读取CSV文件: {csv_path}")
    
    try:
        # 读取CSV文件
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 标准化字段名
                text = row.get('review', row.get('text', ''))
                label = row.get('label', '')
                
                if text and label:
                    try:
                        label_int = int(label)
                        if label_int in [0, 1]:
                            data.append({
                                'text': text.strip(),
                                'label': label_int
                            })
                    except ValueError:
                        continue
        
        print(f"原始数据量: {len(data)} 条")
        
        # 数据清洗：去除重复
        print("\n清洗数据...")
        seen_texts = set()
        unique_data = []
        for item in data:
            if item['text'] not in seen_texts:
                seen_texts.add(item['text'])
                unique_data.append(item)
        
        data = unique_data
        print(f"清洗后数据量: {len(data)} 条")
        
        # 显示标签分布
        print("\n标签分布:")
        positive_count = sum(1 for item in data if item['label'] == 1)
        negative_count = sum(1 for item in data if item['label'] == 0)
        print(f"  正面 (1): {positive_count} 条")
        print(f"  负面 (0): {negative_count} 条")
        
        # 划分数据集：80%训练，10%验证，10%测试
        print("\n划分数据集...")
        # 按标签分组
        positive_data = [item for item in data if item['label'] == 1]
        negative_data = [item for item in data if item['label'] == 0]
        
        # 设置随机种子以确保可重复
        random.seed(42)
        random.shuffle(positive_data)
        random.shuffle(negative_data)
        
        # 划分正面数据
        pos_train_size = int(len(positive_data) * 0.8)
        pos_temp_size = int(len(positive_data) * 0.1)
        pos_train = positive_data[:pos_train_size]
        pos_val = positive_data[pos_train_size:pos_train_size + pos_temp_size]
        pos_test = positive_data[pos_train_size + pos_temp_size:]
        
        # 划分负面数据
        neg_train_size = int(len(negative_data) * 0.8)
        neg_temp_size = int(len(negative_data) * 0.1)
        neg_train = negative_data[:neg_train_size]
        neg_val = negative_data[neg_train_size:neg_train_size + neg_temp_size]
        neg_test = negative_data[neg_train_size + neg_temp_size:]
        
        # 合并
        train_data = pos_train + neg_train
        val_data = pos_val + neg_val
        test_data = pos_test + neg_test
        
        # 打乱顺序
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        print(f"训练集: {len(train_data)} 条")
        print(f"验证集: {len(val_data)} 条")
        print(f"测试集: {len(test_data)} 条")
        
        # 创建输出目录
        output_dir = "data/chn_senti_corp"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON格式
        print("\n保存数据...")
        for split_name, split_data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
            output_file = os.path.join(output_dir, f"{split_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            pos_count = sum(1 for d in split_data if d['label'] == 1)
            neg_count = sum(1 for d in split_data if d['label'] == 0)
            print(f"  ✅ {split_name}.json: {len(split_data)} 条")
            print(f"     正面: {pos_count} 条, 负面: {neg_count} 条")
        
        print(f"\n✅ 数据集转换完成！")
        print(f"数据保存在: {output_dir}/")
        
        # 显示数据示例
        print("\n数据示例:")
        example = train_data[0]
        text_preview = example['text'][:50] + "..." if len(example['text']) > 50 else example['text']
        print(f"  文本: {text_preview}")
        print(f"  标签: {example['label']} ({'正面' if example['label'] == 1 else '负面'})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_dataset()
    if success:
        print("\n" + "=" * 60)
        print("下一步: 运行数据预处理脚本")
        print("python stage1_finetune/preprocess_data.py")
        print("=" * 60)
    else:
        print("\n请检查CSV文件格式是否正确")

