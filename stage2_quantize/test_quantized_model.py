"""
测试量化后的GGUF模型
使用llama-cpp-python库进行推理测试
"""

import argparse
import json
import os

def test_with_llama_cpp(model_path, test_data_path):
    """使用llama-cpp-python测试模型"""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("错误: 未安装 llama-cpp-python")
        print("请运行: pip install llama-cpp-python -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return
    
    print(f"加载模型: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4,
        verbose=False
    )
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 测试几个样本
    print("\n开始测试...")
    correct = 0
    total = min(100, len(test_data))  # 测试前100个样本
    
    for i, item in enumerate(test_data[:total]):
        text = item["text"]
        true_label = item["label"]
        
        # 创建提示
        prompt = f"""<|im_start|>system
你是一个情感分析助手，需要判断文本的情感倾向。<|im_end|>
<|im_start|>user
请分析以下文本的情感倾向（正面/负面）：
{text}<|im_end|>
<|im_start|>assistant
"""
        
        # 生成
        response = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["<|im_end|>"],
            echo=False
        )
        
        output = response["choices"][0]["text"].strip()
        
        # 判断预测标签
        if "正面" in output:
            pred_label = 1
        elif "负面" in output:
            pred_label = 0
        else:
            pred_label = 1 if len(output) > 0 else 0
        
        if pred_label == true_label:
            correct += 1
        
        if (i + 1) % 10 == 0:
            print(f"进度: {i+1}/{total}, 准确率: {correct/(i+1):.4f}")
    
    accuracy = correct / total
    print(f"\n测试完成！")
    print(f"测试样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.4f}")

def test_with_llama_cli(model_path, test_data_path):
    """使用llama-cli命令行工具测试（需要手动运行）"""
    print("使用llama-cli测试模型:")
    print(f"模型路径: {model_path}")
    print(f"\n测试命令示例:")
    print(f'llama-cli -m "{model_path}" -p "这部电影很好看" -n 50')
    print(f"\n批量测试需要编写脚本调用llama-cli")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="GGUF模型路径")
    parser.add_argument("--test_data", type=str, default="data/processed/test.json", help="测试数据路径")
    parser.add_argument("--method", type=str, default="llama_cpp", choices=["llama_cpp", "cli"], help="测试方法")
    args = parser.parse_args()
    
    if args.method == "llama_cpp":
        test_with_llama_cpp(args.model_path, args.test_data)
    else:
        test_with_llama_cli(args.model_path, args.test_data)

if __name__ == "__main__":
    main()

