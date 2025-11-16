"""
批量评估量化后的GGUF模型
支持使用llama-cli命令行工具进行批量测试
"""

import argparse
import json
import os
import subprocess
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_prompt(text: str) -> str:
    """创建推理提示模板"""
    prompt = f"""<|im_start|>system
你是一个情感分析助手，需要判断文本的情感倾向。<|im_end|>
<|im_start|>user
请分析以下文本的情感倾向（正面/负面）：
{text}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def parse_label(output: str) -> int:
    """
    从模型输出中解析标签
    返回: 1 (正面) 或 0 (负面)
    """
    output = output.strip().lower()
    
    # 检查是否包含正面关键词
    positive_keywords = ["正面", "positive", "好", "满意", "推荐", "不错", "喜欢"]
    negative_keywords = ["负面", "negative", "差", "不满意", "不推荐", "不好", "讨厌"]
    
    # 优先检查明确的正面/负面
    if any(kw in output for kw in ["正面", "positive"]):
        return 1
    if any(kw in output for kw in ["负面", "negative"]):
        return 0
    
    # 检查其他关键词
    positive_count = sum(1 for kw in positive_keywords if kw in output)
    negative_count = sum(1 for kw in negative_keywords if kw in output)
    
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return 0
    else:
        # 默认返回正面（如果无法判断）
        return 1 if len(output) > 0 else 0

def test_with_llama_cli(
    model_path: str,
    test_data: List[Dict],
    llama_cli_path: str = None,
    max_samples: int = None,
    temperature: float = 0.1,
    max_tokens: int = 10,
    threads: int = -1,
    gpu_layers: int = 0,
    verbose: bool = False
) -> Dict:
    """
    使用llama-cli命令行工具批量测试模型
    
    Args:
        model_path: GGUF模型路径
        test_data: 测试数据列表
        llama_cli_path: llama-cli可执行文件路径（None则自动查找）
        max_samples: 最大测试样本数（None表示全部）
        temperature: 温度参数
        max_tokens: 最大生成token数
        threads: CPU线程数（-1表示自动）
        gpu_layers: GPU层数（0表示仅CPU）
        verbose: 是否显示详细信息
    
    Returns:
        评估结果字典
    """
    # 查找llama-cli路径
    if llama_cli_path is None:
        # 尝试多个可能的路径
        possible_paths = [
            Path(__file__).parent.parent / "llama.cpp" / "llama-cli",
            Path(__file__).parent.parent / "llama.cpp" / "build" / "bin" / "llama-cli",
            Path(__file__).parent.parent / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                llama_cli_path = str(path.absolute())
                break
        
        if llama_cli_path is None:
            raise FileNotFoundError(
                "找不到llama-cli可执行文件。请指定 --llama_cli_path 参数，"
                "或确保llama-cli在以下位置之一：\n" + 
                "\n".join(str(p) for p in possible_paths)
            )
    
    if not os.path.exists(llama_cli_path):
        raise FileNotFoundError(f"llama-cli路径不存在: {llama_cli_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    print(f"使用llama-cli: {llama_cli_path}")
    print(f"测试模型: {model_path}")
    print(f"测试样本数: {len(test_data) if max_samples is None else min(max_samples, len(test_data))}")
    print("-" * 60)
    
    # 限制测试样本数
    if max_samples is not None:
        test_data = test_data[:max_samples]
    
    # 统计信息
    correct = 0
    total = len(test_data)
    true_positives = 0  # 预测正面，实际正面
    false_positives = 0  # 预测正面，实际负面
    true_negatives = 0   # 预测负面，实际负面
    false_negatives = 0  # 预测负面，实际正面
    
    # 记录每个样本的详细信息
    results = []
    total_time = 0
    
    # 批量测试
    for i, item in enumerate(test_data):
        text = item["text"]
        true_label = item["label"]
        
        # 创建提示
        prompt = create_prompt(text)
        
        # 构建命令
        cmd = [
            llama_cli_path,
            "-m", model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--threads", str(threads),
        ]
        
        if gpu_layers > 0:
            cmd.extend(["--gpu-layers", str(gpu_layers)])
        
        # 运行推理
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30秒超时
                encoding="utf-8"
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            
            if result.returncode != 0:
                print(f"警告: 样本 {i+1} 推理失败: {result.stderr}")
                output = ""
            else:
                # 提取输出（llama-cli的输出格式）
                output = result.stdout.strip()
                # 尝试提取生成的内容（去除提示部分）
                if "<|im_start|>assistant" in output:
                    output = output.split("<|im_start|>assistant")[-1].strip()
                elif "assistant" in output.lower():
                    # 尝试其他格式
                    lines = output.split("\n")
                    for j, line in enumerate(lines):
                        if "assistant" in line.lower() and j + 1 < len(lines):
                            output = "\n".join(lines[j+1:]).strip()
                            break
        except subprocess.TimeoutExpired:
            print(f"警告: 样本 {i+1} 超时")
            output = ""
            inference_time = 30.0
        except Exception as e:
            print(f"警告: 样本 {i+1} 出错: {e}")
            output = ""
            inference_time = 0
        
        # 解析预测标签
        pred_label = parse_label(output)
        
        # 更新统计
        if pred_label == true_label:
            correct += 1
            if pred_label == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if pred_label == 1:
                false_positives += 1
            else:
                false_negatives += 1
        
        # 记录结果
        results.append({
            "index": i + 1,
            "text": text[:50] + "..." if len(text) > 50 else text,
            "true_label": true_label,
            "pred_label": pred_label,
            "output": output[:100] if output else "",
            "correct": pred_label == true_label,
            "time": inference_time
        })
        
        # 显示进度
        if (i + 1) % 10 == 0 or verbose:
            current_acc = correct / (i + 1)
            print(f"进度: {i+1}/{total} | 准确率: {current_acc:.4f} | "
                  f"平均时间: {total_time/(i+1):.3f}s/样本")
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    # 精确率、召回率、F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 负面类的指标
    precision_neg = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
    recall_neg = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
    
    avg_time = total_time / total if total > 0 else 0
    
    # 构建结果字典
    evaluation_result = {
        "model_path": model_path,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "f1_positive": f1,
        "precision_negative": precision_neg,
        "recall_negative": recall_neg,
        "f1_negative": f1_neg,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        },
        "avg_inference_time": avg_time,
        "total_time": total_time,
        "samples": results
    }
    
    return evaluation_result

def print_evaluation_report(result: Dict, output_file: str = None):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    print(f"模型路径: {result['model_path']}")
    print(f"测试样本数: {result['total_samples']}")
    print(f"正确数: {result['correct']}")
    print(f"\n整体指标:")
    print(f"  准确率 (Accuracy): {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"\n正面类别指标:")
    print(f"  精确率 (Precision): {result['precision_positive']:.4f}")
    print(f"  召回率 (Recall): {result['recall_positive']:.4f}")
    print(f"  F1分数: {result['f1_positive']:.4f}")
    print(f"\n负面类别指标:")
    print(f"  精确率 (Precision): {result['precision_negative']:.4f}")
    print(f"  召回率 (Recall): {result['recall_negative']:.4f}")
    print(f"  F1分数: {result['f1_negative']:.4f}")
    print(f"\n混淆矩阵:")
    cm = result['confusion_matrix']
    print(f"  真正例 (TP): {cm['true_positives']}")
    print(f"  假正例 (FP): {cm['false_positives']}")
    print(f"  真负例 (TN): {cm['true_negatives']}")
    print(f"  假负例 (FN): {cm['false_negatives']}")
    print(f"\n性能指标:")
    print(f"  平均推理时间: {result['avg_inference_time']:.3f}秒/样本")
    print(f"  总耗时: {result['total_time']:.2f}秒")
    print("=" * 60)
    
    # 保存到文件
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")

def compare_models(model_paths: List[str], test_data: List[Dict], **kwargs):
    """对比多个模型的性能"""
    print(f"\n对比 {len(model_paths)} 个模型...")
    print("=" * 60)
    
    all_results = []
    for model_path in model_paths:
        print(f"\n测试模型: {model_path}")
        result = test_with_llama_cli(model_path, test_data, **kwargs)
        all_results.append(result)
        print_evaluation_report(result)
    
    # 对比总结
    print("\n" + "=" * 60)
    print("模型对比总结")
    print("=" * 60)
    print(f"{'模型':<50} {'准确率':<10} {'F1(正面)':<10} {'F1(负面)':<10} {'平均时间(s)':<12}")
    print("-" * 60)
    for result in all_results:
        model_name = os.path.basename(result['model_path'])
        print(f"{model_name:<50} {result['accuracy']:<10.4f} "
              f"{result['f1_positive']:<10.4f} {result['f1_negative']:<10.4f} "
              f"{result['avg_inference_time']:<12.3f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="批量评估量化后的GGUF模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个模型
  python test_quantized_model.py --model_path ../outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf

  # 测试多个模型并对比
  python test_quantized_model.py --model_paths ../outputs/gguf/*.gguf --max_samples 100

  # 使用GPU加速
  python test_quantized_model.py --model_path ../outputs/gguf/model.gguf --gpu_layers 20
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="单个GGUF模型路径"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        help="多个GGUF模型路径（用于对比）"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed/test.json",
        help="测试数据路径（默认: data/processed/test.json）"
    )
    parser.add_argument(
        "--llama_cli_path",
        type=str,
        default=None,
        help="llama-cli可执行文件路径（默认: 自动查找）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大测试样本数（默认: 全部）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="温度参数（默认: 0.1）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="最大生成token数（默认: 10）"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="CPU线程数（-1表示自动，默认: -1）"
    )
    parser.add_argument(
        "--gpu_layers",
        type=int,
        default=0,
        help="GPU层数（0表示仅CPU，默认: 0）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存详细结果的JSON文件路径"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.model_path and not args.model_paths:
        parser.error("必须指定 --model_path 或 --model_paths")
    
    # 加载测试数据
    test_data_path = Path(__file__).parent.parent / args.test_data
    if not test_data_path.exists():
        parser.error(f"测试数据文件不存在: {test_data_path}")
    
    print(f"加载测试数据: {test_data_path}")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试数据总数: {len(test_data)}")
    
    # 准备模型路径列表
    if args.model_paths:
        model_paths = args.model_paths
    else:
        model_paths = [args.model_path]
    
    # 测试参数
    test_kwargs = {
        "llama_cli_path": args.llama_cli_path,
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "gpu_layers": args.gpu_layers,
        "verbose": args.verbose
    }
    
    # 执行测试
    if len(model_paths) == 1:
        # 单个模型测试
        result = test_with_llama_cli(model_paths[0], test_data, **test_kwargs)
        output_file = args.output or f"evaluation_{Path(model_paths[0]).stem}.json"
        print_evaluation_report(result, output_file)
    else:
        # 多个模型对比
        results = compare_models(model_paths, test_data, **test_kwargs)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n对比结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
