"""
检查Ollama是否已下载Qwen2.5-0.5B模型
注意：Ollama的GGUF格式不能直接用于HuggingFace训练，需要原始格式
"""

import os
import subprocess
import json
from pathlib import Path

def find_ollama_models():
    """查找Ollama模型位置"""
    print("=" * 60)
    print("检查Ollama模型")
    print("=" * 60)
    
    # 可能的Ollama模型路径
    possible_paths = [
        os.path.expanduser("~/.ollama/models"),
        "/mnt/c/Users/*/.ollama/models",
        "/mnt/c/Users/*/AppData/Local/ollama/models",
    ]
    
    print("\n查找Ollama模型目录...")
    found_paths = []
    
    for path_pattern in possible_paths:
        if "*" in path_pattern:
            # 使用glob查找
            import glob
            paths = glob.glob(path_pattern)
            found_paths.extend(paths)
        elif os.path.exists(path_pattern):
            found_paths.append(path_pattern)
    
    if found_paths:
        print(f"找到Ollama模型目录: {found_paths}")
        for path in found_paths:
            print(f"\n检查: {path}")
            if os.path.exists(path):
                # 查找qwen相关模型
                for root, dirs, files in os.walk(path):
                    for name in dirs + files:
                        if "qwen" in name.lower() or "Qwen" in name:
                            full_path = os.path.join(root, name)
                            print(f"  找到: {full_path}")
    else:
        print("未找到Ollama模型目录")
    
    return found_paths

def check_ollama_cli():
    """检查ollama命令行工具"""
    print("\n" + "=" * 60)
    print("检查Ollama CLI")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama已安装")
            print("\n已下载的模型:")
            print(result.stdout)
            
            # 检查是否有qwen模型
            if "qwen" in result.stdout.lower():
                print("\n✅ 找到Qwen模型！")
                return True
            else:
                print("\n⚠️  未找到Qwen模型")
                return False
        else:
            print("❌ Ollama命令执行失败")
            return False
    except FileNotFoundError:
        print("❌ Ollama未安装或不在PATH中")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """主函数"""
    print("\n重要提示:")
    print("=" * 60)
    print("Ollama下载的模型是GGUF格式，不能直接用于HuggingFace训练！")
    print("HuggingFace训练需要原始格式（safetensors或bin）")
    print("=" * 60)
    
    # 检查ollama CLI
    has_ollama = check_ollama_cli()
    
    # 查找模型文件
    find_ollama_models()
    
    print("\n" + "=" * 60)
    print("建议方案")
    print("=" * 60)
    print("\n方案1: 使用HuggingFace镜像站下载（推荐）")
    print("  python download_model.py")
    print("\n方案2: 使用ModelScope下载（阿里云，国内速度快）")
    print("  pip install modelscope")
    print("  python -c \"from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models/cache')\"")
    print("\n方案3: 使用huggingface-cli（支持断点续传）")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print("  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/cache/Qwen--Qwen2.5-0.5B-Instruct")
    
    if has_ollama:
        print("\n注意: 虽然Ollama已安装，但Ollama的GGUF格式模型")
        print("      不能直接用于训练，只能用于推理。")
        print("      训练仍需要从HuggingFace下载原始格式模型。")

if __name__ == "__main__":
    main()


