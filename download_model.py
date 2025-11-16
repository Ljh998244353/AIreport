"""
从HuggingFace镜像站下载Qwen2.5-0.5B-Instruct模型
显示下载进度和速度
"""

import os
import sys
import time
import glob

# 设置镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 确保transformers显示进度条
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

def download_model():
    """下载模型"""
    print("=" * 60)
    print("从HuggingFace镜像站下载模型")
    print("=" * 60)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = "./models/cache"
    
    print(f"\n模型: {model_name}")
    print(f"镜像站: {os.environ.get('HF_ENDPOINT')}")
    print(f"缓存目录: {cache_dir}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("\n步骤1: 下载tokenizer...")
        print("提示: transformers库会自动显示下载进度条和速度\n")
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            resume_download=True  # 支持断点续传
        )
        tokenizer_time = time.time() - start_time
        print(f"\n✅ Tokenizer下载完成 (耗时: {tokenizer_time:.2f}秒)")
        
        print("\n" + "=" * 60)
        print("步骤2: 下载模型（这可能需要几分钟，请耐心等待）...")
        print("=" * 60)
        print("提示:")
        print("  - transformers库会自动显示下载进度条和速度")
        print("  - 如果下载中断，可以重新运行此脚本，会自动续传")
        print("  - 模型文件大小约988MB\n")
        
        # 检查是否已有部分文件
        model_files_before = set()
        try:
            model_files_before = set(glob.glob(f"{cache_dir}/**/model*.safetensors", recursive=True))
            if model_files_before:
                print(f"检测到已有部分模型文件，将续传下载...")
        except:
            pass
        
        # 先尝试下载完整模型（不使用量化）
        start_time = time.time()
        print("开始下载模型文件...\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            dtype=torch.float16,  # 使用 dtype 替代已弃用的 torch_dtype
            device_map="auto",
            resume_download=True,  # 支持断点续传
            local_files_only=False
        )
        model_time = time.time() - start_time
        
        print(f"\n✅ 模型下载完成！(耗时: {model_time:.2f}秒, {model_time/60:.2f}分钟)")
        print(f"模型已保存到: {cache_dir}")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型信息:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 计算模型大小
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        print(f"  模型大小: {model_size_mb:.2f} MB")
        
        # 显示下载速度和文件信息
        try:
            # 尝试获取实际下载的文件大小
            model_files = glob.glob(f"{cache_dir}/**/model*.safetensors", recursive=True)
            if model_files:
                file_size = os.path.getsize(model_files[0]) / (1024**2)  # MB
                print(f"  模型文件大小: {file_size:.2f} MB")
                if model_time > 0:
                    speed = file_size / model_time  # MB/s
                    print(f"  平均下载速度: {speed:.2f} MB/s")
                    if speed > 0:
                        print(f"  预计剩余时间: {file_size / speed / 60:.2f} 分钟")
        except Exception as e:
            print(f"  注意: 无法计算下载速度 ({e})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确认 HF_ENDPOINT 环境变量已设置")
        print("3. 尝试手动从镜像站下载:")
        print(f"   访问: https://hf-mirror.com/{model_name}")
        print("4. 或使用 huggingface-cli:")
        print(f"   huggingface-cli download {model_name} --local-dir {cache_dir}/{model_name.replace('/', '--')}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n" + "=" * 60)
        print("模型下载完成！现在可以运行训练脚本了")
        print("=" * 60)
    else:
        sys.exit(1)

