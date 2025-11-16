"""
将ModelScope下载的模型链接到HuggingFace期望的位置
"""

import os
import shutil
import glob

def link_modelscope_model():
    """链接ModelScope模型到HuggingFace位置"""
    print("=" * 60)
    print("链接ModelScope模型到HuggingFace位置")
    print("=" * 60)
    
    # ModelScope的模型路径
    modelscope_paths = [
        "./models/cache/qwen/Qwen2___5-0___5B-Instruct",
        "./models/cache/qwen/Qwen2.5-0.5B-Instruct",
    ]
    
    # HuggingFace期望的路径
    hf_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_cache_path = "./models/cache/models--Qwen--Qwen2.5-0.5B-Instruct"
    hf_snapshots_path = os.path.join(hf_cache_path, "snapshots")
    hf_snapshot_hash = "7ae557604adf67be50417f59c2c2f167def9a775"
    hf_model_path = os.path.join(hf_snapshots_path, hf_snapshot_hash)
    
    # 查找ModelScope模型
    modelscope_model_dir = None
    for path in modelscope_paths:
        if os.path.exists(path):
            model_file = os.path.join(path, "model.safetensors")
            if os.path.exists(model_file):
                modelscope_model_dir = path
                print(f"✅ 找到ModelScope模型: {path}")
                break
    
    if not modelscope_model_dir:
        print("❌ 未找到ModelScope模型")
        print("请先运行: python download_from_modelscope.py")
        return False
    
    # 检查模型文件
    model_files = glob.glob(os.path.join(modelscope_model_dir, "*"))
    print(f"\nModelScope模型文件:")
    for f in model_files:
        if os.path.isfile(f):
            size = os.path.getsize(f) / (1024**2)  # MB
            print(f"  {os.path.basename(f)}: {size:.2f} MB")
    
    # 创建HuggingFace目录结构
    os.makedirs(hf_model_path, exist_ok=True)
    
    # 复制或链接文件
    print(f"\n正在链接模型文件到: {hf_model_path}")
    
    copied_files = []
    for file_name in os.listdir(modelscope_model_dir):
        src_file = os.path.join(modelscope_model_dir, file_name)
        dst_file = os.path.join(hf_model_path, file_name)
        
        if os.path.isfile(src_file):
            if not os.path.exists(dst_file):
                try:
                    # 尝试创建硬链接（更快，节省空间）
                    os.link(src_file, dst_file)
                    print(f"  ✅ 硬链接: {file_name}")
                except OSError:
                    # 如果硬链接失败，复制文件
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ 复制: {file_name}")
                copied_files.append(file_name)
            else:
                print(f"  ⏭️  已存在: {file_name}")
    
    print(f"\n✅ 模型链接完成！共处理 {len(copied_files)} 个文件")
    print(f"模型路径: {hf_model_path}")
    
    # 验证关键文件
    model_file = os.path.join(hf_model_path, "model.safetensors")
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024**2)
        print(f"\n✅ 验证: model.safetensors ({size:.2f} MB) 已就绪")
        return True
    else:
        print("\n❌ 警告: model.safetensors 文件不存在")
        return False

if __name__ == "__main__":
    success = link_modelscope_model()
    if success:
        print("\n" + "=" * 60)
        print("现在可以运行训练脚本了！")
        print("python stage1_finetune/train_lora.py --config config/training_config.yaml")
        print("=" * 60)
    else:
        exit(1)


