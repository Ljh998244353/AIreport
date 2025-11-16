"""
使用ModelScope（阿里云）下载Qwen2.5-0.5B-Instruct模型
ModelScope在国内下载速度通常更快
"""

import os
import sys

def download_from_modelscope():
    """从ModelScope下载模型"""
    print("=" * 60)
    print("使用ModelScope下载模型（阿里云，国内速度快）")
    print("=" * 60)
    
    try:
        # 尝试导入ModelScope
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("\n正在安装ModelScope...")
            os.system("pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
            from modelscope import snapshot_download
        
        model_id = "qwen/Qwen2.5-0.5B-Instruct"
        cache_dir = "./models/cache"
        
        print(f"\n模型ID: {model_id}")
        print(f"缓存目录: {cache_dir}")
        print("\n开始下载（这可能需要几分钟）...")
        
        # 下载模型
        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision="master"
        )
        
        print(f"\n✅ 模型下载完成！")
        print(f"模型位置: {model_dir}")
        
        # 检查文件
        import os
        files = os.listdir(model_dir)
        print(f"\n模型文件:")
        for f in files:
            file_path = os.path.join(model_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"  {f}: {size:.2f} MB")
        
        print("\n注意: ModelScope的模型格式与HuggingFace兼容")
        print("      可以直接用于训练！")
        
        # ModelScope下载的模型可以直接使用，但需要确保路径正确
        # 检查模型文件
        import glob
        model_files = glob.glob(os.path.join(model_dir, "*.safetensors")) + \
                     glob.glob(os.path.join(model_dir, "*.bin"))
        
        if model_files:
            print(f"\n✅ 模型文件已下载:")
            for f in model_files:
                size = os.path.getsize(f) / (1024**2)  # MB
                print(f"  {os.path.basename(f)}: {size:.2f} MB")
        
        # 提示：ModelScope的模型可以直接使用，训练脚本会自动识别
        print(f"\n提示: 训练时可以直接使用路径: {model_dir}")
        print("      或者修改训练脚本中的模型路径")
        
        print("\n" + "=" * 60)
        print("模型已准备好，可以开始训练！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n备选方案:")
        print("1. 使用HuggingFace镜像站: python download_model.py")
        print("2. 检查网络连接")
        return False

if __name__ == "__main__":
    success = download_from_modelscope()
    if not success:
        sys.exit(1)

