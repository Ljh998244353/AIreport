# llama.cpp 手动下载配置说明

## 已完成的工作

✅ **目录已重命名**: `llama.cpp-master` → `llama.cpp`

✅ **转换脚本已找到**: `convert_hf_to_gguf.py`

✅ **仓库结构完整**: 包含 Makefile、requirements.txt 等必要文件

## 当前状态

- ✅ 目录位置: `/home/ljh/AI/llama.cpp`
- ✅ 转换脚本: `convert_hf_to_gguf.py`（注意是下划线，不是横线）
- ⚠️  需要编译: 可执行文件（quantize、llama-cli）尚未编译

## 下一步操作

### 1. 编译llama.cpp（必需）

**注意：** 新版本llama.cpp使用CMake构建系统，不再使用Makefile。

**方法1: 使用编译脚本（推荐）**

```bash
cd /home/ljh/AI/llama.cpp
bash 编译llama.cpp.sh
```

**方法2: 手动编译**

```bash
cd /home/ljh/AI/llama.cpp

# 安装编译依赖（如果未安装）
sudo apt update
sudo apt install -y build-essential cmake

# 创建build目录
mkdir -p build
cd build

# 配置CMake（CPU版本）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 或配置CMake（GPU版本，需要CUDA）
# cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUBLAS=ON

# 编译
cmake --build . --config Release -j$(nproc)

# 创建符号链接（可选，方便使用）
cd ..
ln -sf build/bin/llama-cli llama-cli
ln -sf build/bin/llama-quantize quantize
# 注意: 实际文件名是 llama-quantize，创建为 quantize 方便使用
```

**编译完成后会生成：**
- `build/bin/llama-quantize` - 量化工具（注意名称包含llama-前缀）
- `build/bin/llama-cli` - 推理工具
- 其他工具在 `build/bin/` 目录下

### 2. 安装Python依赖（用于转换脚本）

```bash
cd /home/ljh/AI/llama.cpp

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可能需要额外安装
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证配置

```bash
# 运行配置检查脚本
cd /home/ljh/AI
bash setup_llama_cpp.sh
```

## 使用说明

### 转换模型为GGUF格式

```bash
cd /home/ljh/AI/llama.cpp

# 注意：脚本名是 convert_hf_to_gguf.py（下划线）
python convert_hf_to_gguf.py \
    ../outputs/merged_model \
    --outfile ../outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf \
    --outtype f16
```

### 量化模型

```bash
cd /home/ljh/AI/llama.cpp

# 如果创建了符号链接，直接使用：
./quantize \
    ../outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf \
    ../outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    Q4_K_M

# 或使用完整路径（注意：实际文件名是 llama-quantize）：
./build/bin/llama-quantize \
    ../outputs/gguf/qwen2.5-0.5b-instruct-f16.gguf \
    ../outputs/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    Q4_K_M
```

## 常见问题

### Q: 找不到 convert_hf_to_gguf.py？

**A**: 
- 确保在 `llama.cpp` 目录下（不是 `llama.cpp-master`）
- 运行 `ls convert_hf_to_gguf.py` 检查
- 如果不存在，可能是下载的版本不完整

### Q: 编译失败？

**A**: 
- 确保安装了编译工具：`sudo apt install build-essential cmake`
- 检查错误信息，可能需要安装额外的库
- 尝试清理后重新编译：
  ```bash
  rm -rf build
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . --config Release
  ```
- 如果使用GPU版本，确保CUDA已正确安装

### Q: 转换脚本报错？

**A**: 
- 确保已安装Python依赖：`pip install -r requirements.txt`
- 检查模型路径是否正确
- 确保模型已合并（完成阶段二的步骤1）

## 文件位置参考

```
/home/ljh/AI/
├── llama.cpp/                    # llama.cpp工具链（已配置）
│   ├── CMakeLists.txt            # CMake配置文件
│   ├── convert_hf_to_gguf.py    # 转换脚本（下划线）
│   ├── build/                    # 编译目录
│   │   └── bin/                 # 可执行文件目录
│   │       ├── quantize         # 量化工具（编译后）
│   │       └── llama-cli        # 推理工具（编译后）
│   ├── quantize                  # 符号链接（如果创建）
│   └── llama-cli                # 符号链接（如果创建）
├── outputs/
│   ├── merged_model/            # 合并后的模型（阶段二步骤1）
│   └── gguf/                    # GGUF格式模型（阶段二步骤3-4）
└── setup_llama_cpp.sh           # 配置检查脚本
```

## 继续执行

完成编译和依赖安装后，可以继续执行阶段二的步骤：

1. ✅ 合并LoRA权重（已完成或待完成）
2. ⏳ 编译llama.cpp（当前步骤）
3. ⏳ 转换为GGUF格式
4. ⏳ 量化模型

