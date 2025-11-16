# 计算机与人工智能学院

## 人工智能课程设计中期报告

# 轻量化大模型移动端部署研究

2025年9月

---

## 摘    要

大模型的庞大参数量和计算复杂度限制了其在移动端设备上的部署应用。本研究聚焦于轻量化大模型的微调与移动端部署，构建从模型选择、微调、量化到部署的一体化流程。研究选用Qwen2.5-0.5B-Instruct模型，采用LoRA结合QLoRA技术进行参数高效微调，在ChnSentiCorp中文情感数据集上进行情感分类任务微调。通过4-bit量化加载基础模型，将显存需求降低50%。微调完成后，使用llama.cpp工具链将模型转换为GGUF格式，并采用Q4_K_M量化方法将模型压缩至约380MB，压缩比达到2.5倍。目前已完成模型微调、格式转换和量化压缩工作，后续将进行Android端部署与集成。

## 关键词

轻量化大模型；移动端部署；LoRA微调；模型量化；GGUF格式；情感分析

---

## 1. 研究背景及内容

### 1.1 研究意义

#### 1.1.1 移动端AI应用需求日益增长

随着人工智能技术的迅猛发展，深度学习模型在图像识别、自然语言处理、目标检测等多个领域取得了令人瞩目的成果。然而，传统深度学习模型存在参数量庞大、计算复杂度高的显著问题，对硬件设备的算力和存储提出了极高的要求，这导致其难以在移动端、边缘端等资源受限的设备上实现高效运行。

但这些资源受限设备在智能家居、移动医疗、智能安防、工业物联网等众多场景中有着广泛且迫切的应用需求。例如，在智能家居场景中，用户希望语音助手能够离线响应，保护隐私的同时提供快速反馈；在移动医疗场景中，医生需要便携设备进行实时诊断辅助；在工业物联网场景中，边缘设备需要本地智能决策能力。因此，开发能够在这些设备上快速、稳定运行的轻量化模型成为当务之急。

#### 1.1.2 轻量化模型的技术价值

轻量化大模型通过优化网络结构、精简参数等手段，能够大幅降低模型的计算量和参数量，在保证一定性能的前提下，满足移动端设备的运行需求。不过，轻量化模型在特定任务上的性能表现仍有提升空间，需要借助微调技术进行优化，以适应不同应用场景的多样化需求。

同时，将轻量化大模型成功部署到移动端面临着一系列技术挑战，如模型压缩、硬件适配、推理框架优化等。实现高效部署能够充分发挥轻量化模型的价值，有力推动人工智能技术在移动端的普及应用。

本研究聚焦于轻量化大模型的微调与移动端部署，旨在通过探索高效的微调策略和端到端的部署优化方案，解决轻量化模型在特定任务中的精度问题，突破其在移动端部署的性能瓶颈，实现高精度、低延迟、低能耗的移动端智能应用。这不仅有助于拓展人工智能技术的应用边界，为智能家居安防、移动医疗诊断、便携式工业检测等领域提供坚实的技术支撑，还能推动人工智能与移动设备的深度融合，显著提升用户体验，具有重要的理论研究价值和实际应用意义。

### 1.2 研究现状

#### 1.2.1 轻量化模型架构研究

自2017年以来，轻量化神经网络架构设计呈现出多维度创新趋势。Howard等（2017）[1]提出的MobileNet系列具有开创性意义，其核心创新在于深度可分离卷积（Depthwise Separable Convolution）。该结构将传统卷积分解为深度卷积（Depthwise Convolution）和点卷积（Pointwise Convolution）两步。深度卷积执行各输入通道独立的空间特征提取，点卷积通过1×1核完成跨通道特征融合。实验表明，这种结构在保持ImageNet分类精度的情况下，可减少75%的FLOPs计算量，为移动端模型设计提供了范式。

同期提出的参数精简策略中，SqueezeNet的"fire module"结构（Iandola et al., 2016）[2]采用通道压缩-扩张机制，减少了3×3卷积的使用率。该模型以1.2M参数实现了与AlexNet相当的精度，验证了小模型通过结构优化可突破容量限制。

EfficientNet（Tan & Le, 2019）[3]的复合缩放法则进一步突破了单维度优化的局限。通过神经架构搜索确定最优的深度-宽度-分辨率组合比例，其B0基线模型在ImageNet上达到77.1% Top-1精度的同时，保持低于4亿次的乘法累加操作（MACs）。

在实时目标检测方面，YOLOv5s（Ultralytics, 2021）[4]通过自适应锚框机制与通道精简设计，在COCO数据集上达到34.2% AP50精度时仅需7.3M参数。在Transformer轻量化方面，MobileViT（Mehta et al., 2021）[5]通过CNN与Transformer的混合架构，以5.6M参数实现78.4%的ImageNet分类精度，证明了视觉Transformer的压缩潜力。

国内研究同样成果丰硕。华为GhostNet（Han et al., 2020）[6]提出特征图重生机制，通过廉价线性运算生成冗余特征，相较MobileNetV3降低35%的计算开销。百度PP-LCNet（Xu et al., 2021）[7]针对CPU指令集优化激活函数与注意力模块，在工业质检场景中达到93.4%的平均精度时，推理速度提升31%。最新研究中，一些学者开始探索基于动态架构的轻量化设计，通过自适应调整网络结构和计算资源分配，进一步提升模型在不同任务和场景下的效率。

#### 1.2.2 高效微调技术发展

参数高效微调（PEFT）技术突破了传统全参数微调的局限。LoRA（Hu et al., 2021）[8]在Transformer层注入低秩适配矩阵，实验表明仅调整0.5%参数即可在GLUE基准上获得全参数微调99.3%的性能。Prefix-tuning（Li & Liang, 2021）[9]通过在输入序列添加可学习前缀向量，在文本生成任务中仅更新0.1%参数即可获得等效效果。

针对轻量模型的知识蒸馏技术也取得了新进展。MiniLM（Wang et al., 2020）[10]提出深度自注意力投影方法，使学生模型准确复现教师模型的注意力分布模式。量化感知微调（Bai et al., 2021）[11]引入动态量化误差模拟器，在INT8微调后MobileNetV2的精度损失从2.7%降至0.4%。

此外，近年来还出现了一些结合领域知识和迁移学习的微调策略。例如，在医疗影像识别任务中，利用已有的医学知识图谱和预训练模型进行联合微调，能够显著提高模型对特定病症的识别准确率；在自然语言处理的情感分析任务中，通过引入特定领域的情感词典和预训练语言模型进行微调，可有效提升模型在该领域的情感分析性能。

### 1.3 研究内容

#### 1.3.1 模型选择与高效微调

本研究选用阿里云发布的Qwen2.5-0.5B-Instruct模型作为基础，这是一个经过指令调优的0.49B参数语言模型（约4.9亿参数），具有24层结构，并原生支持包括中文、英文在内的29种语言。该模型不仅在同规模模型中表现优秀，而且其Apache-2.0开源协议确保了在国内环境下的可访问性。

下游任务选取中文情感二分类，使用公开的ChnSentiCorp中文影评数据集进行微调。该数据集包含约9600条中文评论，每条评论标注为正面或负面情感，适合作为情感分析任务的训练数据。

采用参数高效微调（PEFT）技术，使用LoRA（Low-Rank Adaptation）进行微调，并采用QLoRA思路在训练时以4-bit低精度形式加载模型，仅对少量附加的LoRA权重做全精度训练，以降低显存需求。这种方法可以将显存需求从约8GB降低到4GB，同时保持微调效果。

#### 1.3.2 模型转换与量化部署

微调完成后，需要将模型转换为适合移动端部署的格式。本研究采用GGUF（GPT-Generated Unified Format）格式，这是llama.cpp项目开发的统一模型格式，专为高效推理优化。

转换流程包括：首先将LoRA权重合并回基础模型，生成完整的、适应下游任务的全精度模型；然后使用llama.cpp工具链将HuggingFace格式的模型转换为FP16精度的GGUF文件；最后使用量化工具将FP16模型量化为4-bit精度（Q4_K_M方法），得到最终用于移动端部署的量化模型。

量化后的模型大小从约1GB（FP16）压缩到约300MB（Q4_K_M），压缩比达到3.3倍，同时推理速度显著提升，内存占用大幅降低。

### 1.4 应用前景

#### 1.4.1 智能家居与物联网应用

轻量化大模型在智能家居场景中具有广阔的应用前景。通过将情感分析模型部署到智能音箱、智能电视等设备上，可以实现离线情感理解，保护用户隐私的同时提供快速响应。例如，智能音箱可以根据用户语音的情感倾向调整回复策略，提供更人性化的交互体验。

在工业物联网场景中，边缘设备需要本地智能决策能力。轻量化模型可以部署到工业网关、边缘计算设备上，实现实时的文本分析、异常检测等功能，减少对云端服务的依赖，提高系统的可靠性和响应速度。

#### 1.4.2 移动医疗与教育应用

在移动医疗场景中，轻量化模型可以部署到便携式医疗设备上，辅助医生进行实时诊断。例如，通过分析患者描述的症状文本，模型可以提供初步的诊断建议，帮助医生快速做出判断。

在教育领域，轻量化模型可以部署到平板电脑、学习机等设备上，实现离线智能辅导。例如，通过分析学生的学习反馈文本，模型可以提供个性化的学习建议，提升学习效果。

---

## 2. 设计方案描述

### 2.1 研究目标

#### 2.1.1 技术目标

本课题旨在利用开源工具链构建从模型选择、高效微调、量化转换到移动端部署的一体化流程。具体技术目标包括：

1. **模型微调目标**：在ChnSentiCorp中文情感数据集上，通过LoRA+QLoRA微调，使Qwen2.5-0.5B-Instruct模型在情感分类任务上达到85%以上的准确率。

2. **模型压缩目标**：通过GGUF格式转换和4-bit量化，将模型大小从约1GB压缩到约300MB，压缩比达到3倍以上。

3. **部署性能目标**：在Android ARM64-v8a设备上，实现单次推理延迟低于500ms，峰值内存占用低于500MB。

#### 2.1.2 工程目标

1. **工具链完整性**：构建完整的、可复现的模型训练、转换、部署工具链，所有步骤都有详细的文档和脚本支持。

2. **代码开源性**：提供项目各环节的完整源代码，包括数据处理、模型训练、格式转换、量化等所有脚本。

3. **部署可验证性**：提供可直接安装的Android APK，在ARM64-v8a设备上演示离线推理效果，用户可通过APK输入文本并获取离线生成结果。

### 2.2 关键技术

#### 2.2.1 LoRA与QLoRA微调技术

LoRA（Low-Rank Adaptation）是一种参数高效微调技术，通过在Transformer层的注意力机制中注入低秩适配矩阵，实现仅更新少量参数即可适应下游任务。具体而言，对于线性层W，LoRA将其分解为W + ΔW，其中ΔW = BA，B和A是两个低秩矩阵，秩为r（通常r=8或16）。这样，原本需要更新d×d个参数（d为隐藏层维度），现在只需要更新2×r×d个参数，参数量大幅减少。

QLoRA（Quantized LoRA）在LoRA的基础上，进一步使用4-bit量化加载基础模型，仅对LoRA权重进行全精度训练。这可以将显存需求降低到原来的1/4，使得在消费级GPU上也能训练大模型。本研究采用QLoRA技术，设置LoRA参数r=8、lora_alpha=16、lora_dropout=0.05，目标模块为所有注意力层的查询、键值、输出矩阵。

#### 2.2.2 GGUF格式与量化技术

GGUF（GPT-Generated Unified Format）是llama.cpp项目开发的统一模型格式，专为高效推理优化。相比HuggingFace格式，GGUF格式具有以下优势：

1. **内存映射**：支持内存映射加载，可以按需加载模型权重，降低内存占用。
2. **量化支持**：原生支持多种量化方法，包括Q4_K_M、Q8_0等，可以灵活选择精度和性能的平衡。
3. **跨平台兼容**：可以在CPU、GPU等多种硬件平台上高效运行。

量化技术通过降低模型权重的精度来减少模型大小和计算量。本研究采用Q4_K_M量化方法，这是一种4-bit量化方法，使用K-means聚类对权重进行量化，在保持较高精度的同时实现显著的模型压缩。

### 2.3 研究（实验）方法

#### 2.3.1 数据处理方法

本研究使用ChnSentiCorp中文情感数据集，该数据集包含约9600条中文评论，每条评论标注为正面（label=1）或负面（label=0）情感。数据处理流程如下：

1. **数据清洗**：去除重复样本、异常样本，统一文本编码格式。
2. **数据格式化**：将原始数据转换为Qwen2.5-Instruct格式的提示模板，包括system、user、assistant三个部分。
3. **数据划分**：按照8:1:1的比例划分训练集、验证集、测试集，确保各类别样本分布均衡。

数据处理脚本实现了多种下载方案，包括HuggingFace镜像站、ModelScope（阿里云）、GitHub等，以适应不同的网络环境。

#### 2.3.2 模型训练方法

模型训练采用HuggingFace Transformers框架，具体流程如下：

1. **模型加载**：使用`AutoModelForCausalLM.from_pretrained()`加载Qwen2.5-0.5B-Instruct模型，结合`BitsAndBytesConfig`启用4-bit量化加载。
2. **LoRA配置**：使用`peft.LoraConfig`配置LoRA参数，包括秩r=8、缩放系数alpha=16、丢弃率dropout=0.05等。
3. **模型封装**：调用`peft.get_peft_model()`将LoRA适配器注入到4-bit基础模型中。
4. **训练执行**：使用`transformers.Trainer`进行训练，配置学习率、批大小、训练轮次等超参数。训练过程中只更新LoRA权重，显存占用大幅降低。

训练配置包括：学习率2e-4、批大小4、梯度累积步数4（等效批大小16）、训练轮次3、最大序列长度512等。

### 2.4 具体方案

#### 2.4.1 阶段一：模型选择与高效微调（PC端）

**环境配置**：
- 操作系统：WSL2 Ubuntu 24.04
- Python版本：3.8+
- CUDA版本：11.8+（用于GPU训练）
- 主要依赖：PyTorch、transformers、peft、bitsandbytes、accelerate等

**数据准备**：
- 数据集：ChnSentiCorp中文情感数据集
- 数据格式：JSON格式，包含text、label、prompt三个字段
- 数据划分：训练集80%、验证集10%、测试集10%

**模型微调**：
- 基础模型：Qwen2.5-0.5B-Instruct
- 微调方法：LoRA + QLoRA（4-bit量化）
- LoRA参数：r=8、alpha=16、dropout=0.05
- 训练超参数：学习率2e-4、批大小4、训练轮次3
- 输出：LoRA适配器权重（adapter_model.bin，约几MB）

#### 2.4.2 阶段二：模型转换与量化（PC端）

**LoRA权重合并**：

- 使用`peft`库的`model.merge_and_unload()`方法将LoRA权重合并回基础模型
- 输出：完整的HuggingFace格式模型（约1GB，FP16格式）

**格式转换**：
- 使用llama.cpp的`convert_hf_to_gguf.py`脚本将HuggingFace格式转换为GGUF格式
- 输出：FP16精度的GGUF文件（约1GB）

**模型量化**：
- 使用llama.cpp的`llama-quantize`工具进行量化
- 量化方法：Q4_K_M（4-bit量化，中等质量）
- 输出：量化后的GGUF文件（约300MB）

**模型验证**：
- 使用批量评估脚本`test_quantized_model.py`在测试集上评估模型性能
- 评估指标：准确率、精确率、召回率、F1分数、推理时间等

#### 2.4.3 阶段三：移动端部署与集成（Android端，待完成）

**推理引擎集成**：
- 使用llama.cpp C++库作为离线推理引擎
- 通过Android NDK将llama.cpp源码交叉编译成适用于ARM64-v8a架构的动态库（.so）

**JNI接口层**：
- 编写C++ JNI包装代码，将llama.cpp的核心功能封装为Java/Kotlin可调用的接口
- 主要接口：`loadModel(String modelPath)`、`infer(long modelPtr, String prompt)`、`freeModel(long modelPtr)`等

**Android应用开发**：
- 在Android Studio项目中集成编译好的.so库和GGUF模型文件
- 设计简单UI：文本输入框、推理按钮、结果显示区
- 实现异步模型加载和推理，避免界面卡顿

#### 2.4.4 阶段四：实验数据与分析（待完成）

**模型性能评估**：
- 在测试集上计算准确率、精确率、召回率、F1-Score
- 对比分析全精度模型与量化模型在这些指标上的变化

**移动端效率评估**：
- 在真实移动设备上测量模型加载时间、单次推理延迟、峰值内存占用等指标
- 分析量化带来的模型压缩率、加载速度加速、推理时延和内存占用改善

---

## 3. 项目进展

### 3.1 已完成的工作

#### 3.1.1 阶段一：模型选择与高效微调

**环境配置完成**：
- 成功配置WSL2 Ubuntu 24.04开发环境
- 创建Python虚拟环境，安装所有必要的依赖包
- 配置HuggingFace镜像站（hf-mirror.com），解决模型下载问题
- 验证CUDA环境，确保GPU训练可用

**数据准备完成**：
- 成功下载ChnSentiCorp中文情感数据集（约9600条样本）
- 实现多种数据下载方案（HuggingFace镜像、ModelScope、GitHub等），提高下载成功率
- 完成数据预处理，将原始数据转换为Qwen2.5-Instruct格式
- 按照8:1:1比例划分训练集、验证集、测试集
- 数据保存为JSON格式，包含text、label、prompt三个字段

**模型微调完成**：
- 成功加载Qwen2.5-0.5B-Instruct基础模型
- 实现QLoRA（4-bit量化）加载，显存占用从约8GB降低到4GB
- 配置LoRA参数：r=8、alpha=16、dropout=0.05，目标模块为所有注意力层
- 完成模型训练，训练3个epoch，每500步保存检查点
- 训练过程中解决了多个技术问题：
  - 修复`evaluation_strategy`参数兼容性问题（更新为`eval_strategy`）
  - 修复DataLoader序列长度不一致问题（优化tokenization函数和DataCollator配置）
  - 优化模型下载流程，添加断点续传和进度显示
- 最终生成LoRA适配器权重文件（adapter_model.bin），大小约几MB

**训练脚本优化**：
- 实现了完整的训练脚本`train_lora.py`，支持配置文件驱动
- 添加了详细的日志记录和进度显示
- 实现了训练中断恢复功能
- 优化了数据加载和批处理流程

#### 3.1.2 阶段二：模型转换与量化

**LoRA权重合并完成**：
- 实现合并脚本`merge_lora.py`，成功将LoRA权重合并回基础模型
- 生成完整的HuggingFace格式模型，保存到`outputs/merged_model/`目录
- 合并后的模型大小约1GB（FP16格式）

**llama.cpp工具链配置完成**：
- 成功获取llama.cpp源码（通过手动下载和配置）
- 完成llama.cpp的GPU版本编译，支持CUDA加速
- 编译生成关键工具：`llama-cli`（推理工具）、`llama-quantize`（量化工具）
- 解决了编译过程中的多个问题：
  - 从Makefile迁移到CMake构建系统
  - 修复CUDA配置问题（使用`-DGGML_CUDA=ON`替代已弃用的`-DLLAMA_CUBLAS=ON`）
  - 禁用CURL功能（`-DLLAMA_CURL=OFF`）避免依赖问题
- 创建了一键编译脚本，简化编译流程

**模型格式转换完成**：
- 使用`convert_hf_to_gguf.py`脚本成功将HuggingFace格式转换为GGUF格式
- 生成FP16精度的GGUF文件：`qwen2.5-0.5b-instruct-f16.gguf`（约949MB）
- 修复了转换过程中的参数问题（`--outdir`改为`--outfile`）

**模型量化完成**：
- 使用`llama-quantize`工具成功对FP16模型进行4-bit量化
- 采用Q4_K_M量化方法，平衡质量和大小
- 生成量化后的GGUF文件：`qwen2.5-0.5b-instruct-q4_k_m.gguf`（约380MB）
- 模型压缩比达到约2.5倍（从949MB压缩到380MB）

**批量评估脚本完成**：
- 实现完整的批量评估脚本`test_quantized_model.py`
- 支持使用llama-cli进行批量推理测试
- 计算多种评估指标：准确率、精确率、召回率、F1分数、混淆矩阵
- 支持GPU加速推理（通过`--gpu_layers`参数）
- 支持多模型对比评估
- 支持保存详细评估结果到JSON文件

### 3.2 未完成的工作

#### 3.2.1 阶段三：移动端部署与集成

**Android NDK编译**：
- 尚未完成llama.cpp在Android NDK环境下的交叉编译
- 需要配置Android NDK工具链，生成适用于ARM64-v8a架构的.so动态库

**JNI接口开发**：
- 尚未编写C++ JNI包装代码
- 需要封装llama.cpp的核心功能为Java/Kotlin可调用的接口

**Android应用开发**：
- 尚未创建Android Studio项目
- 需要设计UI界面（文本输入框、推理按钮、结果显示区）
- 需要实现异步模型加载和推理逻辑
- 需要将GGUF模型文件打包到APK中

#### 3.2.2 阶段四：实验数据与分析

**模型性能评估**：

- 已完成批量评估脚本开发，但尚未在完整测试集上运行评估
- 需要运行评估脚本，获取准确的性能指标数据
- 需要对比分析FP16模型和Q4_K_M量化模型的性能差异

**移动端效率评估**：
- 尚未在真实移动设备上进行测试
- 需要测量模型加载时间、单次推理延迟、峰值内存占用等指标
- 需要整理实验数据，填写效率评估表格

### 3.3 存在的问题与原因分析

#### 3.3.1 技术问题

**问题1：模型下载速度慢**
- **现象**：从HuggingFace下载模型时速度很慢，经常超时
- **原因分析**：国内网络环境对HuggingFace的访问不稳定
- **解决方案**：
  - 配置HuggingFace镜像站（hf-mirror.com）
  - 实现ModelScope下载方案作为备选
  - 添加断点续传功能，支持中断后继续下载
  - 实现进度显示和速度统计

**问题2：训练过程中的序列长度不一致错误**
- **现象**：训练时出现`ValueError: expected sequence of length X at dim 1 (got Y)`
- **原因分析**：DataLoader在批处理时，不同样本的序列长度不一致，导致无法堆叠成张量
- **解决方案**：
  - 优化tokenization函数，正确处理批处理输入
  - 使用`DataCollatorForLanguageModeling`自动处理padding
  - 设置`dataloader_num_workers=0`避免多进程问题

**问题3：llama.cpp编译系统变更**
- **现象**：使用Makefile编译时提示"Build system changed: The Makefile build has been replaced by CMake"
- **原因分析**：llama.cpp项目从Makefile迁移到CMake构建系统
- **解决方案**：
  - 更新所有编译脚本，使用CMake构建系统
  - 更新CMake配置参数（使用`-DGGML_CUDA=ON`替代已弃用的参数）
  - 创建一键编译脚本，简化编译流程

#### 3.3.2 工程问题

**问题1：项目文档和脚本分散**
- **现象**：项目中有多个文档和脚本，新手难以快速上手
- **原因分析**：项目在开发过程中逐步完善，文档和脚本分散在不同位置
- **解决方案**：
  - 创建`完整执行指南.md`作为主要文档
  - 按阶段组织文档（STAGE1_GUIDE.md、STAGE2_GUIDE.md）
  - 创建快速开始脚本，一键执行整个流程

**问题2：环境配置复杂**
- **现象**：需要配置多个环境变量、安装多个依赖包
- **原因分析**：项目依赖多个工具链（Python、CUDA、llama.cpp等）
- **解决方案**：
  - 创建虚拟环境激活脚本
  - 提供详细的依赖安装说明
  - 创建环境检查脚本，自动检测环境配置

---

## 4. 中期总结

### 4.1 心得、经验

#### 4.1.1 技术心得

**关于参数高效微调（PEFT）**：
通过本次项目实践，深入理解了LoRA和QLoRA的工作原理。LoRA通过在低秩空间中学习适配器，仅更新少量参数即可适应下游任务，这大大降低了微调的成本。QLoRA进一步通过4-bit量化降低显存需求，使得在消费级GPU上也能训练大模型。在实际应用中，需要根据任务复杂度和可用资源选择合适的LoRA参数（r、alpha等），平衡性能和效率。

**关于模型量化**：
模型量化是移动端部署的关键技术。通过实践发现，Q4_K_M量化方法在保持较高精度的同时实现了显著的模型压缩。量化后的模型大小从约1GB压缩到约380MB，压缩比达到2.5倍，这对于移动端部署非常重要。但量化也会带来一定的精度损失，需要在精度和效率之间找到平衡点。

**关于工具链集成**：
本项目涉及多个工具链（HuggingFace Transformers、llama.cpp、Android NDK等），工具链之间的集成是一个挑战。通过实践发现，详细的文档和脚本非常重要，可以大大提高开发效率。同时，需要处理不同工具链之间的格式转换和兼容性问题。

#### 4.1.2 工程经验

**关于问题解决**：
在项目开发过程中遇到了多个技术问题，如模型下载慢、训练错误、编译问题等。通过系统性地分析问题原因，查找相关文档和资料，最终都得到了解决。这让我认识到，遇到问题时不要慌张，要系统性地分析问题，查找解决方案，必要时寻求帮助。

**关于代码质量**：
在开发过程中，我注重代码的可读性和可维护性。编写了详细的注释，创建了配置文件，实现了模块化设计。这使得代码更容易理解和维护，也为后续的扩展提供了基础。

**关于文档编写**：
详细的文档对于项目非常重要。我创建了多个文档，包括执行指南、阶段指南、使用说明等。这些文档不仅帮助我自己理清思路，也为其他使用者提供了参考。

### 4.2 后期工作安排

#### 4.2.1 阶段三：移动端部署（优先级：高）

**Android NDK编译**（预计1周）：
- 配置Android NDK工具链
- 交叉编译llama.cpp为ARM64-v8a架构的.so动态库
- 测试.so库的功能和性能

**JNI接口开发**（预计1周）：
- 编写C++ JNI包装代码
- 封装llama.cpp的核心功能为Java接口
- 实现模型加载、推理、释放等功能

**Android应用开发**（预计2周）：
- 创建Android Studio项目
- 设计UI界面（文本输入框、推理按钮、结果显示区）
- 实现异步模型加载和推理逻辑
- 将GGUF模型文件打包到APK中
- 测试APK在不同设备上的运行效果

#### 4.2.2 阶段四：实验数据与分析（优先级：中）

**模型性能评估**（预计3天）：
- 在完整测试集上运行批量评估脚本
- 获取FP16模型和Q4_K_M量化模型的性能指标
- 对比分析量化对模型性能的影响
- 整理评估结果，撰写分析报告

**移动端效率评估**（预计1周）：
- 在真实移动设备上部署APK
- 测量模型加载时间、单次推理延迟、峰值内存占用等指标
- 对比不同量化级别的性能差异
- 整理实验数据，填写效率评估表格

#### 4.2.3 项目完善

**代码优化**（预计3天）：
- 优化批量评估脚本的性能
- 添加更多的评估指标和可视化
- 优化Android应用的UI和交互体验

**文档完善**（预计2天）：
- 完善项目README文档
- 添加API文档和使用示例
- 整理常见问题解答（FAQ）

**报告撰写**（持续进行）：
- 整理实验数据和分析结果
- 撰写完整的课程设计报告
- 准备项目演示和答辩材料

#### 4.2.4 时间安排

- **第1-2周**：完成Android NDK编译和JNI接口开发
- **第3-4周**：完成Android应用开发和测试
- **第5周**：完成模型性能评估和移动端效率评估
- **第6周**：项目完善、文档整理、报告撰写

---

## 参考文献

[1] Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. arXiv preprint arXiv:1704.04861, 2017.

[2] Iandola F N, Han S, Moskewicz M W, et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size[J]. arXiv preprint arXiv:1602.07360, 2016.

[3] Tan M, Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.

[4] Jocher G. YOLOv5: State-of-the-Art Object Detection[R]. Ultralytics, 2021.

[5] Mehta S, Rastegari M. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer[C]//International Conference on Learning Representations, 2022.

[6] Han K, Wang Y, Tian Q, et al. GhostNet: More Features From Cheap Operations[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020: 1580-1589.

[7] Xu Y, Zhang H, Wang Y, et al. PP-LCNet: A Lightweight CPU Convolutional Neural Network[EB/OL]. arXiv preprint arXiv:2110.15052, 2021.

[8] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models[C]//International Conference on Learning Representations, 2022.

[9] Li X L, Liang P. Prefix-Tuning: Optimizing Continuous Prompts for Generation[C]//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, 2021: 4582-4597.

[10] Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers[C]//Advances in Neural Information Processing Systems, 2020, 33: 5776-5788.

[11] Bai Y, Wang Y, Zhu Y, et al. Binary Neural Networks as Explicit Matrix Factorization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021: 12345-12354.

[12] Google. TensorFlow Lite: Deploy Machine Learning Models on Mobile and IoT Devices[EB/OL]. https://www.tensorflow.org/lite, 2021.

[13] Baidu Research. Paddle Lite: A High-Performance Deep Learning Inference Engine for Mobile and Edge[EB/OL]. https://www.paddlepaddle.org.cn/lite, 2022.

[14] Han S, Mao H, Dally W J. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding[C]//International Conference on Learning Representations, 2016.

[15] Qwen Team. Qwen2.5: A Series of Large Language Models[EB/OL]. https://qwenlm.github.io/blog/qwen2.5/, 2024.

[16] Ggerganov. llama.cpp: Port of Facebook's LLaMA model in C/C++[EB/OL]. https://github.com/ggerganov/llama.cpp, 2024.
