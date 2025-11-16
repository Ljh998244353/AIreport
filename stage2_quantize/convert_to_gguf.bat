@echo off
REM 将HuggingFace格式模型转换为GGUF格式
REM 注意：此脚本需要在llama.cpp目录下运行

REM 配置
set HF_MODEL_PATH=..\work\outputs\merged_model
set OUTPUT_DIR=..\work\outputs\gguf
set OUTPUT_TYPE=f16

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 运行转换脚本
echo 开始转换模型为GGUF格式...
python convert-hf-to-gguf.py "%HF_MODEL_PATH%" --outdir "%OUTPUT_DIR%" --outtype %OUTPUT_TYPE%

if %ERRORLEVEL% EQU 0 (
    echo 转换完成！输出目录: %OUTPUT_DIR%
) else (
    echo 转换失败！
    exit /b 1
)

