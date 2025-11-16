@echo off
REM 量化GGUF模型
REM 注意：此脚本需要在llama.cpp\build\bin\Release目录下运行

REM 配置
set INPUT_GGUF=..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-f16.gguf
set OUTPUT_GGUF=..\..\..\work\outputs\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf
set QUANT_TYPE=Q4_K_M

REM 检查输入文件是否存在
if not exist "%INPUT_GGUF%" (
    echo 错误: 输入文件不存在: %INPUT_GGUF%
    exit /b 1
)

REM 创建输出目录
for %%F in ("%OUTPUT_GGUF%") do set OUTPUT_DIR=%%~dpF
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 运行量化
echo 开始量化模型...
echo 输入: %INPUT_GGUF%
echo 输出: %OUTPUT_GGUF%
echo 量化类型: %QUANT_TYPE%

quantize.exe "%INPUT_GGUF%" "%OUTPUT_GGUF%" %QUANT_TYPE%

if %ERRORLEVEL% EQU 0 (
    echo 量化完成！
    echo 输出文件: %OUTPUT_GGUF%
    REM 显示文件大小
    if exist "%OUTPUT_GGUF%" (
        for %%A in ("%OUTPUT_GGUF%") do echo 文件大小: %%~zA 字节
    )
) else (
    echo 量化失败！
    exit /b 1
)

