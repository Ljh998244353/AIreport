#!/bin/bash
# 激活Python虚拟环境脚本
# 用于Ubuntu 24.04等使用PEP 668规范的系统

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虚拟环境不存在，正在创建..."
    python3 -m venv "$VENV_PATH"
    if [ $? -eq 0 ]; then
        echo "✅ 虚拟环境已创建"
    else
        echo "❌ 创建虚拟环境失败"
        exit 1
    fi
fi

echo "激活虚拟环境: $VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "✅ 虚拟环境已激活"
echo "Python路径: $(which python)"
echo "pip路径: $(which pip)"
echo ""
echo "提示: 使用 'deactivate' 退出虚拟环境"



