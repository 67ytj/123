#!/bin/bash
set -e

cd "D:\jiangli\123\tiaoshi 为了让奖励不为0" || exit 1

echo "===== DFC GEOM Diagnostic Run ====="
echo ""
echo "启动诊断脚本 (num_envs=4, headless=True)"
echo ""

# 启动训练并过滤 GEOM 和 DFC 日志
HORA_OUTPUT_NAME=geom_debug python train.py \
    task=ShadowHandHora \
    train=ShadowHandHora \
    headless=True \
    num_envs=4 \
    2>&1 | grep -E "GEOM|DFC|RESET" | head -80

echo ""
echo "===== 诊断完成 ====="
