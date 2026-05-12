#!/bin/bash
set -e

cd "D:\jiangli\123\tiaoshi 为了让奖励不为0" || exit 1

echo "===== DFC Ball Scale 与 Geometry 诊断 ====="
echo ""
echo "Step 1: 启动诊断 (num_envs=4, headless=True)"
echo ""

# 启动训练并过滤关键日志
HORA_OUTPUT_NAME=scale_geom_debug python train.py \
    task=ShadowHandHora \
    train=ShadowHandHora \
    headless=True \
    num_envs=4 \
    2>&1 | tee /tmp/scale_geom_diag.log

echo ""
echo "===== 诊断完成 ====="
echo ""
echo "关键日志输出:"
echo ""
echo "1. Ball Scale 列表:"
grep -E "DFC-SCALES-AVAIL.*Ball" /tmp/scale_geom_diag.log || echo "(未找到 Ball scale 列表)"
echo ""
echo "2. DFC Scale 选择:"
grep -E "DFC-SCALE" /tmp/scale_geom_diag.log | head -10 || echo "(未找到 DFC scale 信息)"
echo ""
echo "3. 几何诊断:"
grep -E "GEOM\]" /tmp/scale_geom_diag.log | head -20 || echo "(未找到 GEOM 诊断)"
echo ""
