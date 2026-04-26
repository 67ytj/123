#!/bin/bash
# 一键启动 exp5 训练的脚本

# 清空旧日志
echo "?? 清空旧 exp4 日志..."
rm -rf outputs/ShadowHandHora/exp4*

# 验证修改
echo "?? 验证所有 patch 已应用..."
if grep -q "max_tip_dist = tip_dists.max" hora/tasks/shadow_hand_hora.py; then
    echo "  ? Patch A (reward 改动) 已应用"
else
    echo "  ? Patch A 未找到，请检查！"
    exit 1
fi

if grep -q "rewards/contact" hora/tasks/shadow_hand_hora.py; then
    echo "  ? Patch B (TB 新指标) 已应用"
else
    echo "  ? Patch B 未找到，请检查！"
    exit 1
fi

if grep -q "reachAlpha: 5.0" configs/task/ShadowHandHora.yaml; then
    echo "  ? Patch C (yaml 改动) 已应用"
else
    echo "  ? Patch C 未找到，请检查！"
    exit 1
fi

echo ""
echo "?? 所有 patch 验证通过！"
echo ""
echo "?? 启动 exp5_maxreach 训练..."
echo ""

# 设置环境变量
export HORA_OUTPUT_NAME="ShadowHandHora/exp5_maxreach"

# 启动训练
bash scripts/train_shadow.sh exp5_maxreach

echo ""
echo "? 训练已启动！"
echo ""
echo "?? 打开 TensorBoard 监控："
echo "   tensorboard --logdir outputs/ShadowHandHora/exp5_maxreach --port 6006"
echo ""
echo "?? 关键曲线："
echo "   - rewards/reach      (应快速下降)"
echo "   - rewards/contact    (应上升到 0.5+)"
echo "   - rewards/lift_low   (应在 3-8M 时出现)"
echo ""
