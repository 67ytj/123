#!/bin/bash
# exp8_reward_v2 快速启动脚本

set -e

echo "?? exp8_reward_v2 - 一次性重设计 reward"
echo "=========================================="
echo ""

# 验证改动
echo "? 验证 Patch A (reward 改动)..."
if grep -q "r_reach = -3.0 \* max_tip_dist" hora/tasks/shadow_hand_hora.py; then
    echo "  ? 线性 reach 已应用"
else
    echo "  ? 线性 reach 未找到！"
    exit 1
fi

if grep -q "r_lift_vel = 10.0 \* torch.clamp" hora/tasks/shadow_hand_hora.py; then
    echo "  ? 速度奖励已应用"
else
    echo "  ? 速度奖励未找到！"
    exit 1
fi

if grep -q "r_anti_drift = -5.0 \* torch.clamp" hora/tasks/shadow_hand_hora.py; then
    echo "  ? 防漂移已应用"
else
    echo "  ? 防漂移未找到！"
    exit 1
fi

echo ""
echo "? 验证 Patch B (TB 日志改动)..."
if grep -q "rewards/lift_s1" hora/tasks/shadow_hand_hora.py; then
    echo "  ? 新 TB 日志已应用"
else
    echo "  ? TB 日志未更新！"
    exit 1
fi

echo ""
echo "? 清空旧输出..."
rm -rf outputs/ShadowHandHora/exp8*

echo ""
echo "? 设置环境变量..."
export HORA_OUTPUT_NAME="ShadowHandHora/exp8_reward_v2"

echo ""
echo "?? 启动训练..."
echo "   输出目录: outputs/ShadowHandHora/exp8_reward_v2"
echo ""
echo "重要提醒："
echo "  1??  这是 from scratch（不 resume）"
echo "  2??  看 TB 时重点关注："
echo "     - rewards/total 应该从负值涨到 500+"
echo "     - success_rate_1cm 10M 时应该 >= 20%"
echo "     - hand_ball_dist 应该维持 0.15-0.25"
echo ""

bash scripts/train_shadow.sh exp8_reward_v2
