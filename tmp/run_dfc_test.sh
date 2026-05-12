#!/bin/bash
set -e

cd "D:\jiangli\123\tiaoshi 为了让奖励不为0" || exit 1

echo "===== DFC Reset Fix 启动测试 ====="
echo ""
echo "配置检查:"
echo "  ? arm_xyz_workspace.z: [0.05, 0.60]"
echo "  ? alignToObject: false"
echo "  ? reset_idx: 完整替换版本，含 [RESET-DFC-VERIFY] 打印"
echo ""
echo "启动训练..."
echo ""

# 清理旧输出（可选）
# rm -rf outputs/shadow_debug

# 激活 conda 环境
# source activate hora  # Linux/Mac
# conda activate hora   # Windows

# 启动训练
python train.py \
    task=ShadowHandHora \
    train=ShadowHandHora \
    headless=True \
    2>&1 | tee /tmp/hora_dfc_test.log

echo ""
echo "===== 训练完成 ====="
echo "日志位置: /tmp/hora_dfc_test.log"
echo ""
echo "关键日志行(应在启动后 5-10 秒内出现):"
echo "  [RESET-DFC] hand_root_pos = ..."
echo "  [RESET-DFC] g[pos][0]     = ..."
echo "  [RESET-DFC-VERIFY] |hand_dof_pos - g[q]|.mean() = ..."
echo ""
