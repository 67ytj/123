#!/bin/bash
cd ~/hora_ws

echo "================================"
echo "Clearing Python cache..."
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Cache cleared."
echo "================================"
echo ""

echo "Starting training: R1 Pure RL (Sphere Grasp)"
echo "Expected: ep_reward_mean should be 0.15-0.40 range"
echo "Starting at $(date)"
echo "================================"
echo ""

bash scripts/scripts/train_shadow.sh r1_pure_rl 2>&1 | tee /tmp/r1_run.log

echo ""
echo "================================"
echo "Training started. Watch for:"
echo "  ? No NameError/KeyError/AssertionError"
echo "  ? [ShadowHandHora][debug] step=1 ... appears"
echo "  ? PPO ep_reward_mean in 0.15-0.40 range after first update"
echo "  ? Run for 3-5 minutes without crashes"
echo ""
echo "TensorBoard (after 3 min):"
echo "  tensorboard --logdir outputs/ShadowHandHora/r1_pure_rl/reward_components --port 6006"
echo "================================"
