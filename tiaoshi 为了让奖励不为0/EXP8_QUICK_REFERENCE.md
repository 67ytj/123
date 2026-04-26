# ? exp8_reward_v2 快速参考卡

## 一行启动命令

```bash
bash launch_exp8.sh
```

## 三个关键改动（秒懂）

| 改动 | 作用 | 期望 |
|------|------|------|
| **删 r_contact** | 根除悬停吸引子 | 悬停不赚钱 |
| **加 r_lift_vel** | 抓的瞬间有+1/step | 加速探索下去 |
| **改 r_reach 线性** | 近处永有梯度 | 不卡住 |

## 4 个必看指标（前 20 小时）

```
1. rewards/total          @ 1M   应是负值（-50 ~ 0）
2. success_rate_1cm       @ 10M  应 ≥ 20%
3. rewards/lift_vel       @ 5M   应 > 0
4. hand_ball_dist         始终   应 0.15-0.25
```

## 快速退出（立刻停）

- 2M 时 total < -100 → reward 权重错
- 10M 时 lift_s1 = 0 → 姿态问题
- 20M 时 Best < 100 → 假设错了

## TB 监控

```bash
tensorboard --logdir outputs/ShadowHandHora/exp8_reward_v2 --port 6006
```

## 重启（清除数据）

```bash
rm -rf outputs/ShadowHandHora/exp8* && bash launch_exp8.sh
```

---

**20 小时内见分晓。** ??
