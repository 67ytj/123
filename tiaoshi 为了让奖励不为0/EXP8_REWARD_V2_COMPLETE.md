# ?? exp8_reward_v2 - One-shot Patch 完成

## ? 三个 Patch 全部应用成功

### ? **Patch A: Reward 函数完全重设计**

**文件**: `hora/tasks/shadow_hand_hora.py` 第 530-611 行

**关键改动**:

| 项目 | 旧版 | 新版 exp8 | 机制 |
|------|------|---------|------|
| **Reach** | tanh（近处饱和） | 线性 -3.0×dist | 永无梯度饱和 |
| **Contact** | r_contact=0.2×tanh | ? 删除 | 根除悬停吸引子 |
| **Lift Vel** | 无 | r_lift_vel=10×clamp(vz) | **新增**：抓的瞬间有信号 |
| **Lift Height** | 4/8/15cm | **1/3/6cm** + binary | 门槛拉低 3-7 倍 |
| **Anti-Drift** | 无 | -5.0×clamp(dist-0.2) | **新增**：防手漂走 |
| **Penalty** | -0.01 | -0.002 | 减弱 5 倍 |

**核心逻辑链**:
```
悬停收益（旧）= -0.18 (reach) + 0.8 (contact) = +0.62/step
       ↓
悬停收益（新）= -0.18 (reach) + 0 (no contact) - 0.01 (drift) = -0.19/step
       ↓
抓取收益（新）= 0 (reach近) + 5 (lift_vel) + 10 (lift_s1) = +15/step
       ↓
EV 翻转！policy 被迫去抓球而非悬停
```

### ? **Patch B: TensorBoard 日志更新**

**新增 7 条曲线**（替代旧的 contact）:
- `rewards/lift_vel` ← 速度奖励（新）
- `rewards/lift_s1/s2/s3` ← 三级 lift（替代旧的 lift_low/mid/high）
- `rewards/anti_drift` ← 防漂移（新）

**诊断曲线更新**:
- 删除 `diagnostics/tip_contact_force_mean`（旧）
- 新增 `diagnostics/success_rate_1cm/3cm/6cm`（对应新阈值）
- 新增 `diagnostics/hand_ball_dist`（监测漂移）
- 新增 `diagnostics/ball_vz`（监测速度信号）

### ? **Patch C: 硬编码参数（不依赖 yaml）**

所有超参直接写入 Python，避免"改 yaml 忘改 Python"的老 bug：

```python
# 线性 reach
r_reach = -3.0 * max_tip_dist

# 三级 lift
r_lift_stage1 = 10.0 * (h_above > 0.01).float()    # 1cm
r_lift_stage2 = 30.0 * (h_above > 0.03).float()    # 3cm
r_lift_stage3 = 60.0 * (h_above > 0.06).float()    # 6cm

# 速度奖励
r_lift_vel = 10.0 * torch.clamp(ball_vz, 0.0, 0.5)

# 防漂移
r_anti_drift = -5.0 * torch.clamp(hand_ball_dist - 0.20, min=0.0)

# 弱 penalty
r_penalty = -0.002 * action_sqnorm
```

---

## ?? 立即启动

```bash
cd "D:\jiangli\123\tiaoshi 为了让奖励不为0"

# 清空旧实验（从 scratch）
rm -rf outputs/ShadowHandHora/exp8*

# 启动 exp8_reward_v2（新输出目录）
export HORA_OUTPUT_NAME="ShadowHandHora/exp8_reward_v2"
bash scripts/train_shadow.sh exp8_reward_v2
```

**关键**：
- ? 不要 `--resume` exp7（critic 毒化）
- ? from scratch（新 reward 下一起学）
- ? 新目录 `exp8_reward_v2`（不覆盖 exp7）

---

## ?? 硬标准验收（必须达成）

| 时间点 | 必看指标 | 预期 | 红线 |
|--------|---------|------|------|
| **0-1M** | `rewards/total` | **负值**(-50~0) | 如果正值=悬停仍赚钱 |
| **2M** | `rewards/reach` | -0.15 ~ -0.25 | 卡在 -0.3 = 探索不足 |
| **5M** | `rewards/lift_vel` | > 0.01 | 一直=0 = 球未被推 |
| **10M** | `success_rate_1cm` | **≥ 20%** | = 0 = 还在悬停 |
| **20M** | `Best > 500` | 破 500 | < 100 = 假设错 |
| **30M** | `success_rate_6cm` | ≥ 10% | < 5% = 需要调参 |
| **50M** | 最终目标 | **Best > 1000** | 成功指标 |

**快速退出条件**:
- 如果 **2M 时 total < -100** → reward 量纲错了，立刻停
- 如果 **10M 时 lift_s1 = 0** → 手不下去，加 pre-grasp 初始化

---

## ?? 实时监控命令

```bash
# 新开终端，打开 TB
tensorboard --logdir outputs/ShadowHandHora/exp8_reward_v2 --port 6006

# 关键看这些曲线
# 1. rewards/total        ← 应该从 -50 涨到 500+
# 2. rewards/lift_vel     ← 应该从 0 涨到 1+
# 3. success_rate_1cm/3cm/6cm  ← 应该逐阶段涨
# 4. hand_ball_dist       ← 应该维持在 0.2 附近（不漂走）
```

---

## ?? 如果 exp8 在某个阶段卡住

### 场景 1: 10M 时 lift_s1 = 0（手不下去）

**症状**: policy 还在悬停，没有下去真抓

**原因**: 光改 reward 不够，初始姿态问题

**修复**: 加 pre-grasp 初始化
```python
# 在 reset_idx 里加：
# 你队友的预抓姿态（从之前的 grasp 轨迹提取）
self.hand_dof_pos[env_ids, self.bend_dof_indices] = 0.3 + noise
```

### 场景 2: 5M 时 lift_vel = 0（球从未被推）

**症状**: action 出了问题，手没有力

**原因**: `arm_action_scale` 太小或 finger 控制不足

**修复**: 
```python
# 在 yaml 或 pre_physics_step 里检查
arm_xyz_action_scale: 0.01 → 0.05 或 0.1
```

### 场景 3: 20M 时 Best 还是 < 200

**症状**: 虽然 lift_vel 有了，但梯度还不足

**原因**: 需要 demo 引导（DAPG）

**修复**: 加 teammate 的成功 trajectory 作为 BC loss

---

## ?? 改动清单（一览）

| 文件 | 行号 | 内容 | 状态 |
|------|------|------|------|
| `hora/tasks/shadow_hand_hora.py` | 530-611 | reward 完全重写 | ? |
| `hora/tasks/shadow_hand_hora.py` | 613-620 | TB 日志更新 | ? |
| `hora/tasks/shadow_hand_hora.py` | 623-646 | 诊断指标更新 | ? |

**yaml 无需改动** —— 所有参数硬编码在 Python

---

## ?? 成功标志

**任意一个达成 = 改动生效**：

- ? Best ≥ 200（从 136 跳到 200+）
- ? `success_rate_1cm` ≥ 20%（10M 时）
- ? `rewards/lift_vel` 平均 > 0.5（有稳定速度信号）
- ? `hand_ball_dist` 维持 0.15-0.25（手没漂走）

---

## ?? 免责声明

这套方案的**三个关键假设**：

1. 悬停是 reward topology 问题（不是观察/动作空间问题）
2. `r_contact` 是主要原因（删了会有效）
3. 线性 reach + 速度 lift 能提供持续梯度

**如果前两个假设错了**，那改 reward 也无效，得上 DAPG 或 curriculum learning。

但基于 TB 数据（`r_contact=170/137`），我 95% 确信。

---

**现在就跑吧！20 小时内应该能看到显著变化。** ??

有问题实时截图 + 说 step 数，我们一起排查。
