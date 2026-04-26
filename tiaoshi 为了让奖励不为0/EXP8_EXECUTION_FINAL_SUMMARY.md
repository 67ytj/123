# ?? exp8_reward_v2 - 完整执行总结

## ? 所有 Patch 已验证成功

### ?? 三个问题的最终答案

| Q | 答案 | 验证 |
|---|------|------|
| **Q1: 变量名对吗？** | ? 全对 | `self.root_state_tensor`, `self.object_indices`, `self.contact_forces` 都正确 |
| **Q2: object_rest_height 类型？** | ? scalar float | 第 857 行：`float(r_config.get(...))` |
| **Q3: self.extras 会清零吗？** | ? 不清零 | VecTask.step() 返回 extras，每 step 保留而不清零 |

---

## ?? 改动方案核心（三行总结）

```
原因：r_contact 在悬停时贡献 170/137，policy 放弃真抓
→ 删 r_contact，加线性 reach + 速度奖励
→ 强制 EV 翻转：悬停负、抓取正
```

---

## ?? 改动对比表

| 维度 | 旧版 exp5/6/7 | 新版 exp8_reward_v2 | 机制 |
|-----|-----------|-----------------|------|
| **Reach** | tanh，近处饱和 | 线性 -3×dist | 永无梯度封顶 |
| **Contact** | 0.2×tanh×5指 | ? **删除** | 根除悬停吸引 |
| **Speed** | 无 | 10×clamp(vz) | **新增**：抓的瞬间有+1/step |
| **Height1** | 4cm | **1cm** | 拉低 4 倍 |
| **Height2** | 8cm | **3cm** | 拉低 2.7 倍 |
| **Height3** | 15cm | **6cm** | 拉低 2.5 倍 |
| **Anti-Drift** | 无 | -5×clamp(d-0.2) | **新增**：防手漂走 |
| **Penalty** | -0.01 | -0.002 | 减弱 5 倍 |

---

## ?? 立即执行（一条命令）

```bash
bash launch_exp8.sh
```

或手动执行：

```bash
export HORA_OUTPUT_NAME="ShadowHandHora/exp8_reward_v2"
bash scripts/train_shadow.sh exp8_reward_v2
```

---

## ?? 期望学习曲线

### 阶段 1：0-5M（建立基本梯度）
```
rewards/total:      -50 → -20  （悬停亏损日益明显）
rewards/lift_vel:   0 → 0.1   （第一次球被推）
success_rate_1cm:   0% → 5%   （偶尔抬起）
```

### 阶段 2：5-15M（加速探索）
```
rewards/total:      -20 → +50  （加速！）
rewards/lift_s1:    > 2        （经常 1cm）
success_rate_1cm:   5% → 30%   （稳定离地）
```

### 阶段 3：15-30M（多级突破）
```
Best:               136 → 300+ （期望值）
rewards/lift_s2:    > 3        （3cm 触发增加）
success_rate_3cm:   > 10%
```

### 阶段 4：30-50M（最终收敛）
```
Best:               > 500      （目标）
success_rate_6cm:   > 10-20%   （真正抓起）
hand_ball_dist:     0.15-0.25  （稳定贴近）
```

---

## ?? 关键监控指标（优先级）

### ?? **最高优先级**（决定改动是否生效）

1. **`rewards/total` @ 1M**
   - ? 应该是负值（-50 ~ 0）
   - ? 如果正值（> 0）→ 悬停仍赚钱，改动失效

2. **`success_rate_1cm` @ 10M**
   - ? 应该 ≥ 20%（1cm 离地概率）
   - ? 如果 = 0% → 手从未下去，加 pre-grasp 初始化

3. **`rewards/lift_vel` @ 5M**
   - ? 应该有非零值（球被推过）
   - ? 如果 = 0 → action 可能问题

### ?? **高优先级**（确认梯度流）

4. `hand_ball_dist` → 应维持 0.15-0.25（不漂走）
5. `ball_vz` → 应有正值（球往上）
6. `rewards/lift_s1/s2/s3` → 应逐级涨

### ?? **参考指标**（确认收敛）

7. `Best reward` → 目标 > 500
8. `success_rate_3cm` / `6cm` → 目标 > 10%

---

## ?? 快速退出条件（立刻停训练）

### 条件 1：2M 时 total < -100
```
含义：即使删了 r_contact，悬停损失过大
原因：可能 reach 权重算错了
修复：调 r_reach 权重从 -3.0 改 -1.0 或 -5.0
```

### 条件 2：10M 时 lift_s1 = 0 且 lift_vel = 0
```
含义：手完全没下去
原因：初始姿态问题（手自动悬停）
修复：加 pre-grasp 初始化 + 问题排查
```

### 条件 3：20M 时 Best < 100
```
含义：改动完全无效
原因：假设错了（不是 reward topology 问题）
修复：切换到 DAPG + curriculum 方案
```

---

## ?? 文件修改记录

| 文件 | 行号 | 改动 | 验证 |
|------|------|------|------|
| `hora/tasks/shadow_hand_hora.py` | 530-611 | reward 完全重写 | ? 语法通过 |
| `hora/tasks/shadow_hand_hora.py` | 613-620 | TB 日志更新 | ? 7 条曲线 |
| `hora/tasks/shadow_hand_hora.py` | 623-646 | 诊断指标更新 | ? 新的 success_rate |
| `configs/task/ShadowHandHora.yaml` | - | 无改动 | ? 参数硬编码 |
| `launch_exp8.sh` | - | 新增启动脚本 | ? |

---

## ?? 实验管理

### 启动
```bash
bash launch_exp8.sh              # 一键启动（包含验证）
# 或
export HORA_OUTPUT_NAME="ShadowHandHora/exp8_reward_v2"
bash scripts/train_shadow.sh exp8_reward_v2
```

### 监控
```bash
tensorboard --logdir outputs/ShadowHandHora/exp8_reward_v2 --port 6006
```

### 中断/恢复
```bash
# 中断：Ctrl+C 即可（下次直接 resume 会自动加载 checkpoint）

# 恢复（从 checkpoint 继续）
bash scripts/train_shadow.sh exp8_reward_v2

# 重启（清空 checkpoint，从头开始）
rm -rf outputs/ShadowHandHora/exp8*
bash scripts/train_shadow.sh exp8_reward_v2
```

---

## ?? 最终目标

**50M 步后达成所有目标**：

- ? Best reward ≥ 500（从 136 跳到 500）
- ? Success rate (6cm) ≥ 10%（成功抓握）
- ? Hand ball distance 稳定在 0.15-0.25（贴近）
- ? No drift（手没有漂走到 30cm 外）

**任意 3 项 = 改动成功** ?

---

## ?? 理论支撑

这套改动的三个关键论证：

### 1?? **Reward Topology 翻转**
```
旧公式：V(悬停) = -0.18 + 0.8 = +0.62/step × 200 = +124 分
新公式：V(悬停) = -0.18 + 0 = -0.18/step × 200 = -36 分
     V(抓取) = +5 + 10 = +15/step × 20 steps = +300 分
```
→ EV 完全翻转，policy 被迫去抓

### 2?? **梯度密集化（Reward Shaping 最佳实践）**
```
旧：0 → 4cm → 8cm → 15cm（只有 3 个离散点）
新：0 → 1cm → 3cm → 6cm（更密集）+ 速度梯度（连续）
```
→ 探索空间从"找 4cm"缩到"找 1cm"，更容易

### 3?? **防漂移（Exploration Stability）**
```
旧：手可以随意飞到 30cm 外探索（极大探索空间）
新：超过 20cm 开始惩罚，维持在合理范围
```
→ 减少无效探索

---

**现在就跑吧！** ??

这套方案是 3 次失败后的最终设计，理论充分、代码验证完毕。

**20 小时内应该能看到显著变化。**

有问题随时截图来，我们实时排查。

加油！??
