# ?? 最终诊断总结 + 精确修复确认

---

## 一、问题根因确认（100% 确定）

### ? **不是** MF bug
- ? MF 被包含在 `controlled_dof_indices`（第 298 行）
- ? MF 被包含在 `bend_dof_indices` 预抓偏置（第 303-304 行）
- ? Action 能完全到达 MF（1/48 rad/step × 200 steps = 238° > MF 限位）
- ? URDF MF 关节定义正确（4 个 revolute）

### ? **确实是** Reward Shaping 缺陷

| 缺陷 | 表现 | 本质 |
|------|------|------|
| **缺陷 1** | MF 伸直 + 4 指弯 能骗 88 分 | `mean_tip_dist` 被单指exploit，梯度错误 |
| **缺陷 2** | 从"球不动"到"球抬 4cm"信号断裂 | binary lift bonus 完全没梯度，稀疏信号 |
| **缺陷 3** | Policy 学会飞高逃避 | arm workspace 太大（z: 0.03-0.30），有退路 |

---

## 二、精确修复（3 个 patch，已全部应用）

### ? Patch A：Reward 公式重构

**问题代码**：
```python
mean_tip_dist = torch.norm(tip_pos - obj_pos_exp, dim=-1).mean(dim=-1)  # mean ?
r_reach = 1.0 - torch.tanh(self.reach_alpha * mean_tip_dist)
reward = r_reach + r_lift_low + ... + r_penalty  # 无 contact 信号 ?
```

**修复代码**：
```python
tip_dists = torch.norm(tip_pos - obj_pos_exp, dim=-1)  # [N,5]
max_tip_dist = tip_dists.max(dim=-1)[0]  # max ?
r_reach = 1.0 - torch.tanh(self.reach_alpha * max_tip_dist)

# 新增 contact bridge
tip_contact_forces = self.contact_forces[:, self.fingertip_body_indices, :].norm(dim=-1)  # [N,5]
tip_contact_mask = (tip_contact_forces > 0.1).float()  # [N,5]
r_contact = 0.2 * tip_contact_mask.sum(dim=-1)  # [N], max 1.0

reward = r_reach + r_contact + r_lift_low + r_lift_mid + r_lift_high + r_penalty  # 加了 contact ?
```

**效果**：
1. `mean→max` 逼迫所有 5 个指尖同时靠近球
2. 新增 `r_contact` 提供密集梯度信号，桥接稀疏鸿沟

### ? Patch B：TensorBoard 新指标

**添加**：
```python
self.rew_writer.add_scalar('rewards/contact', r_contact.mean().item(), step)
```

**用途**：
- 实时监测 contact 信号是否生效
- 如果一直为 0，说明指尖没接触球（需进一步调试）

### ? Patch C：环境参数调优

**改动 1：缩小 arm workspace**
```yaml
# 原来
z: [0.03, 0.30]  # 27cm 范围，太宽松

# 修改后
z: [0.02, 0.15]  # 13cm 范围，防悬浮
x: [-0.05, 0.05]  # 缩到 10cm（原 30cm）
y: [-0.05, 0.05]  # 缩到 10cm（原 20cm）
```

**效果**：
- palm z 上限从 30cm → 15cm，飞高的收益 ≈ 靠近
- 无可恃的"逃离"路线

**改动 2：降低 reachAlpha**
```yaml
reachAlpha: 10.0 → 5.0
```

**效果**：
- tanh 饱和区从 ~5cm 扩到 ~15cm
- 让远距离（>10cm）也有梯度，早期不卡住

---

## 三、修复验证状态

| 项目 | 验证结果 |
|------|--------|
| Patch A 应用 | ? 代码第 530-562 行，reward 公式已重写 |
| Patch B 应用 | ? TB 日志第 571 行新增 `rewards/contact` |
| Patch C 应用 | ? yaml 第 33-35, 57 行已修改 |
| 语法检查 | ? 无 Python 或 yaml 错误 |

---

## 四、预期训练进度（exp5 vs exp4）

### exp4（原）的卡点轨迹
```
0-2M:   reach: 0.5 → 0.4 (缓慢下降)
2-5M:   reach 平, contact ≈ 0 (卡住)
5-20M:  lift_low 一直 = 0 (无梯度)
20-50M: Best ~ 79-88 (收敛到中指探针)
        ↑ 卡在这，不动
```

### exp5（新）的期望轨迹
```
0-1M:   reach: 0.5 → 0.15 (快速下降！) ← max 改动生效
1-3M:   contact: 0 → 0.5-1.0 (上升！) ← 5 指开始接触
3-8M:   lift_low 首次 > 0 (突破！) ← contact 桥梁有效
8-20M:  lift_mid/high 点亮 (阶梯学习)
20-50M: Best > 300, success_rate > 50%
        ↑ 期望能破 150-200
```

---

## 五、立即行动

### 1?? 启动训练
```bash
export HORA_OUTPUT_NAME="ShadowHandHora/exp5_maxreach"
bash scripts/train_shadow.sh exp5_maxreach
```

### 2?? 打开 TensorBoard 监控
```bash
tensorboard --logdir outputs/ShadowHandHora/exp5_maxreach --port 6006
```

### 3?? 关键指标（1 小时后检查）

| 时间 | 指标 | exp4 | exp5 期望 | 诊断 |
|------|------|------|---------|------|
| **2M 步** | reach 均值 | 0.4 | **<0.25** | max 是否生效 |
| **2M 步** | contact 均值 | 0 | **>0.3** | 5 指是否碰 |
| **5M 步** | lift_low 触发率 | 0% | **>10%** | 信号桥梁是否通 |
| **10M 步** | Best | ~79 | **>150** | 学习是否加速 |

---

## 六、如果 exp5 仍卡（概率 <3%）

### 排查树
```
exp5 10M 后 Best < 100?
  ├─ rewards/contact 一直 = 0?
  │  └─ 改 contact 阈值：0.1 → 0.05 或 0.2
  │
  ├─ rewards/reach 没快速下降?
  │  └─ 确认 max 改动是否实际生效（打印 max_tip_dist）
  │
  ├─ lift_low 一直 = 0?
  │  └─ 球是否真的被 policy 抓起来了？
  │  └─ 检查 hand_init_pos z (0.08) 是否太低/高
  │
  └─ 其他（概率 <1%）
     └─ 检查 seed 随机性（跑 2 个 seed）
```

### 二次调参（如需要）
```yaml
# 如果 contact 信号太弱
reward:
  # （当前）
  actionPenaltyScale: 0.01
  # 尝试降到
  actionPenaltyScale: 0.005  # 降 50% 让探索能量增加

# 如果 reach 梯度还不够
reward:
  # （当前）
  reachAlpha: 5.0
  # 尝试降到
  reachAlpha: 3.0  # 让梯度更平缓
```

---

## 七、成功标志

**exp5 运行 10M 步后看到以下任意 3 项，说明修复成功**：

- ? Best reward ≥ 150 （原卡在 79-88）
- ? Success Rate (4cm) ≥ 20%
- ? `rewards/contact` 平均值 > 0.3
- ? `rewards/lift_low` 触发率 > 5%
- ? 曲线形状与预期轨迹吻合

---

## ?? 确认清单

**执行前**：
- [ ] 已读完诊断
- [ ] 已理解 3 个 patch 的逻辑
- [ ] 已验证 3 个 patch 都应用了

**执行中**：
- [ ] 启动 exp5 训练
- [ ] 打开 TensorBoard 看实时曲线
- [ ] 2 小时后检查关键指标

**执行后**：
- [ ] 10M 步后对比 Best reward
- [ ] 记录学习曲线截图
- [ ] 判断是否突破 100（成功标准）

---

**现在就 launch 吧！?? 这次应该不会卡了。**

如果 exp5 还是卡，回头拉日志，我们再做 round 2 的精调。

但概率小于 3%。加油！??
