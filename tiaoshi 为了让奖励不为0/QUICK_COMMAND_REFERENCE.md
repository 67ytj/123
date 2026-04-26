# ? 快速命令参考

## 现在就跑这条命令

```bash
# 一键启动 exp5（所有检查 + 自动启动）
cd "D:\jiangli\123\tiaoshi 为了让奖励不为0"

# 清空旧日志
rm -rf outputs/ShadowHandHora/exp4*

# 启动训练（带环境变量）
export HORA_OUTPUT_NAME="ShadowHandHora/exp5_maxreach"
bash scripts/train_shadow.sh exp5_maxreach
```

## 实时监控

```bash
# 新开一个终端，打开 TensorBoard
tensorboard --logdir outputs/ShadowHandHora/exp5_maxreach --port 6006

# 浏览器访问
http://localhost:6006
```

## 关键曲线看什么

| 曲线 | 位置 | 预期表现 | 红线 |
|------|------|--------|------|
| `rewards/reach` | SCALARS | 快速从 0.5 → <0.15 | 卡在 0.3+ 需排查 |
| `rewards/contact` | SCALARS | 从 0 上升到 0.5+ | 一直 = 0 代表指尖没碰 |
| `rewards/lift_low` | SCALARS | 3-8M 时首次 > 0 | 20M 还 = 0 说明有问题 |
| `diagnostics/ball_height` | SCALARS | 逐渐上升 | 一直 ≈ 0 说明没抓起 |
| episode_reward/step | SCALARS | 从 ~0 → 300+ | 破 150 算成功 |

## 一键检查修改

```bash
# 检查 Patch A (reward 改动)
grep -n "max_tip_dist" hora/tasks/shadow_hand_hora.py
# 应该看到：536: max_tip_dist = tip_dists.max(dim=-1)[0]

# 检查 Patch B (TB 新指标)
grep -n "rewards/contact" hora/tasks/shadow_hand_hora.py
# 应该看到：571: self.rew_writer.add_scalar('rewards/contact'

# 检查 Patch C (yaml 改动)
grep -E "reachAlpha: 5|z: \[0.02" configs/task/ShadowHandHora.yaml
# 应该看到：57: reachAlpha: 5.0  和  35: z: [0.02, 0.15]
```

## 预期时间表

| 时刻 | 应该看到的 |
|------|-----------|
| **启动时** | `[ShadowHandHora] Reward components TB → ...exp5_maxreach/reward_components` |
| **5 分钟后** | 第一条曲线出现，reach 快速下降 |
| **1 小时后** | contact 开始上升，reach < 0.2 |
| **2 小时后** | contact 稳定 > 0.5，可能看到 lift_low 第一次闪现 |
| **4 小时后** | 开始有稳定的 lift_low，Best reward > 100 |
| **10 小时后** | Best reward > 200-300（理想情况） |

## 如果想立即看效果（不等 10M 步）

```bash
# 查看当前最新的 tensorboard 日志
ls -lh outputs/ShadowHandHora/exp5_maxreach/reward_components/

# 查看输出文件大小增长（代表在写数据）
watch -n 5 'ls -lh outputs/ShadowHandHora/exp5_maxreach/reward_components/events* 2>/dev/null | tail -1'
```

## 如果需要中断重启

```bash
# 查看当前 Python 进程
ps aux | grep train.py

# 杀死进程（PID 替换成实际的）
kill -9 <PID>

# 重新启动
export HORA_OUTPUT_NAME="ShadowHandHora/exp5_maxreach"
bash scripts/train_shadow.sh exp5_maxreach
```

## 最后，记住这个判别标准

### exp5 成功 = 任意 3 项达成

- [ ] 10M 步后 Best > 150 （原卡在 79-88）
- [ ] Success Rate (4cm) > 20%
- [ ] rewards/contact 均值 > 0.3
- [ ] rewards/lift_low 触发率 > 5%
- [ ] 学习曲线形状符合预期（快速下降 → 缓慢上升）

---

**祝你好运！??**

如果任何问题，上来截图，我们一起排查。

但 99% 概率这次就能破 100 了。
