# ? 从 exp5 Best=136 冲刺到 300+ 的最后两步

## ?? 核心改动（2 行代码的威力）

### 改动 1：梯度永远有（不再封顶）
```python
# 从这样（力量强就没梯度）：
tip_contact_mask = (tip_contact_forces > 0.1).float()
r_contact = 0.2 * tip_contact_mask.sum(dim=-1)

# 改成这样（力量越强分越高）：
per_tip_contact = torch.tanh(tip_contact_forces / 3.0)
r_contact = 0.2 * per_tip_contact.sum(dim=-1)
```

**效果**：policy 永远有"抓得更紧"的奖励，不会卡在力量 0.5N

### 改动 2：第一级奖励更容易触发
```yaml
# 从这样（抬 4cm 才+2）：
liftHeightLow: 0.04

# 改成这样（抬 2cm 就+2）：
liftHeightLow: 0.02
```

**效果**：梯度密集，policy 能看到"稍微抬一下"的价值

---

## ?? 立即执行

```bash
# 假设已经停止了 exp5 训练
# 如果要从 checkpoint 继续，别清日志

# 直接启动（会自动加载最新 checkpoint）
export HORA_OUTPUT_NAME="ShadowHandHora/exp5_maxreach"
bash scripts/train_shadow.sh exp5_maxreach

# 打开 TB 看实时效果
tensorboard --logdir outputs/ShadowHandHora/exp5_maxreach --port 6006
```

---

## ?? 看这两条曲线判断有没有效

1. **`rewards/contact`** 
   - ? 如果变成**平滑上升**（之前是阶梯状）→ 改动 1 生效
   - ? 还是阶梯 → 可能失效

2. **`episode_reward/step`**
   - ? 从 136 **陡峭向上冲** → 两个改动都生效
   - ? 缓慢爬 → 需要进一步调参

---

## ?? 最终目标

| 时刻 | 应该看到的 |
|------|-----------|
| **1 小时后** | contact 曲线变平滑 |
| **4 小时后** | Best 从 136 → 180+ |
| **10 小时后** | Best 从 180 → 250+ |
| **20 小时后** | Best ≥ 300（成功！） |

---

**就这样，继续跑！这次应该稳了。** ??
