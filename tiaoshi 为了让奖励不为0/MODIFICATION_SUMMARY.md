# TensorBoard 奖励分量记录 - 修改完成总结

## ? 已完成修改

已成功对 `hora/tasks/shadow_hand_hora.py` 进行以下三处修改：

### 1?? 导入部分 (文件顶部)
```python
import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, tensor_clamp, torch_rand_float

from .base.vec_task import VecTask
```

**变更内容**: 
- 添加 `import os` 用于路径操作
- 添加 `from torch.utils.tensorboard import SummaryWriter` 用于TensorBoard记录

---

### 2?? `__init__` 方法末尾初始化 (约第 156-170 行)
```python
# cache of initial hand root state (filled in _create_envs)
if not hasattr(self, 'hand_init_state'):
    self.hand_init_state = None

# ===== Reward 分量 TensorBoard logger =====
# 放到和 PPO 同目录的 reward_components 子目录下
exp_name = self.config.get('exp_name', 'default')
output_name = self.config.get('output_name', f'ShadowHandHora/{exp_name}')
rew_log_dir = os.path.join('outputs', output_name, 'reward_components')
os.makedirs(rew_log_dir, exist_ok=True)
self.rew_writer = SummaryWriter(log_dir=rew_log_dir)
self.rew_log_counter = 0
print(f'[ShadowHandHora] Reward components TB → {rew_log_dir}')
```

**变更内容**:
- 初始化 TensorBoard SummaryWriter
- 设置日志目录为 `outputs/{output_name}/reward_components`
- 初始化日志计数器 `self.rew_log_counter = 0`

---

### 3?? `post_physics_step` 方法中奖励计算后 (约第 550-585 行)
```python
reward = r_reach + r_lift_low + r_lift_mid + r_lift_high + r_penalty
self.rew_buf[:] = reward

# ===== 每 50 个 env-step 写一次 TB，避免太密 =====
self.rew_log_counter += 1
if self.rew_log_counter % 50 == 0:
    step = self.rew_log_counter * self.num_envs  # 换算成 agent steps

    # Reward 分量（全 batch 平均）
    self.rew_writer.add_scalar('rewards/reach',     r_reach.mean().item(),     step)
    self.rew_writer.add_scalar('rewards/lift_low',  r_lift_low.mean().item(),  step)
    self.rew_writer.add_scalar('rewards/lift_mid',  r_lift_mid.mean().item(),  step)
    self.rew_writer.add_scalar('rewards/lift_high', r_lift_high.mean().item(), step)
    self.rew_writer.add_scalar('rewards/penalty',   r_penalty.mean().item(),   step)
    self.rew_writer.add_scalar('rewards/total',     reward.mean().item(),      step)

    # 诊断指标（关键！）
    tip_dist = torch.norm(
        tip_pos - obj_pos_exp, dim=-1
    ).mean(dim=-1)
    ball_h = self.object_pos[:, 2] - self.object_rest_height

    self.rew_writer.add_scalar('diagnostics/tip_contact_force_mean', 
                                self.contact_forces[:, self.fingertip_body_indices, :].norm(dim=-1).mean().item(), step)
    self.rew_writer.add_scalar('diagnostics/ball_height',  
                                ball_h.mean().item(), step)
    self.rew_writer.add_scalar('diagnostics/mean_tip_dist', 
                                tip_dist.mean().item(), step)

    # 成功率（球被抬超过 4cm 的比例）
    success_rate = (ball_h > self.lift_height_low).float().mean().item()
    self.rew_writer.add_scalar('diagnostics/success_rate_4cm', success_rate, step)
```

**变更内容**:
- 每 50 个环境步长记录一次TensorBoard指标
- 记录 6 个奖励分量: reach, lift_low, lift_mid, lift_high, penalty, total
- 记录 4 个诊断指标: 触觉力平均值, 球高度, 指尖平均距离, 成功率

---

## ?? TensorBoard 输出结构

修改后，TensorBoard 将记录以下指标：

```
rewards/
  ├── reach          ← 伸手接近球的分数
  ├── lift_low       ← 抬 4cm 的 bonus
  ├── lift_mid       ← 抬 8cm 的 bonus
  ├── lift_high      ← 抬 15cm 的 bonus
  ├── penalty        ← action 惩罚
  └── total          ← 总和

diagnostics/
  ├── tip_contact_force_mean   ← 触觉闭环是否激活
  ├── ball_height              ← 球被抬了多高
  ├── mean_tip_dist            ← 指尖平均离球多远
  └── success_rate_4cm         ← 成功率（球被抬超过4cm的比例）
```

---

## ?? 验证步骤

### 1. 检查语法
```bash
cd "D:\jiangli\123\tiaoshi 为了让奖励不为0"
python test_syntax.py
```

预期输出: `? [OK] File syntax is valid!`

### 2. 启动训练
```bash
bash scripts/train_shadow.sh exp4_lowhand
```

启动时应该看到:
```
[ShadowHandHora] Reward components TB → outputs/ShadowHandHora/exp4_lowhand/reward_components
```

### 3. 开启 TensorBoard
```bash
tensorboard --logdir outputs/ShadowHandHora --port 6006 --reload_multifile true
```

在浏览器打开 `http://localhost:6006`，应该能看到:
- SCALARS 标签页中有新增的 9 条曲线
- 约每 50 个环境步长更新一次

---

## ?? 注意事项

1. **配置名称**: 代码假设配置中有 `exp_name` 和 `output_name` 字段。如果不存在，将使用默认值 `'default'`。

2. **磁盘空间**: 50M 步长约产生 50MB 的日志数据，无需担心磁盘溢出。

3. **变量正确性**: 代码使用以下变量，确保它们在类中已定义:
   - `self.contact_forces` - 接触力张量
   - `self.fingertip_body_indices` - 指尖刚体索引
   - `self.object_rest_height` - 球的静止高度
   - `self.lift_height_low` - 低位lift高度阈值
   - `tip_pos`, `obj_pos_exp` - 指尖和物体位置（在post_physics_step中定义）

---

## ?? 诊断用途

| 现象 | 对应TB曲线 | 意义 |
|------|-----------|------|
| 只会伸手不会抓 | reach ↑, lift_low = 0 | reach-dominated local optimum（当前病状） |
| 假装抓（拍飞球） | lift_high ↑ 但 contact_force = 0 | reward hack 回归 |
| 健康学习 | lift_low → mid → high 依次点亮 | 想要的收敛轨迹 |
| 太消极 | reach 也不涨、penalty 很大 | 需要降 action penalty |
| 触觉闭环坏 | contact_force_mean < 0.01 | 指尖索引错了 |

---

## ? 已完成的所有改动

? 文件: `hora/tasks/shadow_hand_hora.py`
- ? 步骤1: 添加导入
- ? 步骤2: 初始化TensorBoard writer
- ? 步骤3: 添加奖励记录逻辑

所有修改已自动应用，无需手动编辑。
