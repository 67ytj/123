# ?? 故障排查指南

## 问题 1: SyntaxError 或 ImportError

### 症状
```
SyntaxError: invalid syntax at line XXX
或
ImportError: No module named 'torch.utils.tensorboard'
```

### 解决方案

**检查 Python 版本**:
```bash
python --version
# 应该 >= 3.7
```

**检查 PyTorch 安装**:
```bash
python -c "from torch.utils.tensorboard import SummaryWriter; print('OK')"
```

如果失败，安装 PyTorch:
```bash
pip install torch tensorboard
```

---

## 问题 2: AttributeError - 变量不存在

### 症状
```
AttributeError: 'ShadowHandHora' object has no attribute 'contact_forces'
或
AttributeError: 'ShadowHandHora' object has no attribute 'fingertip_body_indices'
```

### 排查步骤

1. **确认基类初始化正确**
   ```bash
   grep -n "self.contact_forces\|self.fingertip_body_indices" hora/tasks/base/vec_task.py
   ```

2. **确认 _acquire_tensors 被调用**
   ```bash
   grep -n "_acquire_tensors" hora/tasks/shadow_hand_hora.py
   ```
   应该在 __init__ 中有调用

3. **确认刷新顺序**
   在 `post_physics_step` 中确认:
   ```python
   self._refresh_gym()  # 必须在前面
   # ... 然后才能用 self.contact_forces
   ```

### 解决方案

检查代码第 530 行是否有 `self._refresh_gym()`:
```python
def post_physics_step(self):
    self.progress_buf += 1
    self.reset_buf[:] = 0

    self._refresh_gym()  ← ? 必须有这一行
    self._debug_print_root_state_once()

    with torch.no_grad():
        # ... 现在可以用 self.contact_forces
```

---

## 问题 3: TensorBoard 没有看到新增曲线

### 症状
- TensorBoard 打开后显示 "No dashboards are active"
- 或者只能看到 episode_rewards 但看不到 rewards/reach 等

### 排查步骤

1. **确认日志目录创建成功**
   ```bash
   ls -la outputs/ShadowHandHora/exp4_lowhand/reward_components/
   # 应该看到 events.out.tfevents... 文件
   ```

2. **确认训练日志输出**
   看训练启动时是否输出:
   ```
   [ShadowHandHora] Reward components TB → outputs/ShadowHandHora/exp4_lowhand/reward_components
   ```

   如果没有，说明 __init__ 中的初始化代码没有执行

3. **检查日志计数**
   确认已跑超过 50 个环境步
   ```
   rew_log_counter >= 50  →  才会记录第一条数据
   ```

### 解决方案

**清理旧日志后重新开始**:
```bash
rm -rf outputs/ShadowHandHora/exp4_lowhand/reward_components/
# 然后重新启动训练
```

**检查 TensorBoard logdir**:
```bash
tensorboard --logdir outputs/ShadowHandHora --port 6006 --reload_multifile true
# 确保指向父目录，这样才能同时看PPO原日志和reward_components子目录
```

**刷新浏览器**:
- Ctrl+F5 (强制刷新，不用缓存)
- 稍等 30 秒让 TensorBoard 加载新数据

---

## 问题 4: 日志文件太大或崩溃

### 症状
- TensorBoard 卡顿或无响应
- 磁盘空间不足
- events.out.tfevents 文件超过 1GB

### 原因分析

每步（50个env-step）记录 10 个标量：
- 50 个环境 × 50 步 = 2500 步
- 50M 总步数 ÷ 2500 = 20,000 次记录
- 20,000 × 10 个标量 = 200,000 条数据
- 约 50MB 大小 ? 正常

### 解决方案

如果要减少日志大小，修改 `post_physics_step` 中的记录间隔:
```python
# 当前: 每 50 个env-step 记录一次
if self.rew_log_counter % 50 == 0:

# 改为: 每 100 个env-step 记录一次（减半）
if self.rew_log_counter % 100 == 0:

# 或每 200 个env-step 记录一次（四分之一）
if self.rew_log_counter % 200 == 0:
```

然后删除旧日志重新跑:
```bash
rm -rf outputs/ShadowHandHora/*/reward_components/
```

---

## 问题 5: 曲线数值都是 NaN 或 Inf

### 症状
```
TensorBoard 中曲线显示为空或全是平线
或者数值非常大/异常
```

### 排查步骤

1. **检查奖励值计算**
   ```bash
   # 启用debug日志
   configs/task/ShadowHandHora.yaml 中设置:
   debug:
     enable: true
   ```

2. **查看控制台输出**
   训练日志中应该看到:
   ```
   [ShadowHandHora][debug] ... tip_contact_force_mean=0.001234
   ```

3. **检查配置参数**
   ```bash
   grep -E "reachAlpha|liftBonus|actionPenalty" configs/task/ShadowHandHora.yaml
   ```

   合理范围:
   - reachAlpha: 5-20 (不要太大)
   - liftBonus: 1-100 (阶梯式)
   - actionPenaltyScale: 0.001-0.1

### 解决方案

重置为默认值:
```python
# 在 _setup_reward_config 中
self.reach_alpha = float(r_config.get('reachAlpha', 10.0))
self.lift_bonus_low = float(r_config.get('liftBonusLow', 2.0))
self.lift_bonus_mid = float(r_config.get('liftBonusMid', 10.0))
self.lift_bonus_high = float(r_config.get('liftBonusHigh', 20.0))
self.action_penalty_scale = float(r_config.get('actionPenaltyScale', 0.01))
```

---

## 问题 6: 指尖接触力始终为 0

### 症状
```
diagnostics/tip_contact_force_mean 一直是 0 或接近 0
```

### 原因分析

1. **指尖索引错误** - `self.fingertip_body_indices` 指向了错的刚体
2. **接触还没发生** - 球离得太远，还没接触
3. **物理参数问题** - friction 或 stiffness 设置不对

### 排查步骤

1. **打印刚体名称**
   ```bash
   grep -n "Body names:" logs/training_output.log
   ```
   确认有 "HandDigit5Tip", "MiddleDigitTip" 等 tip 刚体

2. **检查 Body 索引**
   ```bash
   python -c "
   from hora.tasks.shadow_hand_hora import ShadowHandHora
   # 手动创建一个实例看看 fingertip_body_indices
   "
   ```

3. **检查接触设置**
   ```bash
   grep -n "friction\|contactOffset" configs/task/ShadowHandHora.yaml
   ```

### 解决方案

如果 tip 刚体命名不标准，修改识别逻辑:
```python
# 在 _create_envs 中找到类似这样的代码:
for i, name in enumerate(body_names):
    if 'tip' in name.lower():
        # 正确索引
```

或者手动在 __init__ 中设置:
```python
self.fingertip_body_indices = torch.tensor([5, 9, 13, 17, 21], device=self.device)  # 例子
```

---

## 问题 7: 内存泄漏或显存溢出

### 症状
- 训练到 N 步后内存持续增长
- CUDA out of memory 错误
- TensorBoard 占用大量内存

### 原因分析

TensorBoard writer 可能没有 flush 导致缓存堆积

### 解决方案

修改 `post_physics_step` 添加 flush:
```python
if self.rew_log_counter % 50 == 0:
    # ... 所有 add_scalar 调用 ...
    self.rew_writer.flush()  # ← 添加这一行
```

或者在 episode 末尾添加 flush（在 reset_idx 中）:
```python
def reset_idx(self, env_ids):
    # ... 现有代码 ...
    if hasattr(self, 'rew_writer'):
        self.rew_writer.flush()
```

---

## 问题 8: 脚本不兼容（旧PyTorch版本）

### 症状
```
AttributeError: module 'torch' has no attribute 'float'
或其他版本相关错误
```

### 解决方案

升级 PyTorch 和 TensorBoard:
```bash
pip install --upgrade torch tensorboard
```

最低版本要求:
- PyTorch >= 1.6
- TensorBoard >= 2.4
- Python >= 3.7

---

## 快速诊断脚本

将以下代码保存为 `diagnose.py`:
```python
#!/usr/bin/env python3
import sys
import torch

print("?? 诊断信息")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# 检查 TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    print("? TensorBoard 可用")
except ImportError as e:
    print(f"? TensorBoard 不可用: {e}")

# 检查文件
import os
if os.path.exists('hora/tasks/shadow_hand_hora.py'):
    print("? shadow_hand_hora.py 存在")
    # 简单的语法检查
    import ast
    try:
        with open('hora/tasks/shadow_hand_hora.py') as f:
            ast.parse(f.read())
        print("? 文件语法正确")
    except SyntaxError as e:
        print(f"? 文件有语法错误: {e}")
else:
    print("? 找不到 shadow_hand_hora.py")

print("\n如果全是?，说明环境配置正确。")
```

运行:
```bash
python diagnose.py
```

---

## 联系支持

如果上述方案都无法解决，请收集以下信息:
1. `python -V` 输出
2. `pip list | grep -i "torch\|tensorboard"` 输出
3. 错误的完整 stack trace
4. `hora/tasks/shadow_hand_hora.py` 第 1-20 行和 550-585 行代码
5. 训练启动的完整命令

