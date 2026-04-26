# ?? Bug 修复总结 - HORA_OUTPUT_NAME 环境变量

## ?? 问题描述

代码原来从 `self.config` 中读取 `output_name`：
```python
output_name = self.config.get('output_name', f'ShadowHandHora/{exp_name}')
```

**问题**: `output_name` 是 PPO 训练器的配置项（`train.ppo.output_name`），**不在 task 的 config 里**。

**后果**: 代码永远读不到真实的 output_name，只能用默认值 `'ShadowHandHora/default'`，导致 TensorBoard 日志写到错误的路径。

**实际现象**:
```
训练命令: train.ppo.output_name=ShadowHandHora/exp4_lowhand
TB日志实际写到: outputs/ShadowHandHora/default/reward_components/  ? 错误！
应该写到:       outputs/ShadowHandHora/exp4_lowhand/reward_components/  ? 正确
```

---

## ? 修复内容

### 修改 1: `hora/tasks/shadow_hand_hora.py` - __init__ 方法

**修改前**:
```python
# ===== Reward 分量 TensorBoard logger =====
# 放到和 PPO 同目录的 reward_components 子目录下
exp_name = self.config.get('exp_name', 'default')
output_name = self.config.get('output_name', f'ShadowHandHora/{exp_name}')
rew_log_dir = os.path.join('outputs', output_name, 'reward_components')
```

**修改后**:
```python
# ===== Reward 分量 TensorBoard logger =====
# 从环境变量读取 output_name（由训练脚本设置）
output_name = os.environ.get('HORA_OUTPUT_NAME', 'ShadowHandHora/default')
rew_log_dir = os.path.join('outputs', output_name, 'reward_components')
```

**关键点**:
- 从环境变量 `HORA_OUTPUT_NAME` 读取
- 默认值为 `'ShadowHandHora/default'` 以保证向后兼容

### 修改 2: `scripts/train_shadow.sh` - 训练脚本

**修改前**:
```bash
#!/bin/bash
NAME=${1:-run_$(date +%m%d_%H%M)}

CUDA_VISIBLE_DEVICES=0 python train.py \
  task=ShadowHandHora train=ShadowHandHora headless=True \
  train.ppo.output_name=ShadowHandHora/${NAME} \
  "${@:2}"
```

**修改后**:
```bash
#!/bin/bash
NAME=${1:-run_$(date +%m%d_%H%M)}

export HORA_OUTPUT_NAME=ShadowHandHora/${NAME}  ← 新增此行

CUDA_VISIBLE_DEVICES=0 python train.py \
  task=ShadowHandHora train=ShadowHandHora headless=True \
  train.ppo.output_name=ShadowHandHora/${NAME} \
  "${@:2}"
```

**关键点**:
- 添加 `export HORA_OUTPUT_NAME=ShadowHandHora/${NAME}` 
- 这样 task 的 __init__ 就能通过环境变量读到正确的 output_name

---

## ?? 验证修复

### 运行训练脚本
```bash
bash scripts/train_shadow.sh exp4_lowhand
```

### 预期输出
启动日志中应该看到:
```
[ShadowHandHora] Reward components TB → outputs/ShadowHandHora/exp4_lowhand/reward_components
```

### 验证日志路径
```bash
# 列出生成的日志
ls -la outputs/ShadowHandHora/exp4_lowhand/reward_components/

# 应该看到 events.out.tfevents... 文件存在
```

### 打开 TensorBoard
```bash
tensorboard --logdir outputs/ShadowHandHora/exp4_lowhand --port 6006

# 在浏览器访问 http://localhost:6006
# 应该能看到 rewards/* 和 diagnostics/* 的 10 条曲线
```

---

## ?? 修复对比表

| 项目 | 修复前 | 修复后 |
|------|-------|--------|
| 从哪里读 output_name | `self.config` (错误的源) | `os.environ` (正确的源) |
| 环境变量 | 无 | `HORA_OUTPUT_NAME` |
| TB 日志路径 | `outputs/ShadowHandHora/default/...` | `outputs/ShadowHandHora/{exp_name}/...` |
| 多个实验是否隔离 | ? 全写到 default | ? 各自独立目录 |
| 与 PPO output_name 同步 | ? 不同步 | ? 完全同步 |

---

## ?? 环境变量流向图

```
train_shadow.sh
    ↓
export HORA_OUTPUT_NAME=ShadowHandHora/exp4_lowhand
    ↓
python train.py (子进程继承环境变量)
    ↓
ShadowHandHora.__init__()
    ↓
os.environ.get('HORA_OUTPUT_NAME')  ← 读取成功
    ↓
TB 日志写到正确路径:
outputs/ShadowHandHora/exp4_lowhand/reward_components/
```

---

## ?? 文件修改清单

| 文件 | 修改类型 | 行数 | 变更 |
|------|---------|------|------|
| `hora/tasks/shadow_hand_hora.py` | 代码修改 | 161-169 | 改用环境变量读 output_name |
| `scripts/train_shadow.sh` | 脚本修改 | 第 3 行新增 | `export HORA_OUTPUT_NAME=...` |

---

## ? 测试场景

### 场景 1: 使用默认 exp_name
```bash
bash scripts/train_shadow.sh
# 预期: TB 日志路径 = outputs/ShadowHandHora/run_MMDD_HHMM/reward_components/
```

### 场景 2: 指定 exp_name
```bash
bash scripts/train_shadow.sh exp4_lowhand
# 预期: TB 日志路径 = outputs/ShadowHandHora/exp4_lowhand/reward_components/
```

### 场景 3: 多次运行不同实验
```bash
bash scripts/train_shadow.sh exp1
bash scripts/train_shadow.sh exp2

# 预期结果:
# outputs/ShadowHandHora/exp1/reward_components/  ← 分别独立
# outputs/ShadowHandHora/exp2/reward_components/  ← 分别独立
# (修复前都会写到 default，相互覆盖)
```

---

## ?? 为什么用环境变量而不是其他方式？

1. **环境变量优点**:
   - ? 可跨进程传递（父进程→子进程）
   - ? 不需要修改 task config（config 就是 task 特定的）
   - ? 训练脚本可以完全控制
   - ? 与 PPO 的 `output_name` 参数完全同步

2. **为什么不能用 config**:
   - ? `output_name` 不在 task.config 中（在 train.ppo.config 中）
   - ? VecTask 初始化时还没读取 train config

3. **为什么不能用命令行参数**:
   - ? task 初始化时命令行参数已经解析完了
   - ? 需要额外的参数传递机制

---

## ?? 向后兼容性

修复代码添加了默认值:
```python
output_name = os.environ.get('HORA_OUTPUT_NAME', 'ShadowHandHora/default')
```

**含义**:
- 如果 `HORA_OUTPUT_NAME` 环境变量存在 → 使用它（新行为）
- 如果不存在 → 使用 `'ShadowHandHora/default'`（兼容旧行为）

即使没有更新训练脚本，代码也能正常运行（日志写到 default 目录）。

---

## ?? 完成状态

? **修复完成**

| 组件 | 状态 |
|------|------|
| `hora/tasks/shadow_hand_hora.py` | ? 已修改 |
| `scripts/train_shadow.sh` | ? 已修改 |
| 环境变量支持 | ? 就绪 |
| 向后兼容性 | ? 保证 |
| 验证文档 | ? 完成 |

**现在可以放心使用！** ??
