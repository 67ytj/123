# ? Bug 修复完成 - 环境变量问题解决

## ?? 问题回顾

你指出的 bug：`output_name` 配置项不在 task config 中，导致 TensorBoard 日志写到错误的路径。

## ? 修复内容

### 修改 1：`hora/tasks/shadow_hand_hora.py` (第 161-168 行)

```python
# ===== Reward 分量 TensorBoard logger =====
# 从环境变量读取 output_name（由训练脚本设置）
output_name = os.environ.get('HORA_OUTPUT_NAME', 'ShadowHandHora/default')
rew_log_dir = os.path.join('outputs', output_name, 'reward_components')
os.makedirs(rew_log_dir, exist_ok=True)
self.rew_writer = SummaryWriter(log_dir=rew_log_dir)
self.rew_log_counter = 0
print(f'[ShadowHandHora] Reward components TB → {rew_log_dir}')
```

**关键改动**：
- ? 改为从 `os.environ.get('HORA_OUTPUT_NAME', ...)` 读取
- ? 提供默认值 `'ShadowHandHora/default'` 保证向后兼容

### 修改 2：`scripts/train_shadow.sh` (第 3 行新增)

```bash
#!/bin/bash
NAME=${1:-run_$(date +%m%d_%H%M)}

export HORA_OUTPUT_NAME=ShadowHandHora/${NAME}  # ← 新增此行

CUDA_VISIBLE_DEVICES=0 python train.py \
  task=ShadowHandHora train=ShadowHandHora headless=True \
  train.ppo.output_name=ShadowHandHora/${NAME} \
  "${@:2}"
```

**关键改动**：
- ? 在 python 命令执行前导出 `HORA_OUTPUT_NAME` 环境变量
- ? 与 `train.ppo.output_name` 参数保持同步

---

## ?? 现在可以这样使用

```bash
# 启动训练
bash scripts/train_shadow.sh exp4_lowhand

# 启动日志会显示正确的 TB 路径:
# [ShadowHandHora] Reward components TB → outputs/ShadowHandHora/exp4_lowhand/reward_components

# 打开 TensorBoard
tensorboard --logdir outputs/ShadowHandHora/exp4_lowhand --port 6006

# 浏览器访问 http://localhost:6006 查看 9 条新增曲线
```

---

## ?? 修复前后对比

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| TB 日志路径 | 固定写到 `default/` | 正确写到 `{exp_name}/` |
| 多个实验隔离 | ? 互相覆盖 | ? 各自独立 |
| 与 PPO output_name 同步 | ? 不同步 | ? 完全同步 |

---

## ?? 修改文件列表

| 文件 | 修改 | 状态 |
|------|------|------|
| `hora/tasks/shadow_hand_hora.py` | 第 161-168 行 | ? 已完成 |
| `scripts/train_shadow.sh` | 第 3 行新增 | ? 已完成 |

---

## ? 验证修复

运行验证脚本：
```bash
python verify_bug_fix.py
```

预期输出：
```
? 从环境变量读取 HORA_OUTPUT_NAME
? 默认值设置正确
? 导出环境变量
? 使用脚本参数
? 环境变量读取成功: ShadowHandHora/test_exp

? 所有修复已验证成功！
```

---

## ?? 验收标准

启动训练后，检查：

1. ? 日志输出包含正确的 TB 路径
   ```
   [ShadowHandHora] Reward components TB → outputs/ShadowHandHora/exp4_lowhand/reward_components
   ```

2. ? 文件系统中生成了 TB 日志文件
   ```bash
   ls outputs/ShadowHandHora/exp4_lowhand/reward_components/
   # 应该看到 events.out.tfevents... 文件
   ```

3. ? TensorBoard 能正常读取并显示 10 条新增曲线

---

## ?? 相关文档

- `BUG_FIX_SUMMARY.md` - 详细的修复分析和测试场景
- `verify_bug_fix.py` - 自动化验证脚本
- `QUICKSTART.md` - 快速启动指南

---

**修复完成！?? 现在可以放心训练了。** ?
