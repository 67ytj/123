# DFC Reset Fix 检查清单

## ? 代码改动确认

### 文件: shadow_hand_hora.py
- [x] `reset_idx()` 完整替换（行 403-562）
  - [x] DFC 分支：`g['q']` 直接写 DOF，不被 noise 覆盖
  - [x] 非-DFC 分支：保留原 noise 初始化
  - [x] 共同段：统一 hand+object root state set
  - [x] J0 镜像：两个分支都有
  - [x] 验证打印：`[RESET-DFC-VERIFY]` MAE 检查
- [x] 删除 warm-up 代码（原 pre_physics_step 中）
- [x] LIFT = 0.20m（防止手进地面）

### 文件: ShadowHandHora.yaml
- [x] `arm_xyz_workspace.z`: [0.05, 0.60]（从 0.35 改为 0.60）
- [x] `alignToObject`: false（DFC 不需要对齐）

---

## ?? 三个隐患检查

### 隐患 1: Arm workspace 被夹住
**现象**: `[RESET-DFC] hand_root_pos` 的 z 值 > 0.40 时，第一步动作会把手夹下来
**检查方法**: 
```
grep "RESET-DFC] hand_root_pos" /tmp/hora_dfc_test.log | head -1
# 如果 z > 0.40，说明 workspace 需要更大（已改为 0.60，应该不会撞上）
```

### 隐患 2: DOF 顺序排错
**现象**: `[RESET-DFC-VERIFY]` 的 MAE > 0.01
**检查方法**:
```
grep "RESET-DFC-VERIFY\] |hand_dof_pos" /tmp/hora_dfc_test.log
grep "actual\[:8\]" /tmp/hora_dfc_test.log
grep "expect\[:8\]" /tmp/hora_dfc_test.log

# 如果三行都是 0，? 顺序对
# 如果 MAE > 0 但数字对得上，说明顺序反了，改 FINGER_DOF_NAMES_22
```

### 隐患 3: alignToObject 冲突
**现象**: DFC 模式下手位置被偏移
**检查方法**: 已改 yaml 为 false，应该不会撞上

---

## ?? 启动命令

```sh
cd ~/hora_ws

# 备份
cp hora/tasks/shadow_hand_hora.py \
   hora/tasks/shadow_hand_hora.py.bak_dfc_ready

# 清理旧输出
rm -rf outputs/shadow_debug

# 启动训练
conda activate hora
HORA_OUTPUT_NAME=shadow_debug python train.py \
    task=ShadowHandHora \
    train=ShadowHandHora \
    headless=True \
    2>&1 | tee /tmp/hora_dfc_test.log

# 或用脚本
bash /tmp/run_dfc_test.sh
```

---

## ?? 关键日志提取

启动后立刻运行：
```sh
# 提取前 20 秒的关键日志
sleep 5 && head -100 /tmp/hora_dfc_test.log | grep -E "RESET-DFC|RESET-DFC-VERIFY|hand_root|g\[pos\]|MAE|actual\[:8\]|expect\[:8\]"
```

期望看到：
```
[RESET-DFC] hand_root_pos = [ 0.xx  0.yy  0.zz]       (z 应该 < 0.60)
[RESET-DFC] obj_pos       = [ 0.xx  0.yy  0.xx+0.20]
[RESET-DFC] g[pos][0]     = [ 0.xx  0.yy  0.zz]       (lifted by LIFT=0.20)
[RESET-DFC] g[obj_z][0]   = 0.xx
[RESET-DFC] g[q][0]       = [0.12 0.85 0.93 ...]
[RESET-DFC-VERIFY] |hand_dof_pos - g[q]|.mean() = 0.00000  (must be ~0)
[RESET-DFC-VERIFY] actual[:8] = [0.12 0.85 0.93 ...]
[RESET-DFC-VERIFY] expect[:8] = [0.12 0.85 0.93 ...]
```

---

## ?? 训练预期目标 (5-20M 步)

| 指标 | 修复前 | 修复后预期 |
|------|--------|----------|
| `diagnostics/F_total` | 0.17-0.23N | 启动就 0.8-3N |
| `rewards/contact` | ~0 | 0.3-0.8 |
| `rewards/lift` | 0 | 0.2-1.0 |
| `rewards/success` | 0 | >0 |
| Total reward | 192 卡死 | 突破 250+ |

---

## ?? 收集信息清单

运行 1-2M 步后，把以下信息贴给我：

1. **启动日志**（前 30 秒）
2. **TensorBoard 截图** (TB 显示 5-20M 步的曲线)
   - `diagnostics/F_total`
   - `rewards/contact`
   - `rewards/lift`
   - `rewards/success`
   - `rewards/total`
3. **关键数字** (grep 结果)
   - hand_root_pos z 值
   - MAE 值

---

## ? 完成度检查

- [x] 配置文件改对
- [x] reset_idx 代码改对
- [x] 三个隐患都排查了
- [x] 启动脚本准备好
- [ ] 实际跑过一次（待执行）
