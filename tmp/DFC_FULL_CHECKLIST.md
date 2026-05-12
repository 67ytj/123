# ?? DFC Reset Fix 完整执行清单

## ? 代码修改状态

### shadow_hand_hora.py
- [x] reset_idx 完整替换 (行 403-562)
  - [x] DFC 分支：g['q'] 直接写 DOF
  - [x] 非-DFC 分支：保留 noise 初始化
  - [x] 共同段：统一 hand+object root state set
  - [x] J0 镜像：两分支都有
  - [x] VERIFY 打印：MAE 检查
- [x] 一次性几何诊断脚本：[GEOM] 输出

### ShadowHandHora.yaml
- [x] arm_xyz_workspace.z: [0.05, 0.60] (从 0.35 升级)
- [x] alignToObject: false (DFC 不需要对齐)

---

## ?? 诊断流程

### Step 1: 验证代码 (2 分钟)

```bash
cd ~/hora_ws

# 检查 reset_idx 是否改对
grep -n "def reset_idx" hora/tasks/shadow_hand_hora.py
# 应该显示: 403

# 检查 DFC 分支长度
sed -n '404,500p' hora/tasks/shadow_hand_hora.py | grep -c "GEOM"
# 应该显示: 1 (诊断脚本存在)

# 检查 YAML 改对没
grep "arm_xyz_workspace\|alignToObject" configs/task/ShadowHandHora.yaml
# 应该显示:
#   z: [0.05, 0.60]
#   alignToObject: false
```

### Step 2: 运行几何诊断 (5 分钟)

```bash
cd ~/hora_ws

# 备份代码
cp hora/tasks/shadow_hand_hora.py \
   hora/tasks/shadow_hand_hora.py.bak_geom_ready

# 清理输出
rm -rf outputs/geom_debug

# 启动诊断
bash /tmp/run_geom_diag.sh
```

### Step 3: 读诊断输出

期望看到这 6 组关键数据（按顺序）：

#### A. RESET-DFC-VERIFY
```
[RESET-DFC-VERIFY] |hand_dof_pos - g[q]|.mean() = 0.00000  ← 必须 = 0
```
- **0.00000** ? DOF 顺序对，g['q'] 成功写入
- "> 0.01" ? DOF 顺序排错，需要反向 FINGER_DOF_NAMES_22

#### B. GEOM hand_root_xyz
```
[GEOM] hand_root_xyz = [ 0.00  0.00  0.40]
```
- **z 值 < 0.60** ? workspace 没被夹住
- **z 值 > 0.60** ? 隐患 1，需要改 LIFT 或再增大 workspace

#### C. GEOM palm-ball 距离
```
[GEOM] palm-ball = [ 0.00  0.00  0.13]  (dist=0.130)
```
- **0.10-0.15m** ? 合理的抓姿
- "< 0.05m" ? 手指太紧或穿进球
- "> 0.20m" ? 手太开，g['q'] 可能不够紧

#### D. GEOM tip-ball 距离（5 根手指）
```
[GEOM] FF_tip-ball = [ 0.00  0.01 -0.03]  dist=0.031m  (球面距=0.006m)
[GEOM] MF_tip-ball = [ 0.01  0.00 -0.03]  dist=0.032m  (球面距=0.007m)
[GEOM] RF_tip-ball = [-0.01  0.00 -0.03]  dist=0.031m  (球面距=0.006m)
[GEOM] LF_tip-ball = [ 0.00 -0.01 -0.03]  dist=0.030m  (球面距=0.005m)
[GEOM] TH_tip-ball = [ 0.00  0.00 -0.04]  dist=0.040m  (球面距=0.015m)
```
- **球面距 0.005-0.010m** ? 手指接触球表面
- "< 0.003m" 或 "穿进" ? g['q'] 太极端
- "> 0.020m" ? 手太开

#### E. GEOM g[q][0]
```
[GEOM] g[q][0] = [0.12 0.85 0.93 0.93 0.10 0.82 0.89 0.89 ...]
```
- **0.1-0.95 之间** ? 正常手指角度
- "0.0 或 1.0" ? 角度极端，可能穿进

#### F. GEOM tip_F_mag
```
[GEOM] tip_F_mag = [0. 0. 0. 0. 0.]  total=0.000N
```
- **= 0** ? reset 后无接触力（符合预期）
- "> 0" ? 有初始接触力（不正常，可能是时间积分或碰撞检测问题）

---

## ?? 诊断通过标准

### 必要条件 (全部 ? 才能继续)

- [ ] MAE = 0.00000 (±1e-5)
- [ ] hand_root_xyz z < 0.60
- [ ] palm-ball dist 0.10-0.15m
- [ ] FF/MF/RF/LF/TH tip-ball 球面距都 > 0.003m
- [ ] tip_F_mag[all] = 0 at reset

### 可接受的偏差

- hand_root_xyz z: 0.20-0.50m (只要 < 0.60)
- palm-ball: 0.08-0.18m (只要有接触，不穿进)
- tip_F_mag: < 0.001N (浮点噪声)

---

## ?? 通过诊断后

1. **运行完整训练** (5-20M 步)
   ```bash
   HORA_OUTPUT_NAME=shadow_debug python train.py \
       task=ShadowHandHora train=ShadowHandHora headless=True \
       2>&1 | tee /tmp/hora_full_train.log
   ```

2. **监控 TensorBoard** (同时启用)
   ```bash
   tensorboard --logdir outputs/shadow_debug/reward_components --port 6006
   ```

3. **查看关键曲线**
   - `diagnostics/F_total`: 启动就 > 0.5N，后续 1-3N
   - `rewards/contact`: 0.3+
   - `rewards/lift`: 0.2+
   - `rewards/success`: > 0 (开始有成功)
   - `rewards/total`: 突破 250+

---

## ?? 快速命令集

```bash
# 1. 诊断
cd ~/hora_ws && bash /tmp/run_geom_diag.sh

# 2. 过滤关键日志
grep -E "GEOM|MAE|hand_root|palm-ball|tip-ball|tip_F" /tmp/hora_dfc_test.log | head -60

# 3. 完整训练
conda activate hora
HORA_OUTPUT_NAME=shadow_debug python train.py \
    task=ShadowHandHora train=ShadowHandHora headless=True \
    2>&1 | tee /tmp/hora_full_train.log

# 4. TensorBoard
tensorboard --logdir outputs/shadow_debug/reward_components --port 6006
```

---

## ?? 故障排查

| 症状 | 原因 | 修复 |
|------|------|------|
| MAE > 0.01 | DOF 顺序反了 | `sed -i ... FINGER_DOF_NAMES_22` 反向 |
| hand_root z > 0.60 | workspace 夹住 | 再增大 arm_xyz_workspace.z 上限 |
| palm-ball < 0.05m | 手穿进球 | 检查 g['q'] 是否合理，或改 LIFT |
| tip_F_mag > 0.01N | 碰撞检测延迟 | 刷新 refresh 顺序或增大 sim substeps |

---

## ? 预期成功标志

诊断运行后看到这样的完整输出块：

```
============================================================
[GEOM] hand_root_xyz = [ 0.00  0.00  0.40]
[GEOM] palm_xyz      = [ 0.00  0.00  0.38]
[GEOM] ball_xyz      = [ 0.00  0.00  0.25]
[GEOM] palm-ball     = [ 0.00  0.00  0.13]  (dist=0.130)
[GEOM] FF_tip-ball   = [ 0.00  0.01 -0.03]  dist=0.031m  (球面距=0.006m)
[GEOM] MF_tip-ball   = [ 0.01  0.00 -0.03]  dist=0.032m  (球面距=0.007m)
[GEOM] RF_tip-ball   = [-0.01  0.00 -0.03]  dist=0.031m  (球面距=0.006m)
[GEOM] LF_tip-ball   = [ 0.00 -0.01 -0.03]  dist=0.030m  (球面距=0.005m)
[GEOM] TH_tip-ball   = [ 0.00  0.00 -0.04]  dist=0.040m  (球面距=0.015m)
[GEOM] g[pos][0]     = [0.0 0.0 0.4]
[GEOM] g[obj_z][0]   = 0.25
[GEOM] g[obj_eul][0] = [0. 0.]
[GEOM] g[q][0]       = [0.12 0.85 0.93 0.93 ...]
[GEOM] tip_F_mag     = [0. 0. 0. 0. 0.]  total=0.000N
============================================================
[RESET-DFC-VERIFY] |hand_dof_pos - g[q]|.mean() = 0.00000  (must be ~0)
```

**看到这个就 ? 可以正式训练了！**

---

生成时间: 2024
版本: DFC Reset Fix v1.0
