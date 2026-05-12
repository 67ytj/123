# DFC 几何诊断快速参考

## 一键诊断命令

```bash
cd ~/hora_ws

# 方法 1: 用脚本
bash /tmp/run_geom_diag.sh

# 方法 2: 直接命令
HORA_OUTPUT_NAME=geom_debug python train.py \
    task=ShadowHandHora train=ShadowHandHora headless=True num_envs=4 \
    2>&1 | grep -E "GEOM|DFC|RESET" | head -80
```

## 期望诊断输出

```
[RESET-DFC] hand_root_pos = [ 0.00  0.00  0.40]  (z 应该 < 0.60，通常 0.25-0.45)
[RESET-DFC] g[pos][0]     = [ 0.00  0.00  0.40]
[RESET-DFC] g[obj_z][0]   = 0.25 (lifted by 0.20)
[RESET-DFC] g[q][0]       = [0.12 0.85 0.93 0.93 ...]
[RESET-DFC-VERIFY] |hand_dof_pos - g[q]|.mean() = 0.00000  ? 必须 = 0

============================================================
[GEOM] hand_root_xyz = [ 0.00  0.00  0.40]
[GEOM] palm_xyz      = [ 0.00  0.00  0.38]
[GEOM] ball_xyz      = [ 0.00  0.00  0.25]
[GEOM] palm-ball     = [ 0.00  0.00  0.13]  (dist=0.130)  ? 应该 0.10-0.15m
[GEOM] FF_tip-ball   = [ 0.00  0.01 -0.03]  dist=0.031m  (球面距=0.006m)  ? 应该接触
[GEOM] MF_tip-ball   = [ 0.01  0.00 -0.03]  dist=0.032m  (球面距=0.007m)  ? 应该接触
[GEOM] RF_tip-ball   = [-0.01  0.00 -0.03]  dist=0.031m  (球面距=0.006m)  ? 应该接触
[GEOM] LF_tip-ball   = [ 0.00 -0.01 -0.03]  dist=0.030m  (球面距=0.005m)  ? 应该接触
[GEOM] TH_tip-ball   = [ 0.00  0.00 -0.04]  dist=0.040m  (球面距=0.015m)  ? 应该有点远
[GEOM] g[obj_eul][0] = [0.0  0.0]  (reset 时球还没转)
[GEOM] tip_F_mag     = [0. 0. 0. 0. 0.]  total=0.000N  ? reset 后无接触力
============================================================
```

## 诊断要点

### ? 正常情况 (DFC reset 工作)

1. **palm-ball 距离** ~0.10-0.15m
   - 过大(>0.20m) → hand_root_xyz 太高或 LIFT 太大
   - 过小(<0.05m) → hand 穿进球 (workspace 夹住了)

2. **tip-ball 距离** ~0.03m (球面距 ~0.005-0.010m)
   - 说明手指已经贴球
   - 不应该 > 0.05m (否则手太开)

3. **MAE (hand_dof_pos - g[q])**  = 0.00000
   - 说明 DOF 顺序对，g['q'] 成功写入

4. **tip_F_mag** reset 后第 0 帧 = 0
   - 符合预期（还没有运动学接触）

### ?? 异常情况

| 现象 | 原因 | 修复 |
|------|------|------|
| palm-ball > 0.20m | LIFT 太大或 hand_root_xyz 坐标错 | 检查 g['pos'] 的 z |
| palm-ball < 0.05m | workspace 被夹住 | 增大 `arm_xyz_workspace.z` 上限 |
| MAE > 0.01 | DOF 顺序排错 | 反向 `FINGER_DOF_NAMES_22` |
| tip_F_mag > 0 at reset | 有初始接触力(不正常) | 检查 g['q'] 是否过极端 |

## 数据来源验证

```
[GEOM] g[pos][0]     = [ X  Y  Z]     ← 应该 = hand_root_xyz
[GEOM] g[obj_z][0]   = Z_lifted       ← 应该 = ball_xyz[:, 2]
[GEOM] g[obj_eul][0] = [roll pitch]  ← 从 DFC 读出的物体初始姿态
[GEOM] g[q][0]       = [q0...q21]    ← 应该 = hand_dof_pos (22 dims)
```

## 后续步骤

如果诊断成功（palm-ball ~0.13m, MAE=0），可以：

1. **跑完整训练** (5-20M 步)
2. **看 TensorBoard** 的 F_total / contact / lift / success 曲线
3. **判断是否破 192 平台**

如果诊断失败，立即贴完整诊断输出，我来判断原因。

---

## 一句话检查清单

- [ ] hand_root_xyz z 值 < 0.60 ?
- [ ] palm-ball 距离 0.10-0.15m ?
- [ ] MAE = 0.00000 ?
- [ ] tip_F_mag = 0 at reset ?

全部 ? → 可以正式训练

---

生成时间: $(date)
