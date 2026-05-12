# ?? Ball Scale 与 Geometry 诊断清单

## 一键诊断命令

```bash
cd ~/hora_ws

# 方法 1: 用脚本
bash /tmp/run_scale_geom_diag.sh 2>&1 | tee /tmp/scale_check.log

# 方法 2: 直接命令
HORA_OUTPUT_NAME=scale_geom_debug python train.py \
    task=ShadowHandHora train=ShadowHandHora headless=True num_envs=4 \
    2>&1 | grep -E "DFC-SCALES|DFC-SCALE|GEOM\]"
```

---

## ?? 诊断输出解读

### A. Ball 物体 Scale 列表

**期望看到：**
```
[DFC-SCALES-AVAIL] Ball-xxx: [0.06, 0.08, 0.10, ...]
```

**检查要点：**
- 是否有 `0.06` scale（题目提到的）
- 有几种 scale 可选（通常 3-5 种）
- picked scale 是哪个（应该 ≈ 0.08，根据 `targetScale=0.08`）

### B. DFC Scale 信息 (sample 时打印)

**期望看到：**
```
[DFC-SCALE] obj_code     = Ball-xxx
[DFC-SCALE] picked scale = 0.08
[DFC-SCALE] N poses      = 5700
[DFC-SCALE] q[0]         = [0.12 0.85 0.93 ...]
[DFC-SCALE] pos[0]       = [0.00 0.00 0.25]
[DFC-SCALE] obj_z[0]     = 0.05
[DFC-SCALE] obj_eul[0]   = [0.0 0.0]
```

**检查要点：**
- `picked scale` 值（应该是 0.06-0.10 之间）
- `N poses` 数量（应该 > 100，通常 5000+）
- `q[0]` 角度范围（应该 0.0-1.0 之间）
- `pos[0]` z 坐标（对应球的初始高度，通常 0.03-0.10）

### C. 几何诊断 (GEOM 块)

**期望看到：**
```
============================================================
[GEOM] base_obj_scale = 0.8
[GEOM] hand_root_xyz = [ 0.00  0.00  0.40]
[GEOM] palm_xyz      = [ 0.00  0.00  0.38]
[GEOM] ball_xyz      = [ 0.00  0.00  0.25]
[GEOM] palm-ball_xyz = [ 0.00  0.00  0.13]
[GEOM] palm-ball_dist= 0.1300m
[GEOM] FF: tip-ball=[...] dist=0.0310m  F=0.000N
[GEOM] MF: tip-ball=[...] dist=0.0320m  F=0.000N
[GEOM] RF: tip-ball=[...] dist=0.0310m  F=0.000N
[GEOM] LF: tip-ball=[...] dist=0.0300m  F=0.000N
[GEOM] TH: tip-ball=[...] dist=0.0400m  F=0.000N
============================================================
```

**检查要点：**
- `base_obj_scale`: 应该是 0.8 (config 中的 baseObjScale)
- `ball_xyz` z 坐标: 应该 ~0.25 (= g['obj_z'] + LIFT)
- `palm-ball_dist`: 应该 0.10-0.15m（合理抓距）
- 每根手指的 `dist`: 应该 0.025-0.040m（接触球表面）
- 每根手指的 `F`: reset 后应该 ~0（无初始接触力）

---

## ?? 关键数据点对比

| 数据 | 预期值 | 实际值 | 说明 |
|------|--------|--------|------|
| picked scale | 0.06-0.10 | | DFC 选中的 ball size |
| N poses | > 100 | | DFC 数据集 pose 数量 |
| base_obj_scale | 0.8 | | config 中的缩放因子 |
| ball_xyz z | 0.20-0.45 | | 球的实际高度 (LIFT=0.20) |
| palm-ball_dist | 0.10-0.15m | | 手掌到球的距离 |
| FF/MF/RF/LF dist | 0.025-0.040m | | 每根手指到球的距离 |
| TH dist | 0.030-0.050m | | 大拇指到球的距离 (可能略远) |
| tip F | ~0 N | | reset 后无接触力 |

---

## ?? 常见问题排查

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| 没有 `DFC-SCALES-AVAIL` 输出 | DFC 数据没加载 | 检查 `assets/datasetv4.1_posedata.npy` 路径 |
| MAE > 0.01 | DOF 顺序排错 | 检查 FINGER_DOF_NAMES_22 顺序 |
| palm-ball_dist < 0.05m | 手穿进球 | g['q'] 太极端或球太大 |
| palm-ball_dist > 0.20m | 手太开 | 减小 LIFT 或调整 g['q'] |
| tip F > 0.01N | 初始碰撞力 | 检查物理参数或刚体 refresh 顺序 |

---

## ? 诊断通过标准

- [ ] 有 `DFC-SCALES-AVAIL` 输出，显示 Ball-xxx scales
- [ ] `picked scale` 在 0.06-0.10 范围内
- [ ] `N poses` > 100
- [ ] `GEOM` 块输出完整
- [ ] `palm-ball_dist` 0.10-0.15m
- [ ] 所有手指 `dist` 0.025-0.050m
- [ ] 所有手指 `F` ≈ 0 N

**全部 ?** → 可以进行完整训练

---

## 下一步

1. **诊断成功** → 修改配置进行完整训练
2. **有问题** → 根据上表修复代码/配置，重新诊断

