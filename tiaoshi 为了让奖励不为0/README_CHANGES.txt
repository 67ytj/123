╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              ? TensorBoard 奖励分量记录 - 修改完成 ?                      ║
║                                                                            ║
║                   ?? 拆分 TensorBoard 记录实现方案                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

?? 项目概述
════════════════════════════════════════════════════════════════════════════════

目标: 将原来只有 episode_rewards/step (总奖励) 的 TensorBoard 记录拆分成
      9 条详细的曲线，清楚地显示每个奖励分量和诊断指标。

修改范围: ? ONLY 1 FILE → hora/tasks/shadow_hand_hora.py


?? 修改清单
════════════════════════════════════════════════════════════════════════════════

? 修改 1: 文件顶部导入
   - 添加 `import os`
   - 添加 `from torch.utils.tensorboard import SummaryWriter`

? 修改 2: __init__ 末尾初始化
   - 创建 TensorBoard SummaryWriter 实例
   - 设置日志目录为 outputs/{output_name}/reward_components
   - 初始化日志计数器

? 修改 3: post_physics_step 中记录
   - 每 50 个环境步长记录一次
   - 记录 6 个奖励分量 (reach, lift_low, lift_mid, lift_high, penalty, total)
   - 记录 4 个诊断指标 (contact_force, ball_height, tip_dist, success_rate)


?? 新增 TensorBoard 指标 (共 10 条)
════════════════════════════════════════════════════════════════════════════════

奖励分量 (6条):
  rewards/reach           ← 指尖接近度奖励 (0-1)
  rewards/lift_low        ← 抬 4cm bonus
  rewards/lift_mid        ← 抬 8cm bonus
  rewards/lift_high       ← 抬 15cm bonus
  rewards/penalty         ← 动作惩罚 (通常 ≤ 0)
  rewards/total           ← 总奖励 = 前 5 者之和

诊断指标 (4条):
  diagnostics/tip_contact_force_mean  ← 平均接触力 (检查手指闭环)
  diagnostics/ball_height             ← 球被抬起的高度 (m)
  diagnostics/mean_tip_dist           ← 指尖到球的平均距离 (m)
  diagnostics/success_rate_4cm        ← 成功率 (球抬过 4cm 的比例, 0-1)


?? 诊断价值
════════════════════════════════════════════════════════════════════════════════

原病状: "Reward 在涨，但不知道是为什么涨"
        ↓ (无法区分是 reach 在涨还是 lift 在涨)

解决后: 清晰的 9 条曲线讲述 policy 的完整故事

┌─────────────────────────────────────────────────────────────────────────┐
│ 现象                    │ TB 曲线表现          │ 含义              │ 对策 │
├─────────────────────────────────────────────────────────────────────────┤
│ 只会伸手不会抓          │ reach ↑             │ reach local      │ 降权 │
│                        │ lift_* = 0          │ optimum          │ 重                     │
├─────────────────────────────────────────────────────────────────────────┤
│ 假装抓（拍飞球）        │ lift_high ↑         │ reward hack      │ 加   │
│                        │ contact_force ≈ 0  │                  │ 约束 │
├─────────────────────────────────────────────────────────────────────────┤
│ 健康学习（想要的）      │ lift_low → mid →   │ 阶梯式学习       │ ?   │
│                        │ high 依次点亮       │ trajectory       │      │
├─────────────────────────────────────────────────────────────────────────┤
│ 探索能力不足            │ 所有曲线都平        │ 学习率/样本量    │ 调   │
│                        │                    │ 不足             │ 参数 │
├─────────────────────────────────────────────────────────────────────────┤
│ 指尖索引错了            │ contact_force      │ 身体部位         │ 检查 │
│                        │ 始终 ≈ 0            │ 名称映射错误     │ 索引 │
└─────────────────────────────────────────────────────────────────────────┘


?? 使用步骤
════════════════════════════════════════════════════════════════════════════════

1?? 验证修改
   python test_syntax.py
   ? 应输出: ? [OK] File syntax is valid!

2?? 启动训练
   bash scripts/train_shadow.sh exp4_lowhand
   ? 应输出: [ShadowHandHora] Reward components TB → ...

3?? 打开 TensorBoard
   tensorboard --logdir outputs/ShadowHandHora --port 6006 --reload_multifile true
   ? 浏览器访问 http://localhost:6006

4?? 查看新增曲线
   SCALARS 标签页 → 应该看到 rewards/* 和 diagnostics/* 开头的 10 条曲线


?? 文件目录结构
════════════════════════════════════════════════════════════════════════════════

修改前:
  outputs/
  └── ShadowHandHora/
      └── exp4_lowhand/
          ├── events.out.tfevents...  (PPO 原始日志)
          └── ... (其他PPO相关文件)

修改后:
  outputs/
  └── ShadowHandHora/
      └── exp4_lowhand/
          ├── events.out.tfevents...  (PPO 原始日志)
          ├── ... (其他PPO相关文件)
          └── reward_components/       ← ?? 新子目录
              └── events.out.tfevents...  (9条新指标的日志)

关键: TensorBoard 指向 outputs/ShadowHandHora (父目录)，
      这样可以同时看到 PPO 原日志和 reward_components 子目录的内容


?? 日志数据量估算
════════════════════════════════════════════════════════════════════════════════

记录频率: 每 50 个环境步长 1 次
指标数:   10 个

按 50M 训练步长计算:
  记录次数 = 50,000,000 ÷ (50 × num_envs)

对于 num_envs = 64 (典型值):
  记录次数 = 50,000,000 ÷ 3,200 ≈ 15,600 次
  数据条数 = 15,600 × 10 = 156,000 条
  磁盘占用 ≈ 40-50 MB ? 可接受

对于 num_envs = 128:
  磁盘占用 ≈ 20-25 MB ? 更小


?? 配置依赖
════════════════════════════════════════════════════════════════════════════════

代码自动读取的配置项:
  - self.config.get('exp_name')      (默认: 'default')
  - self.config.get('output_name')   (默认: f'ShadowHandHora/{exp_name}')

这些通常在启动脚本中传入，例如:
  exp_name: exp4_lowhand
  output_name: ShadowHandHora/exp4_lowhand


??? 可选优化
════════════════════════════════════════════════════════════════════════════════

如果需要调整记录频率，修改 post_physics_step 中的条件:

当前 (每 50 步):
  if self.rew_log_counter % 50 == 0:

改为每 100 步 (记录数减半):
  if self.rew_log_counter % 100 == 0:

改为每 200 步 (记录数为原来四分之一):
  if self.rew_log_counter % 200 == 0:

然后需要删除旧日志:
  rm -rf outputs/ShadowHandHora/*/reward_components/


?? 排查指南
════════════════════════════════════════════════════════════════════════════════

问题: SyntaxError / ImportError
解决: 检查 Python >= 3.7, PyTorch >= 1.6
     pip install --upgrade torch tensorboard

问题: TensorBoard 看不到新曲线
解决: 1. 确认训练已跑超过 50 步
     2. 检查日志目录: ls outputs/ShadowHandHora/exp4_lowhand/reward_components/
     3. 刷新浏览器 (Ctrl+F5)

问题: 接触力始终为 0
解决: 检查 self.fingertip_body_indices 索引是否正确
     grep "Body names:" 训练日志

详见: TROUBLESHOOTING.md


?? 文档生成
════════════════════════════════════════════════════════════════════════════════

本次修改生成的文档:

  ? QUICKSTART.md            - 快速开始指南 (最常用)
  ? MODIFICATION_SUMMARY.md  - 详细修改内容说明
  ? TROUBLESHOOTING.md       - 常见问题和解决方案
  ? verify_changes.py        - 修改验证脚本
  ? test_syntax.py           - 语法检查脚本
  ? THIS FILE (README.txt)   - 总体概述


? 完成标志
════════════════════════════════════════════════════════════════════════════════

执行以下命令后看到这个输出，说明安装成功:

  $ tensorboard --logdir outputs/ShadowHandHora --port 6006

  # 在浏览器中应该能看到:

  SCALARS 标签页:
  ├── rewards/reach
  ├── rewards/lift_low
  ├── rewards/lift_mid
  ├── rewards/lift_high
  ├── rewards/penalty
  ├── rewards/total
  ├── diagnostics/tip_contact_force_mean
  ├── diagnostics/ball_height
  ├── diagnostics/mean_tip_dist
  └── diagnostics/success_rate_4cm

  ?? 恭喜！9 条新曲线已激活！


?? 后续步骤
════════════════════════════════════════════════════════════════════════════════

1. 运行 5M-10M 步实验，观察 9 条曲线的演变
2. 对比多个实验的曲线形状，找出最优的奖励参数组合
3. （可选）后续可以添加图像记录等更高级的监测功能


════════════════════════════════════════════════════════════════════════════════
修改完成日期: 2025
修改文件: hora/tasks/shadow_hand_hora.py (三处修改)
修改行数: ~40 行代码新增
新增属性: 2 个 (rew_writer, rew_log_counter)
新增指标: 10 条 TensorBoard 标量
════════════════════════════════════════════════════════════════════════════════
