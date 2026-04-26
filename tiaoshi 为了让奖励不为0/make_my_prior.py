import numpy as np

# 1. 读取原始 3.6cm 数据
expert_data = np.load("data_1773900223188_2.npy", allow_pickle=True).item()

qpos = expert_data['joint_angles']
print("�� 修改前的手指角度:\n", qpos)

# 2. 让手指微张 (专门适配 4cm 大球)
# 索引 1, 2, 5, 6, 9, 10, 14, 15 是 Allegro 机械手主要控制手指弯曲的关节
bend_joints = [1, 2, 5, 6, 9, 10, 14, 15] 

for idx in bend_joints:
    # 减去 0.08 弧度（大概 4.5 度），让手指往外稍微扩一点点
    qpos[idx] -= 0.08  

# 3. 将修改后的角度和新的球体半径更新到字典中
expert_data['joint_angles'] = qpos
expert_data['radius'] = 0.040 

# 4. 另存为你的专属 4cm 专家数据
np.save("my_expert_data_4cm.npy", expert_data)
print("\n✅ 成功！专属数据已生成：my_expert_data_4cm.npy")