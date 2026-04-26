# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float
from glob import glob
from hora.utils.misc import tprint
from .base.vec_task import VecTask


class AllegroHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        self.device = sim_device # 提前定义，防止 Patch A 报错

        # ==========================================
        # �� 补丁 A：强制覆盖 Config 字典，对齐 19 维架构
        # ==========================================
        config['env']['numActions'] = 19
        config['env']['numObservations'] = 102 
        config['env']['object']['type'] = 'simple_tennis_ball'
        self.num_actions = 19

        # ==========================================
        # �� 专家数据池加载 (支持多个 .npy 文件随机抽取)
        # ==========================================# 1. 获取并【强制排序】专家文件列表
       # ==========================================
        # �� 专家数据池加载 (带强制排序)
        # ==========================================
        import glob
        import os
        current_dir = os.path.dirname(os.path.realpath(__file__))
        raw_files = glob.glob(os.path.join(current_dir, "*.npy"))
        
        # ⚠️ 致命关键：强制排序！
        self.expert_files = sorted(raw_files)  

        print("\n" + "��"*15)
        print("【专家数据加载顺序 (ID 映射)】")
        for i, f in enumerate(self.expert_files):
            print(f"  ID {i:02d} -> {os.path.basename(f)}")
        print("��"*15 + "\n")

        self.expert_qpos_pool = []
        self.expert_rel_pos_pool = []
        self.expert_quat_pool = []
        self.expert_ball_radius_pool = []
        
        # ⚠️ 注意这里必须用 self.expert_files
        if not self.expert_files:
            print("⚠️ 警告: 没找到任何 .npy 文件！请检查文件是否放在任务脚本同级目录下。")
            self.expert_qpos_pool.append(torch.zeros(16, device=self.device))
            self.expert_rel_pos_pool.append(torch.tensor([0.0, 0.0, 0.15], device=self.device))
            self.expert_quat_pool.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device))
            self.expert_ball_radius_pool.append(torch.tensor(0.04, device=self.device))
        else:
            for f in self.expert_files:  # ⚠️ 注意这里也是 self.expert_files
                try:
                    data = np.load(f, allow_pickle=True).item()
                    self.expert_qpos_pool.append(torch.tensor(data['joint_angles'], dtype=torch.float, device=self.device))
                    self.expert_rel_pos_pool.append(torch.tensor(data['rel_pos'], dtype=torch.float, device=self.device))
                    
                    raw_quat = torch.tensor(data['hand_quat'], dtype=torch.float, device=self.device)
                    self.expert_quat_pool.append(raw_quat / torch.norm(raw_quat))
                    
                    radius = data.get('ball_radius', 0.04)
                    self.expert_ball_radius_pool.append(torch.tensor(radius, dtype=torch.float, device=self.device))
                    print(f"✅ 成功加入专家库: {os.path.basename(f)}")
                except Exception as e:
                    print(f"❌ 加载 {f} 失败: {e}")


        self.expert_qpos_pool = torch.stack(self.expert_qpos_pool)    # [K, 16]
        self.expert_rel_pos_pool = torch.stack(self.expert_rel_pos_pool) # [K, 3]
        self.expert_quat_pool = torch.stack(self.expert_quat_pool)       # �� [K, 4]
        self.expert_ball_radius_pool = torch.stack(self.expert_ball_radius_pool) # �� [K]
        
        self.num_experts = self.expert_qpos_pool.shape[0]
        # 为每个环境准备一个专家索引桶
        self.env_expert_idx = torch.zeros(config['env']['numEnvs'], dtype=torch.long, device=self.device)
        

        # ==========================================
        # �� 定义“张开手”姿态 (核心！防止重置穿模)
        # ==========================================
        self.open_qpos = torch.tensor([
            0.0, 0.1, 0.1, 0.1,  # FF
            0.0, 0.1, 0.1, 0.1,  # MF
            0.0, 0.1, 0.1, 0.1,  # RF
            1.1, 0.3, 0.1, 0.1   # TH (大拇指外展，留出空间给球)
        ], dtype=torch.float, device=self.device)

        # 【修正】底座移动限制
        self.base_ema_alpha = 0.7
        self.base_ws_x = [-0.3, 0.3]
        self.base_ws_y = [-0.3, 0.3]
        self.base_ws_z = [0.05, 0.8]

        # ==========================================
        #  VecTask 标准初始化流程
        # ==========================================
        self._setup_domain_rand_config(config['env']['randomization'])
        self._setup_priv_option_config(config['env']['privInfo'])
        self._setup_object_info(config['env']['object'])
        self._setup_reward_config(config['env']['reward'])
        
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3), 'obj_scale': (3, 4), 'obj_mass': (4, 5),
            'obj_friction': (5, 6), 'obj_com': (6, 9),
        }

        # 调用父类构造函数
        super().__init__(config, sim_device, graphics_device_id, headless)

        # �� 新增：在启动前，先为每个环境随机分配一个专家
        # 这样第一局动作就是多样化的
        self.env_expert_idx = torch.randint(
            0, self.num_experts, (self.num_envs,), device=self.device
        )

        self.debug_viz = self.config['env']['enableDebugVis']
        self.debug_viz = self.config['env']['enableDebugVis']
        # �� 加上这行相册模式开关 (默认关闭)
        self.enable_expert_album = bool(self.config['env'].get('enableExpertAlbum', False))
        # �� [新增] 完整抓取开关：默认 False (代表从 15cm 高空张开手下降)
        # 如果设为 True，则开局直接瞬移到专家抓取姿态
        self.use_expert_reset = bool(self.config['env'].get('useExpertResetPose', False))
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        # 控制参数
        self.base_max_speed = 0.5 
        self.base_max_rot_speed = 1.0

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # 获取 GPU tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # ==========================================
        # �� 修正：初始化所有环境的控制目标为“张开姿态”
        # ==========================================
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        open_pose_batch = self.open_qpos.unsqueeze(0).expand(self.num_envs, -1)
        self.prev_targets[:, :self.num_allegro_hand_dofs] = open_pose_batch
        self.cur_targets[:, :self.num_allegro_hand_dofs] = open_pose_batch

        # 随机扰动力参数
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.saved_grasping_states = {} 
        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.d_gain

        # 统计数据
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

        # ==========================================
        # �� 补丁 B：底层空间劫持，欺骗 PPO 算法
        # ==========================================
        import gym
        self.act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)

# [调试用] 专家索引计数器
        self.debug_exp_idx = 0
        
        # 绑定按键 (加上相册开关保护)
        if self.viewer and self.enable_expert_album:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "next_expert")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "prev_expert")


    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        # set allegro_hand dof properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        
        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 1.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
# === 【B】仿真器 DOF 验血代码开始 ===
            if i == 0:
                # 1. 获取仿真器里的乱序真实名字
                sim_dof_names = self.gym.get_actor_dof_names(env_ptr, hand_actor)
                
                # 2. 定义专家数据录制时的标准逻辑顺序 (0到15)
                canonical_names = [f"joint_{j}.0" for j in range(16)]
                name_to_canonical_idx = {name: idx for idx, name in enumerate(canonical_names)}
                
                # 3. 计算洗牌索引
                perm_indices = [name_to_canonical_idx[name] for name in sim_dof_names]
                self.expert_dof_perm = torch.tensor(perm_indices, dtype=torch.long, device=self.device)

                # ==========================================
                # �� 全局终极洗牌：一劳永逸！
                # ==========================================
                # 把张开手的姿态永久洗牌成物理引擎顺序
                self.open_qpos = self.open_qpos[self.expert_dof_perm]
               
                
                # 把整个专家姿态池永久洗牌成物理引擎顺序！
                self.expert_qpos_pool = self.expert_qpos_pool[:, self.expert_dof_perm]
                # ==========================================

                print("\n" + "��"*15)
                print("【全局数据已永久洗牌，与物理引擎 100% 对齐！】")
                print(f"仿真器顺序: {sim_dof_names}")
                print(f"洗牌索引 (Perm): {perm_indices}")
                print("��"*15 + "\n")
            # === 【B】仿真器 DOF 验血代码结束 ===
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
            
            # 找到这一行
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)

# �� 在它下面增加打印
            if i == 0: # 只打印第一个环境，防止刷屏
    # 假设 URDF 里定义的原始半径是 0.04m
                print(f"--- 物理引擎初始化 ---")
                print(f"球体原始半径: 0.04m")
                print(f"当前缩放比例 (Scale): {obj_scale:.4f}")
                print(f"实际物理半径: {0.04 * obj_scale:.4f}m")
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # 清理了多余的重复定义
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # =========================================================================
        # �� 修复 1：保存初始手掌状态 (解决 AttributeError: hand_start_states)
        # =========================================================================
        hand_pose, _ = self._init_object_pose()
        initial_hand_quat = torch.tensor(
            [hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w],
            device=self.device, dtype=torch.float
        )
        self.hand_start_states = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        # 把刚才 _init_object_pose 里算好的位置直接传进来，绝不写死！
        self.hand_start_states[:, 0] = hand_pose.p.x
        self.hand_start_states[:, 1] = hand_pose.p.y
        self.hand_start_states[:, 2] = hand_pose.p.z # z（与 _init_object_pose 里的初始高度一致）
        self.hand_start_states[:, 3:7] = initial_hand_quat.unsqueeze(0).repeat(self.num_envs, 1)

        # =========================================================================
        # 缓存 hand 的完整初始 root state (13维)，用于 reset + 速度控制闭环
        # =========================================================================
        self.hand_init_state = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float)
        self.hand_init_state[:, 0:3] = self.hand_start_states[:, 0:3]   # pos
        self.hand_init_state[:, 3:7] = self.hand_start_states[:, 3:7]   # quat
        # linvel and angvel remain zero
        self.base_vel_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)


       # =========================================================================
        # �� 修复 2：获取手掌和指尖的刚体索引 (Rigid Body Indices)
        # =========================================================================
        env_ptr0 = self.envs[0]
        hand_actor0 = self.gym.find_actor_handle(env_ptr0, 'hand')

        # 打印出当前机械手所有的刚体名字，方便你找对名字！
        body_names = self.gym.get_actor_rigid_body_names(env_ptr0, hand_actor0)
        print("\n" + "="*50)
        print("�� 机械手包含的所有刚体名称如下：")
        print(body_names)
        print("="*50 + "\n")

        # 获取手掌基座索引
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'base_link')

        # =======================================================
        # �� 必须修改：使用带有 DIGIT 传感器物理厚度的 Tip
        # =======================================================
        ff_name = 'link_3.0_tip' 
        mf_name = 'link_7.0_tip' 
        rf_name = 'link_11.0_tip' 
        th_name = 'link_15.0_tip' 

        self.ff_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, ff_name)  
        self.mf_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, mf_name)  
        self.rf_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, rf_name)  
        self.th_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, th_name)  
        
        # 终极保险：只要有一个是 -1，说明名字写错了，立刻让程序崩溃并报错！
        assert self.ff_idx != -1, f"找不到食指刚体: {ff_name}！请检查终端上面打印出来的列表，看看叫啥名字！"
        assert self.mf_idx != -1, f"找不到中指刚体: {mf_name}！"
        assert self.rf_idx != -1, f"找不到无名指刚体: {rf_name}！"
        assert self.th_idx != -1, f"找不到拇指刚体: {th_name}！"

       


    def check_debug_keys(self):
        # 检查是否有按键事件发生
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0: # 只有按下时触发，抬起不触发
                if evt.action == "next_expert":
                    self.debug_exp_idx = (self.debug_exp_idx + 1) % self.num_experts
                elif evt.action == "prev_expert":
                    self.debug_exp_idx = (self.debug_exp_idx - 1) % self.num_experts
                
                # 核心：按键后立刻强制重置所有环境，刷新到新的专家位姿
                self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        # -------------------- 0) env0 判断（可靠做法） --------------------
        has_env0 = (env_ids == 0).any().item()

        # -------------------- 1) PD 随机化 --------------------
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper,
                (len(env_ids), self.num_dofs), device=self.device
            ).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper,
                (len(env_ids), self.num_dofs), device=self.device
            ).squeeze(1)

# -------------------- 2) 专家选择（相册模式 vs 训练模式） --------------------
        if hasattr(self, 'enable_expert_album') and self.enable_expert_album:
            # 【相册模式】：只控制 env0 的 expert_idx，方便键盘精准调试
            if has_env0:
                self.env_expert_idx[0] = int(self.debug_exp_idx)
        else:
            # 【训练模式】：全随机分配，保证 AI 学会所有专家动作的多样性
            self.env_expert_idx[env_ids] = torch.randint(0, self.num_experts, (len(env_ids),), device=self.device)

        
        exp_ids = self.env_expert_idx[env_ids]

        # -------------------- 3) 重置到安全初值（兜底） --------------------
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        hand_root_indices = self.hand_indices[env_ids]
        self.root_state_tensor[hand_root_indices, 0:7] = self.hand_init_state[env_ids, 0:7].clone()
        self.root_state_tensor[hand_root_indices, 7:13] = 0.0
# -------------------- 4) & 5) 覆盖：手掌空间位姿 & 手指关节 --------------------
        c_obj_pos = self.root_state_tensor[self.object_indices[env_ids], 0:3]

        if self.use_expert_reset:
            # 【课程学习/测试模式】：直接瞬移到专家相对位姿和关节角度
            self.root_state_tensor[hand_root_indices, 0:3] = c_obj_pos - self.expert_rel_pos_pool[exp_ids]
            self.root_state_tensor[hand_root_indices, 3:7] = self.expert_quat_pool[exp_ids]
            
            remap_q = self.expert_qpos_pool[exp_ids] 
            self.allegro_hand_dof_pos[env_ids, :] = remap_q
            self.allegro_hand_dof_vel[env_ids, :] = 0.0
            self.prev_targets[env_ids, :16] = remap_q
            self.cur_targets[env_ids, :16] = remap_q
        else:
            # 【完整抓取模式】（默认）：从球的正上方 15cm 处，张开手开始下降！
            approach_offset = torch.tensor([0.0, 0.0, 0.15], device=self.device)
            self.root_state_tensor[hand_root_indices, 0:3] = c_obj_pos + approach_offset
            # 保持最初定义好的“掌心垂直朝下”的初始姿态
            self.root_state_tensor[hand_root_indices, 3:7] = self.hand_init_state[env_ids, 3:7].clone()
            
            # 手指强制恢复到张开姿态 (open_qpos 已经在 __init__ 里洗过牌了)
            open_q = self.open_qpos.unsqueeze(0).repeat(len(env_ids), 1)
            self.allegro_hand_dof_pos[env_ids, :] = open_q
            self.allegro_hand_dof_vel[env_ids, :] = 0.0 # 清空速度，防止弹射
            self.prev_targets[env_ids, :16] = open_q
            self.cur_targets[env_ids, :16] = open_q

        # -------------------- 6) 一次性写入仿真器 --------------------
        all_indices = torch.cat([self.object_indices[env_ids], hand_root_indices]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )

        hand_indices_int32 = hand_root_indices.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices_int32),
            len(env_ids),
        )

        # -------------------- 7) 终极验血打印 (仅 Env 0) --------------------
        if False:
            self.gym.refresh_dof_state_tensor(self.sim)
            eidx = int(self.env_expert_idx[0].item())
            
            exp_written = remap_q[(env_ids == 0).nonzero(as_tuple=False)[0, 0]].detach().cpu().numpy()
            sim_read = self.allegro_hand_dof_pos[0].detach().cpu().numpy()
            max_err = float(np.max(np.abs(exp_written - sim_read)))

            import os
            print("\n" + "=" * 80)
            print(f"[DEBUG RESET] 当前锁定专家: ID {eidx} -> {os.path.basename(self.expert_files[eidx])}")
            print(f"最大关节写入误差 max_abs_err: {max_err:.6f}")
            
            # 追加向量方向验血
            ball_pos = self.root_state_tensor[self.object_indices[0], 0:3]
            hand_pos = self.root_state_tensor[self.hand_indices[0], 0:3]
            print(f"当前仿真里 (球 - 手): {(ball_pos - hand_pos).cpu().numpy()}")
            print(f"专家数据里 rel_pos: {self.expert_rel_pos_pool[eidx].cpu().numpy()}")
            print("=" * 80 + "\n")

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

        
    def compute_observations(self):
        
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        # -----------------------------------------------------------------------
        # 空间观测（追加到关节 lag history 之后）
        # -----------------------------------------------------------------------
        palm_pos_obs = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        rel_pos_obs = self.object_pos - palm_pos_obs
       # �� 修改：乘以一个缩放系数（比如 5.0 或 10.0）
        # 假设工作空间半径在 0.2m 左右，乘以 5 就能将其映射到 [-1, 1] 附近
        self.obs_buf[:, 96:99] = palm_pos_obs * 5.0 
        self.obs_buf[:, 99:102] = rel_pos_obs * 5.0

        # 【修正】已经把报错的 self.diffusion_targets 清理掉了

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
    def compute_reward(self, actions):
        # ==========================================
        # 1. 刷新并提取物理状态
        # ==========================================
        
        palm_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        
        # 一次性提取四个指尖的坐标 [num_envs, 4, 3]
        tips_idx = [self.ff_idx, self.mf_idx, self.rf_idx, self.th_idx]
        tips_pos = self.rigid_body_states[:, tips_idx, 0:3]
        
        # 计算每个指尖到球心的距离 [num_envs, 4]
        # self.object_pos.unsqueeze(1) 是为了对齐维度进行广播计算
        d_tips = torch.norm(tips_pos - self.object_pos.unsqueeze(1), p=2, dim=-1)
        mean_d = d_tips.mean(dim=-1)

        # ==========================================
        # 2. 动态专家对齐 (核心逻辑)
        # ==========================================
        # 根据 reset_idx 中分配的索引，提取每个环境对应的专家数据
        expert_qpos = self.expert_qpos_pool[self.env_expert_idx]
        expert_rel_pos = self.expert_rel_pos_pool[self.env_expert_idx]
        expert_ball_r = self.expert_ball_radius_pool[self.env_expert_idx] # �� 新增：动态读取该专家的球体半径

        # �� 修复 1：动态调整收紧比例 (Grip Phase)
        # 废除写死的 0.08 和 0.055。
        # 开始收缩：球半径 + 4.5cm。 完全握紧：球半径 + 1.5cm。
        start_grip_dist = expert_ball_r + 0.045
        full_grip_dist = expert_ball_r + 0.015
        
        # 将阈值扩维以匹配 mean_d
        grip_phase = torch.clamp((start_grip_dist - mean_d) / (start_grip_dist - full_grip_dist), 0.0, 1.0).unsqueeze(-1)
        
        # 插值计算当前的目标关节姿态
        # 没碰到球前模仿 open_qpos，碰到过程中平滑过渡到 expert_qpos
        open_qpos_batch = self.open_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        qpos_target = (1.0 - grip_phase) * open_qpos_batch + grip_phase * expert_qpos

        # 计算误差项
        delta_qpos = self.allegro_hand_dof_pos - qpos_target
        
        # �� 修复 2：彻底解决 XY 错位
        # 使用实锤的减法顺序：(当前球坐标 - 当前手坐标) 减去 (专家记录的相对位置)
        delta_target_hand_pos = (self.object_pos - palm_pos) - expert_rel_pos
        base_actions = actions[:, 0:3] 
        
        # �� 修复 3：动态接触阈值 (在这里放宽接触容差！)
        # 废除写死的 0.06。使用：球半径 + 1.5cm(指尖物理厚度) + 1.5cm(临时放宽的容差)
        # 相当于给球套了一个更大的虚拟光环，让 AI 先学会“包围”
        contact_threshold = (expert_ball_r + 0.030).unsqueeze(-1)
        contact_count = (d_tips < contact_threshold).float().sum(dim=-1)

        # ==========================================
        # 3. 调用底层 JIT 算分引擎 (把算好的 contact_count 传进去)
        # ==========================================
        self.rew_buf[:], self.reset_buf[:], r_dist, r_imit, r_act, r_up = compute_hand_reward(
            self.object_init_state[:, 2], self.reset_buf, self.progress_buf, self.max_episode_length,
            self.object_pos, palm_pos,
            self.reset_z_threshold,
            delta_qpos, delta_target_hand_pos, base_actions, contact_count
        )

        # 4. 记录核心数据供 TensorBoard 观察
        self.extras['hand_dist'] = torch.norm(palm_pos - self.object_pos, p=2, dim=-1).mean()
        
        # ��【新增】把分离的奖励丢进 extras 字典，PPO算法会自动把它们画进 TensorBoard
        self.extras['Reward_Breakdown/1_Distance_Penalty'] = r_dist.mean()
        self.extras['Reward_Breakdown/2_Imitation_Penalty'] = r_imit.mean()
        self.extras['Reward_Breakdown/3_Action_Penalty'] = r_act.mean()
        self.extras['Reward_Breakdown/4_Lift_Reward'] = r_up.mean()
        
        
        # ==========================================
        # �� 新增：核心物理状态 Debug 监控面板
        # ==========================================
        self.extras['Debug/1_palm_z'] = palm_pos[:, 2].mean()               # 手掌当前的平均高度
        self.extras['Debug/2_obj_z'] = self.object_pos[:, 2].mean()         # 球当前的平均高度
        self.extras['Debug/3_contact_count'] = contact_count.mean()         # 平均有几根手指碰到球
        
        # 顺便把内部 JIT 算出来的门控状态也拿出来看 (需要稍微算一下)
        is_touching_log = ((contact_count >= 2.0) & (self.extras['hand_dist'] < 0.12)).float().mean() # 注意这里用放宽后的 0.12
        is_grasping_log = ((contact_count >= 3.0) & (self.extras['hand_dist'] < 0.09)).float().mean()
        self.extras['Debug/4_is_touching_rate'] = is_touching_log           # 触发 Touching 的环境比例
        self.extras['Debug/5_is_grasping_rate'] = is_grasping_log           # 触发 Grasping 的环境比例
        # ==========================================
        # 真正成功的定义：球被抬起超过 5cm，且至少有 3 根手指抓稳
        lift_success = ((self.object_pos[:, 2] - self.object_init_state[:, 2]) > 0.05) & (contact_count >= 3.0)
        self.extras['lift_success'] = lift_success.float().mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'success_rate: {self.extras["lift_success"]:.4f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()


    def post_physics_step(self):

        self.progress_buf += 1

        self.reset_buf[:] = 0

        self._refresh_gym()

        self.compute_reward(self.actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:

            self.reset_idx(env_ids)

        self.compute_observations()
        # 只有在开了相册模式并且有画面时，才去每帧检测按键
        if self.viewer and self.enable_expert_album:
            self.check_debug_keys()



        if self.viewer and self.debug_viz:

            # draw axes on target object

            self.gym.clear_lines(self.viewer)

            self.gym.refresh_rigid_body_state_tensor(self.sim)



            for i in range(self.num_envs):

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()

                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()

                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()



                p0 = self.object_pos[i].cpu().numpy()

                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])

                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])

                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        # =========================================================
        # ��️ 补丁 C：终极拦截器 (化解 Dummy Step 的空动作)
        # =========================================================
        assert actions.shape[1] == 19, f"Expected 19-dim actions [base(3)+finger(16)], got {actions.shape}"

        self.actions = actions.clone().to(self.device)

        # ==========================================
        # 1. 拦截与分流 (Action Splitting)
        # ==========================================
        # 把 19 维动作拆开：前 3 维控制 Base 移动，后 16 维控制手指关节
        base_actions = self.actions[:, 0:3]
        finger_actions = self.actions[:, 3:19]

        # ==========================================
        # 2. 控制手指关节 (保留 HORA 原有逻辑)
        # ==========================================
        targets = self.prev_targets + 1 / 24 * finger_actions
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits,
                                           self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()

        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        # ==========================================
        # 3. 控制手掌底座移动（速度控制 + root state 积分 + workspace clamp + EMA 平滑）
        # ==========================================
        # 将 [-1,1] 的动作缩放为速度 (m/s)
        vel_cmd = base_actions * self.base_max_speed
        
        # EMA 平滑
        smoothed_vel = self.base_ema_alpha * self.base_vel_prev + (1.0 - self.base_ema_alpha) * vel_cmd
        self.base_vel_prev[:] = smoothed_vel

        # 读取当前手掌位置并积分
        hand_root_indices = self.hand_indices  # shape: (num_envs,)
        cur_pos = self.root_state_tensor[hand_root_indices, 0:3].clone()
        
        # 使用完整的 RL 时间步长 (control_freq_inv * dt) 进行积分
        rl_dt = self.dt * self.control_freq_inv
        new_pos = cur_pos + smoothed_vel * rl_dt

        # workspace clamp (边界限制)
        new_pos[:, 0] = torch.clamp(new_pos[:, 0], self.base_ws_x[0], self.base_ws_x[1])
        new_pos[:, 1] = torch.clamp(new_pos[:, 1], self.base_ws_y[0], self.base_ws_y[1])
        new_pos[:, 2] = torch.clamp(new_pos[:, 2], self.base_ws_z[0], self.base_ws_z[1])

        # 写回 root state tensor (更新位置，清零线速度)
        self.root_state_tensor[hand_root_indices, 0:3] = new_pos
        self.root_state_tensor[hand_root_indices, 7:10] = 0.0  
       # ==========================================
        # �� 优化 2：基于距离的四元数平滑插值 (防突变翻转)
        # ==========================================
        expert_quat_batch = self.expert_quat_pool[self.env_expert_idx]
        
        # ⚠️ 注意：hand_init_state 是 env 级别的张量，直接用 [:, 3:7] 提取即可，切勿使用全局 hand_root_indices！
        init_quat_batch = self.hand_init_state[:, 3:7] 
        
        # 1. 四元数同向化 (Hemisphere Check) 极其关键！防止手腕突然 360 度大风车
        dot_product = torch.sum(init_quat_batch * expert_quat_batch, dim=-1, keepdim=True)
        expert_quat_batch = torch.where(dot_product < 0, -expert_quat_batch, expert_quat_batch)
        
        # 2. 计算当前手到球的距离 (✅ 修复 2：使用更新后的 new_pos 消除 1 帧延迟)
        dist = torch.norm(self.object_pos - new_pos, dim=-1, keepdim=True)
        
        # 3. 距离 > 15cm 时 rot_weight=0 (保持朝下)；距离 < 5cm 时 rot_weight=1 (对齐专家)
        rot_weight = torch.clamp((0.15 - dist) / 0.10, 0.0, 1.0)
        
        # 4. 线性插值并归一化
        blended_quat = (1.0 - rot_weight) * init_quat_batch + rot_weight * expert_quat_batch
        blended_quat = blended_quat / torch.norm(blended_quat, dim=-1, keepdim=True)
        
        self.root_state_tensor[hand_root_indices, 3:7] = blended_quat 
        self.root_state_tensor[hand_root_indices, 10:13] = 0.0  # 角速度归零，保持稳定
        
        # ==========================================
        # ✅ 修复 3：将运动学 Base 的状态真正推送到仿真器底层！
        # ==========================================
        hand_root_indices_int32 = hand_root_indices.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(hand_root_indices_int32),
            len(hand_root_indices_int32)
        )
# ==========================================
        # �� 优化 2：基于距离的四元数平滑插值 (防突变翻转)
        # ==========================================
        expert_quat_batch = self.expert_quat_pool[self.env_expert_idx]          # 形状: (num_envs, 4)
        
        # ⚠️ 保持 [:, 3:7]，因为 hand_init_state 的长度是 num_envs，天然对齐 expert_quat_batch
        init_quat_batch   = self.hand_init_state[:, 3:7]                        # 形状: (num_envs, 4)

        # 1. Hemisphere Check：保证四元数插值走最短弧，防止手腕 360 度大风车
        dot_product = torch.sum(init_quat_batch * expert_quat_batch, dim=-1, keepdim=True)
        expert_quat_batch = torch.where(dot_product < 0, -expert_quat_batch, expert_quat_batch)

        # 2. ✅ 使用积分后的 new_pos 算距离，消除 1 帧延迟
        dist = torch.norm(self.object_pos - new_pos, dim=-1, keepdim=True)

        # 3. 距离 > 15cm 时 rot_weight=0 (保持朝下)；距离 < 5cm 时 rot_weight=1 (对齐专家)
        rot_weight = torch.clamp((0.15 - dist) / 0.10, 0.0, 1.0)

        # 4. 线性插值 (Lerp) 并归一化
        blended_quat = (1.0 - rot_weight) * init_quat_batch + rot_weight * expert_quat_batch
        blended_quat = blended_quat / torch.norm(blended_quat, dim=-1, keepdim=True)

        self.root_state_tensor[hand_root_indices, 3:7] = blended_quat
        self.root_state_tensor[hand_root_indices, 10:13] = 0.0

        # ==========================================
        # �� 修复 3：将运动学 Base 的状态真正推送到仿真器底层！
        # ==========================================
        hand_root_indices_int32 = hand_root_indices.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(hand_root_indices_int32),
            len(hand_root_indices_int32)
        )
        # ==========================================
        # 4. 保留原有的随机扰动力逻辑 (Domain Randomization)
        # ==========================================
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale

            # 对球施加扰动力
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.ENV_SPACE)
            


    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.num_env_factors = self.config['env']['hora']['privInfoDim']
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

        # 覆盖 VecTask 中按 num_obs//3 分配的 lag history，固定为 32 维
        # (16 关节位置 + 16 关节目标)，与 numObservations 解耦，
        # 使 obs_buf 剩余维度可用于追加空间观测（手掌位置、球相对位置）
        # 80 = VecTask._allocate_buffers 中的 lag history 容量（最近 80 步）
        self.obs_buf_lag_history = torch.zeros(
            (num_envs, 80, 32), device=self.device, dtype=torch.float
        )

    def _setup_reward_config(self, r_config):
        # 保留方法以与其他 _setup_* 方法保持一致的初始化模式
        # 奖励阈值已内置在 compute_hand_reward JIT 函数中:
        #   approach_rew: hand_dist < 0.3 时线性正奖励（稠密接近奖励）
        #   flag gate: finger_dist <= 0.6, hand_dist <= 0.12
        #   lift_z = object_init_z + 0.1 + 0.003（目标抬升高度 ≈ 0.143m）
        #   target_pos.z = object_init_z + 0.15（目标位置高度 ≈ 0.19m）
        pass

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        
        # �� 核心修改：解开底座，并加上巨大的空气阻尼防止太空乱飞
        hand_asset_options.fix_base_link = False
        hand_asset_options.linear_damping = 10.0
        hand_asset_options.angular_damping = 10.0 # ⚠️ 修复：下方的 0.01 错误覆盖已被移除
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)

        # load object asset
        self.object_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        
        # 1. 高度修正：把手的底座放在桌面上方 19 厘米处
        allegro_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.19) 
        
        # 2. 【真正朝下】：绕 X 轴旋转 180 度 (np.pi)，让手完全倒立，掌心对准地面！
        allegro_hand_start_pose.r = gymapi.Quat(0, 0, 0, 1)

        object_start_pose = gymapi.Transform()
        
        # 3. 球老老实实呆在桌面上 (Z=0.04m)
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.04) 
        
        return allegro_hand_start_pose, object_start_pose


@torch.jit.script
def compute_hand_reward(
        object_init_z: torch.Tensor, reset_buf: torch.Tensor, progress_buf: torch.Tensor, max_episode_length: float,
        object_pos: torch.Tensor, palm_pos: torch.Tensor,
        reset_z_threshold: float,
        delta_qpos: torch.Tensor, delta_target_hand_pos: torch.Tensor, base_actions: torch.Tensor,
        contact_count: torch.Tensor 
):
    # ==========================================
    # 1. 基础惩罚：鼓励靠近 + 动作平滑
    # ==========================================
    hand_dist = torch.norm(object_pos - palm_pos, p=2, dim=-1)
    
    # 针对 kinematic base 仅惩罚动作，不惩罚物理速度
    action_penalty = 0.05 * torch.sum(base_actions ** 2, dim=-1)

    # ==========================================
    # 2. 状态门控 (State Gating) - �� 临时放宽距离限制
    # ==========================================
    # 把 hand_dist 的限制从 0.08 放宽到 0.12，从 0.065 放宽到 0.09
    is_touching = ((contact_count >= 2.0) & (hand_dist < 0.12)).float()
    is_grasping = ((contact_count >= 3.0) & (hand_dist < 0.09)).float()

    # ==========================================
    # 3. 模仿奖励 (Imitation Penalty)
    # ==========================================
    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)

    # �� 优化 3：位置权重的平滑渐变 (Curriculum)
    # 扩大过渡带：从 15cm 开始慢慢教，到 5cm 才要求对齐
    w_near = torch.clamp((0.15 - hand_dist) / 0.10, 0.0, 1.0)
    
    # 降低近距离的绝对惩罚压制：最大权重降到 0.4
    pos_weight = (0.05 * (1.0 - w_near) + 0.4 * w_near) * (1.0 - 0.5 * is_touching)
    qpos_weight = is_touching * 0.05 + is_grasping * 0.05
    
    imitation_penalty = pos_weight * delta_hand_pos_value + qpos_weight * delta_qpos_value

    # ==========================================
    # 4. 阶梯式抬起奖励 (防拍飞门槛)
    # ==========================================
    lowest = object_pos[:, 2]
    lift_z = object_init_z + 0.05 # 提高到 5cm，确保抓取稳固后再给大分
    lift_val = lowest - object_init_z
    
    # 分段奖励：包络(>=3指)给 10+，轻触(2指)给 2+
    lift_reward = torch.where(
        is_grasping > 0.5, 
        10.0 + 20.0 * lift_val, 
        torch.where(is_touching > 0.5, 2.0 + 5.0 * lift_val, torch.zeros_like(lowest))
    )
    # 必须超过 5cm 门槛才激活
    hand_up = torch.where(lowest >= lift_z, lift_reward, torch.zeros_like(lowest))

    # ==========================================
    # 5. 汇总
    # ==========================================
    reward = -1.0 * hand_dist - imitation_penalty - action_penalty + hand_up

    # ==========================================
    # 6. 重置逻辑 (Resets)
    # ==========================================
    palm_dist_origin = torch.norm(palm_pos, p=2, dim=-1)
    
    resets = torch.where(lowest < reset_z_threshold, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(palm_dist_origin > 1.0, torch.ones_like(resets), resets) 

    return reward, resets, (-1.0 * hand_dist), (-imitation_penalty), (-action_penalty), hand_up

def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles