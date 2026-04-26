import torch
import torch.nn as nn
import numpy as np

# 必须和训练时的结构一模一样
class DiffusionMLP(nn.Module):
    def __init__(self, state_dim=25, cond_dim=1, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x, cond, t):
        t = t.view(-1, 1).float() / 100.0
        inputs = torch.cat([x, cond, t], dim=-1)
        return self.network(inputs)

def test_inference(test_radius=0.033):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = DiffusionMLP().to(device)
    if not torch.os.path.exists('grasp_diffusion_model.pth'):
        print("❌ 找不到权重文件！请先运行 step2")
        return
    model.load_state_dict(torch.load('grasp_diffusion_model.pth'))
    model.eval()

    # 2. 设置推理参数
    T = 100
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 3. 准备输入：给一个半径条件
    cond = torch.tensor([[test_radius]], device=device).float()
    # 从纯噪声开始 [1, 25]
    x = torch.randn((1, 25), device=device)

    # 4. 逐步去噪过程 (Reverse Diffusion)
    print(f"正在为半径 {test_radius} 生成抓取姿态...")
    with torch.no_grad():
        for i in reversed(range(T)):
            t = torch.tensor([i], device=device).long()
            pred_noise = model(x, cond, t)
            
            # DDPM 采样公式
            alpha_t = alphas[i]
            alpha_t_cum = alphas_cumprod[i]
            beta_t = betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cum) * pred_noise) + torch.sqrt(beta_t) * noise

    # 5. 解读结果
    res = x.cpu().numpy()[0]
    print("\n--- �� Diffusion 生成结果 ---")
    print(f"球位置 (Ball XYZ): {res[0]:.3f}, {res[1]:.3f}, {res[2]:.3f}")
    print(f"手掌位置 (Palm XYZ): {res[3]:.3f}, {res[4]:.3f}, {res[5]:.3f}")
    print(f"手掌欧拉角 (Palm RPY): {res[6]:.3f}, {res[7]:.3f}, {res[8]:.3f}")
    print(f"手指关节 (J0-J15): \n{res[9:13]}\n{res[13:17]}\n{res[17:21]}\n{res[21:25]}")

if __name__ == "__main__":
    test_inference(0.033) # 你可以换成其他半径试试
