import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 定义简单的 MLP 扩散网络
class DiffusionMLP(nn.Module):
    def __init__(self, state_dim=25, cond_dim=1, hidden_dim=256):
        super().__init__()
        # 这是一个简单的网络，输入是：噪声 + 时间步 + 环境条件（半径）
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
        # 拼接所有输入：x 是噪声/中间状态, cond 是半径, t 是当前时间步
        t = t.view(-1, 1) / 100.0  # 时间步归一化
        inputs = torch.cat([x, cond, t], dim=-1)
        return self.network(inputs)

# 2. 训练逻辑
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载第一步生成的 .pt 文件
    data = torch.load('diffusion_input_data.pt')
    conditions = data['condition'].to(device)
    targets = data['target'].to(device)
    
    dataset = TensorDataset(conditions, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型初始化
    model = DiffusionMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 扩散步数
    T = 100 
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print("开始炼丹 (Training Diffusion Model)...")
    model.train()
    
    for epoch in range(2000): # 循环次数可以根据损失调整
        epoch_loss = 0
        for cond_batch, target_batch in loader:
            batch_size = cond_batch.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, T, (batch_size,), device=device).long()
            
            # 生成噪声并加噪
            noise = torch.randn_like(target_batch)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            x_noisy = torch.sqrt(alpha_t) * target_batch + torch.sqrt(1 - alpha_t) * noise
            
            # 预测噪声并计算损失
            pred_noise = model(x_noisy, cond_batch, t)
            loss = nn.MSELoss()(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.6f}")

    # 保存权重
    torch.save(model.state_dict(), 'grasp_diffusion_model.pth')
    print("✅ 权重已保存为: grasp_diffusion_model.pth")

if __name__ == "__main__":
    train()