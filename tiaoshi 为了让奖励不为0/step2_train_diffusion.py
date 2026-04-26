import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 定义简单的 MLP 扩散网络
class DiffusionMLP(nn.Module):
    def __init__(self, state_dim=25, cond_dim=1, hidden_dim=256):
        super().__init__()
        # 输入：噪声(25) + 时间步(1) + 半径条件(1) = 27维
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
        # 归一化时间步
        t = t.view(-1, 1).float() / 100.0
        inputs = torch.cat([x, cond, t], dim=-1)
        return self.network(inputs)

# 2. 训练逻辑
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")
    
    if not os.path.exists('diffusion_input_data.pt'):
        print("❌ 错误：找不到 diffusion_input_data.pt，请先运行 step1")
        return

    # 加载数据
    data = torch.load('diffusion_input_data.pt')
    conditions = data['condition'].to(device)
    targets = data['target'].to(device)
    
    dataset = TensorDataset(conditions, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DiffusionMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    T = 100 
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print("开始训练 (Training Diffusion Model)...")
    model.train()
    
    for epoch in range(2000):
        epoch_loss = 0
        for cond_batch, target_batch in loader:
            batch_size = cond_batch.shape[0]
            t = torch.randint(0, T, (batch_size,), device=device).long()
            
            noise = torch.randn_like(target_batch)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            x_noisy = torch.sqrt(alpha_t) * target_batch + torch.sqrt(1 - alpha_t) * noise
            
            pred_noise = model(x_noisy, cond_batch, t)
            loss = nn.MSELoss()(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), 'grasp_diffusion_model.pth')
    print("✅ 权重已保存为: grasp_diffusion_model.pth")

if __name__ == "__main__":
    train()
