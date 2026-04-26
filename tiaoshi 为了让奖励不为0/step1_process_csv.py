import pandas as pd
import torch
import os

def preprocess_data(csv_path, save_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}，请确保它在当前文件夹下！")
        # 列出当前目录下的文件，帮你排查
        print("当前目录下的文件有:", os.listdir('.'))
        return

    try:
        # 1. 加载数据
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取数据，共 {len(df)} 条记录")

        # 2. 提取输入条件 (Condition): 物体半径
        condition = torch.tensor(df['Obj_Radius'].values).float().reshape(-1, 1)

        # 3. 提取输出目标 (Target): 25维
        target_cols = [
            'Ball_X', 'Ball_Y', 'Ball_Z', 
            'Palm_X', 'Palm_Y', 'Palm_Z', 
            'Palm_Rx', 'Palm_Ry', 'Palm_Rz'
        ] + [f'J{i}' for i in range(16)]
        
        target_data = torch.tensor(df[target_cols].values).float()

        # 4. 保存
        processed_data = {
            'condition': condition,
            'target': target_data,
            'column_names': target_cols
        }
        
        torch.save(processed_data, save_path)
        print(f"�� 预处理完成！数据已保存至: {save_path}")
        print(f"数据预览: Condition {condition.shape}, Target {target_data.shape}")
    except Exception as e:
        print(f"❌ 处理出错: {e}")

if __name__ == "__main__":
    preprocess_data('allegro_mapped_dataset.csv', 'diffusion_input_data.pt')
