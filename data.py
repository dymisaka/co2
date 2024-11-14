import torch
import numpy as np
# from sklearn.preprocessing import StandardScaler

# 初始化标准化器
# scaler = StandardScaler()

def preprocess_input(data):
    """
    预处理输入数据，将其转换为模型可以使用的张量。
    
    Args:
        data (list): 包含输入特征的列表。
        
    Returns:
        torch.Tensor: 标准化后的特征张量。
    """
    # data_scaled = scaler.fit_transform([data])
    input_array = np.array(data, dtype=np.float32)
    input_tensor = torch.from_numpy(input_array).float()
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor
