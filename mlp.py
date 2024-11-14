import torch
import torch.nn as nn
from data import preprocess_input

class MLP(nn.Module):
    def __init__(self, input_dim=8):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def load_model(model_path='mlp_model.pth'):
    """
    加载已训练的 MLP 模型权重并返回模型。
    
    Args:
        model_path (str): 模型文件的路径。
        
    Returns:
        MLP: 加载好的 MLP 模型。
    """
    model = MLP()
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

def predict_CO2_emissions_mlp(input_data):
    """
    使用 MLP 模型进行 CO₂ 排放量预测。
    
    Args:
        input_data (torch.Tensor): 已标准化的输入特征张量。
        
    Returns:
        list: 预测的 CO₂ 排放量。
    """
    # 转换为PyTorch张量
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.FloatTensor(input_data)
    
    # 确保输入数据形状正确
    if len(input_data.shape) == 1:
        input_data = input_data.unsqueeze(0)
    
    # 加载模型
    mlp_model = load_model()
    
    # 进行预测
    with torch.no_grad():
        prediction = mlp_model(input_data)
        return [prediction.item()/10000]

if __name__ == "__main__":
    # 测试预测功能
    X_scaled = preprocess_input([189, 32, 8, 32839, 57970, 57970, 1241, 13])
    prediction_value = predict_CO2_emissions_mlp(X_scaled)
    print(f"predicted_CO2_emissions: {prediction_value}")