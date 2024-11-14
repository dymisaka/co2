import torch
import torch.nn as nn
from data import preprocess_input  

# 定义 ANN 模型类
class ANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ANNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_model(input_dim, model_path='ann_model_full.pth'):
    """
    加载已训练的 ANN 模型权重并返回模型。
    
    Args:
        input_dim (int): 输入特征的维度。
        model_path (str): 模型文件的路径。
        
    Returns:
        ANNRegressor: 加载好的 ANN 模型。
    """
    model = ANNRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()
    return model

def predict_CO2_emissions_ann(input_data):
    """
    使用 ANN 模型进行 CO₂ 排放量预测。
    
    Args:
        input_data (torch.Tensor): 已标准化的输入特征张量。
        
    Returns:
        list: 各折模型预测的 CO₂ 排放量。
    """
    # 加载不同折的模型
    ann_model = load_model(input_dim=input_data.shape[1], model_path='ann_model_state_dict.pth')
    ann_fold_1 = load_model(input_dim=input_data.shape[1], model_path='ann_model_fold_1.pth')
    ann_fold_2 = load_model(input_dim=input_data.shape[1], model_path='ann_model_fold_2.pth')
    ann_fold_3 = load_model(input_dim=input_data.shape[1], model_path='ann_model_fold_3.pth')
    ann_fold_4 = load_model(input_dim=input_data.shape[1], model_path='ann_model_fold_4.pth')
    # ann_fold_5 = load_model(input_dim=input_data.shape[1], model_path='ann_model_fold_5.pth')
    
    # 进行预测
    with torch.no_grad():
        prediction = ann_model(input_data)
        prediction_fold_1 = ann_fold_1(input_data)
        prediction_fold_2 = ann_fold_2(input_data)
        prediction_fold_3 = ann_fold_3(input_data)
        prediction_fold_4 = ann_fold_4(input_data)
        # prediction_fold_5 = ann_fold_5(input_data)
        
        return [
            prediction.item()/10000,
            prediction_fold_1.item()/10000,
            prediction_fold_2.item()/10000,
            prediction_fold_3.item()/10000,
            prediction_fold_4.item()/10000
            # prediction_fold_5.item()
        ]

## 测试
if __name__ == "__main__":
    # 预处理输入数据
    X_scaled = preprocess_input([189, 32, 8, 32839, 57970, 57970, 1241, 13])
        
    # 调用 ANN 模型的预测函数
    prediction_value_list = predict_CO2_emissions_ann(X_scaled)
    print(f"predicted_CO2_emissions: {prediction_value_list}")
