import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data import preprocess_input

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def load_model(input_dim, model_path='gcn_model.pth'):
    """
    加载已训练的 GCN 模型权重并返回模型。
    
    Args:
        input_dim (int): 输入特征的维度。
        model_path (str): 模型文件的路径。
        
    Returns:
        GCN: 加载好的 GCN 模型。
    """
    model = GCN(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_CO2_emissions_gcn(input_data):
    """
    使用 GCN 模型进行 CO₂ 排放量预测。
    
    Args:
        input_data (torch.Tensor): 已标准化的输入特征张量。
        
    Returns:
        list: 预测的 CO₂ 排放量。
    """
    # 创建一个简单的边索引，因为这是单个预测
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    # 加载模型
    gcn_model = load_model(input_dim=input_data.shape[1])
    
    # 进行预测
    with torch.no_grad():
        prediction = gcn_model(input_data, edge_index)
        # 返回预测值，除以10000以保持与其他模型一致的比例
        return [prediction.item()]

if __name__ == "__main__":
    # 预处理输入数据
    X_scaled = preprocess_input([189, 32, 8, 32839, 57970, 57970, 1241, 13])
        
    # 调用 GCN 模型的预测函数
    prediction_value = predict_CO2_emissions_gcn(X_scaled)
    print(f"predicted_CO2_emissions: {prediction_value}")
