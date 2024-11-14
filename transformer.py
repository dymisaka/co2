import torch
import torch.nn as nn

# 定义 Transformer 模型类
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=2, num_layers=1, output_dim=1):
        super(TransformerRegressor, self).__init__()
        self.input_emb = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_emb(src)
        src = src.unsqueeze(1)  # 增加序列长度维度
        transformer_output = self.transformer_encoder(src)
        transformer_output = transformer_output.squeeze(1)  # 移除序列长度维度
        output = self.output_layer(transformer_output)
        return output

def load_model(input_dim,model_path='transformer_regressor_model.pth'):
    """
    加载已训练的模型权重并返回模型。
    
    Args:
        input_dim (int): 输入特征的维度。
        
    Returns:
        TransformerRegressor: 加载好的模型。
    """
    model = TransformerRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_CO2_emissions_transformer(input_data):
    """
    使用 Transformer 模型进行预测。
    
    Args:
        input_data (torch.Tensor): 已标准化的输入特征张量。
        
    Returns:
        float: 预测的 CO₂ 排放量。
    """
    # 加载模型
    transformer_regressor_model = load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_model.pth')
    transformer_regressor_fold_1 =load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_fold_1.pth')
    transformer_regressor_fold_2 =load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_fold_2.pth')
    transformer_regressor_fold_3 =load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_fold_3.pth')
    transformer_regressor_fold_4 =load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_fold_4.pth')
    transformer_regressor_fold_5 =load_model(input_dim=input_data.shape[1],model_path='transformer_regressor_fold_5.pth')
    # 进行预测
    with torch.no_grad():
        prediction = transformer_regressor_model(input_data)
        prediction_fold_1 = transformer_regressor_fold_1(input_data)
        prediction_fold_2 = transformer_regressor_fold_2(input_data)
        prediction_fold_3 = transformer_regressor_fold_3(input_data)
        prediction_fold_4 = transformer_regressor_fold_4(input_data)
        prediction_fold_5 = transformer_regressor_fold_5(input_data)
        return [prediction.item(),prediction_fold_1.item(),prediction_fold_2.item(),prediction_fold_3.item(),prediction_fold_4.item(),prediction_fold_5.item()]
