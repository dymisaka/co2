from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template
app = Flask(__name__)

# 模型类
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerRegressor, self).__init__()
        self.input_emb = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_emb(src)
        src = src.unsqueeze(1)  # 增加序列长度的维度
        transformer_output = self.transformer_encoder(src)
        transformer_output = transformer_output.squeeze(1)  # 移除序列长度的维度
        output = self.output_layer(transformer_output)
        return output

# 加载已训练模型
def load_model(input_dim, model_dim=128, num_heads=2, num_layers=1, output_dim=1):
    model = TransformerRegressor(input_dim, model_dim, num_heads, num_layers, output_dim)
    model.load_state_dict(torch.load('transformer_regressor_fold_1.pth'))
    # model.load_state_dict(torch.load('transformer_regressor_fold_1.pth'))
    model.eval()
    return model

# 数据标准化函数
def preprocess_input(data):
    # 数据标准化处理（假设前端输入的数据已为正确格式）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform([data])
    return torch.tensor(data_scaled, dtype=torch.float32)

@app.route('/')
def home():
    return render_template('index.html')  # Flask 会从 templates 文件夹中查找 index.html

# 预测函数
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()  # 获取前端传来的数据
        input_values = [
            input_data['length'],
            input_data['breadth'],
            input_data['draught'],
            input_data['gross_tonnage'],
            input_data['deadweight'],
            input_data['summer_deadweight'],
            input_data['Annual Time spent at sea [hours]'],
            input_data['avg_speed']
        ]
        print(input_data)
        # 数据预处理
        X_scaled = preprocess_input(input_values)
        
        # 加载模型
        model = load_model(input_dim=len(input_values))
        
        # 进行预测
        with torch.no_grad():
            prediction = model(X_scaled)
            prediction_value = prediction.item()  # 获取预测值
        print(f"predict: {prediction_value}")
        return jsonify({"predicted_CO2_emissions": prediction_value}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
