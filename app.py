from flask import Flask, request, jsonify, render_template
from ann import predict_CO2_emissions_ann
from data import preprocess_input
from transformer import predict_CO2_emissions_transformer
from gcn import predict_CO2_emissions_gcn
from mlp import predict_CO2_emissions_mlp


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # 返回前端 HTML 页

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取前端传来的 JSON 数据
        input_data = request.get_json()
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
        
        # 数据预处理
        print("get data: ")
        print(input_values)
        X_scaled = preprocess_input(input_values)
        
        # 调用 transformer.py 中的预测函数
        prediction_transformer = predict_CO2_emissions_transformer(X_scaled)
        prediction_ann=predict_CO2_emissions_ann(X_scaled)
        gcn_predictions = predict_CO2_emissions_gcn(X_scaled)
        mlp_predictions = predict_CO2_emissions_mlp(X_scaled)
        print(f"transformer predictions: {prediction_transformer}")
        print(f"ANN predictions: {prediction_ann}")
        print(f"GCN predictions: {gcn_predictions}")
        print(f"MLP predictions: {mlp_predictions}")
        return jsonify({"transformer_predictions": prediction_transformer,
            "ann_predictions": prediction_ann,
            "gcn_predictions": gcn_predictions,
            "mlp_predictions": mlp_predictions
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
