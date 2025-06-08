
# import numpy as np
# import pandas as pd
# from flask import Flask, jsonify, request
# from tensorflow.keras.models import load_model
# import pickle
# import datetime
# import os
# from tiingo import TiingoClient
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# # Tiingo API config
# config = {
#     'api_key': os.getenv('TIINGO_API_KEY', '88f4bd6b0b4ecae4d2bf1a44b437de363c621d8f'),
#     'session': True
# }
# client = TiingoClient(config)

# # Dynamically load model and scaler for a symbol
# def load_model_and_scaler(symbol):
#     folder_path = os.path.join("model", symbol)

#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"Folder not found for symbol '{symbol}'")

#     # Automatically find model (.h5) and scaler (.pkl)
#     model_file = next((f for f in os.listdir(folder_path) if f.endswith(".h5")), None)
#     scaler_file = next((f for f in os.listdir(folder_path) if f.endswith(".pkl")), None)

#     if not model_file or not scaler_file:
#         raise FileNotFoundError(f"Model or scaler file missing in '{folder_path}'")

#     model_path = os.path.join(folder_path, model_file)
#     scaler_path = os.path.join(folder_path, scaler_file)

#     model = load_model(model_path)
    
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)

#     return model, scaler

# # Fetch stock data
# def fetch_stock_data(symbol, start_date='1987-01-01', end_date=datetime.datetime.today().strftime('%Y-%m-%d')):
#     df = client.get_dataframe(symbol, frequency='daily', startDate=start_date, endDate=end_date)
#     return df['close'].values

# # Preprocess data
# def preprocess_data(data, scaler, time_step=100):
#     data = scaler.transform(np.array(data).reshape(-1, 1))
#     X = [data[i:i + time_step, 0] for i in range(len(data) - time_step - 1)]
#     X = np.array(X).reshape(len(X), time_step, 1)
#     return X

# # Prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict_stock_price():
#     try:
#         data = request.get_json()
#         symbol = data.get('symbol', 'AAPL').upper()
#         num_days = data.get('num_days', 30)

#         model, scaler = load_model_and_scaler(symbol)
#         raw_data = fetch_stock_data(symbol)
#         X = preprocess_data(raw_data, scaler)

#         temp_input = X[-1].flatten().tolist()
#         predictions = []

#         for _ in range(num_days):
#             x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
#             yhat = model.predict(x_input, verbose=0)
#             predictions.append(float(scaler.inverse_transform(yhat.reshape(-1, 1))[0][0]))
#             temp_input.extend(yhat.flatten().tolist())

#         predicted_dates = pd.date_range(start=datetime.datetime.today(), periods=num_days, freq='D')
#         result = {str(predicted_dates[i].date()): predictions[i] for i in range(num_days)}

#         return jsonify(result), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import datetime
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import yfinance as yf

app = Flask(__name__)

CORS(app, origins=[
    'http://localhost:3000',
    'https://tradexai.netlify.app/'  # Replace with actual URL
])

# Supported symbols
SUPPORTED_SYMBOLS = {"AAPL", "GOOG", "MSFT", "TSLA"}

# Helper to fetch historical stock data
def fetch_stock_data(symbol, period="2y"):
    df = yf.download(symbol, period=period)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df['Close'].values.reshape(-1, 1)

# Preprocess data for LSTM input
def preprocess_data(data, scaler):
    scaled_data = scaler.transform(data)
    sequence_length = 100
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
    return np.array(X)

# Dynamically load model and scaler
def load_model_and_scaler(symbol):
    folder_path = os.path.join("model", symbol)

    # Fallback to AAPL if folder doesn't exist
    if not os.path.exists(folder_path):
        symbol = "AAPL"
        folder_path = os.path.join("model", symbol)

    model_file = next((f for f in os.listdir(folder_path) if f.endswith(".h5")), None)
    scaler_file = next((f for f in os.listdir(folder_path) if f.endswith(".pkl")), None)

    if not model_file or not scaler_file:
        raise FileNotFoundError(f"Model or scaler missing in {folder_path}")

    model = load_model(os.path.join(folder_path, model_file))
    with open(os.path.join(folder_path, scaler_file), 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_stock_price():
    try:
        data = request.get_json()
        input_symbol = data.get('symbol', 'AAPL').upper()
        num_days = int(data.get('num_days', 30))

        # Use fallback if unsupported
        symbol = input_symbol if input_symbol in SUPPORTED_SYMBOLS else "AAPL"

        model, scaler = load_model_and_scaler(symbol)
        raw_data = fetch_stock_data(input_symbol)  # Always fetch real data of requested symbol
        X = preprocess_data(raw_data, scaler)

        temp_input = X[-1].flatten().tolist()
        predictions = []

        for _ in range(num_days):
            x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
            yhat = model.predict(x_input, verbose=0)
            predicted_price = float(scaler.inverse_transform(yhat.reshape(-1, 1))[0][0])
            predictions.append(predicted_price)
            temp_input.extend(yhat.flatten().tolist())

        predicted_dates = pd.date_range(start=datetime.datetime.today(), periods=num_days, freq='D')
        result = {str(predicted_dates[i].date()): predictions[i] for i in range(num_days)}

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
