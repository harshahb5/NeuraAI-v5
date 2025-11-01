from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import requests

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
TRADE_HISTORY_PATH = "trade_history.csv"
API_KEY = "demo"  # You can get a free API key from financialmodelingprep.com

# ======================================
# Try importing MetaTrader5 (Optional)
# ======================================
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available, using API fallback mode")

# ==========================
# MT5 Connection Functions
# ==========================
def connect_mt5(account=None, password=None, server=None):
    if not MT5_AVAILABLE:
        return True, "Running in API mode (MT5 unavailable)"
    mt5.initialize()
    if not mt5.login(account, password=password, server=server):
        return False, mt5.last_error()
    return True, "Connected successfully"

# ==========================
# Live Data Fetch
# ==========================
def get_live_data(symbol="XAUUSD", timeframe="15min", limit=500):
    if MT5_AVAILABLE:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, limit)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
    else:
        # --- Fallback to FinancialModelingPrep API ---
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}?apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['date'])
        df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low'}, inplace=True)
        df = df[['time', 'open', 'high', 'low', 'close']].sort_values('time')

    # Calculate EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema15'] = df['close'].ewm(span=15, adjust=False).mean()
    return df.tail(limit)

# ==========================
# AI Model Functions
# ==========================
def train_model():
    if not os.path.exists(TRADE_HISTORY_PATH):
        return "No trade history found."

    df = pd.read_csv(TRADE_HISTORY_PATH)
    if len(df) < 10:
        return "Not enough data for training."

    X = df[['entry_point', 'exit_point', 'pnl_points']].values
    y = np.where(df['pnl_points'] > 0, 1, 0)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return "Model retrained successfully."

def predict_next_trade(df):
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    last_row = df.iloc[-1]
    X_new = np.array([[last_row['ema9'], last_row['ema15'], last_row['close']]])
    prediction = model.predict(X_new)
    return "BUY" if prediction[0] == 1 else "SELL"

# ==========================
# Trade History Management
# ==========================
def save_trade(symbol, direction, entry_time, exit_time, pnl_points, entry_point, exit_point):
    trade = {
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl_points": pnl_points,
        "entry_point": entry_point,
        "exit_point": exit_point
    }
    if os.path.exists(TRADE_HISTORY_PATH):
        df = pd.read_csv(TRADE_HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    else:
        df = pd.DataFrame([trade])
    df.to_csv(TRADE_HISTORY_PATH, index=False)

# ==========================
# Flask Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect', methods=['POST'])
def connect():
    data = request.json
    account = data.get('account')
    password = data.get('password')
    server = data.get('server')
    success, msg = connect_mt5(account, password, server)
    return jsonify({"success": success, "message": msg})

@app.route('/get_data', methods=['GET'])
def get_data():
    df = get_live_data()
    df_json = df.to_dict(orient='records')
    prediction = predict_next_trade(df)
    return jsonify({"data": df_json, "prediction": prediction})

@app.route('/save_trade', methods=['POST'])
def save_trade_route():
    data = request.json
    save_trade(
        data['symbol'], data['direction'], data['entry_time'],
        data['exit_time'], data['pnl_points'],
        data['entry_point'], data['exit_point']
    )
    return jsonify({"status": "Trade saved successfully."})

@app.route('/load_history', methods=['GET'])
def load_history():
    if not os.path.exists(TRADE_HISTORY_PATH):
        return jsonify([])
    df = pd.read_csv(TRADE_HISTORY_PATH)
    return jsonify(df.to_dict(orient='records'))

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    msg = train_model()
    return jsonify({"message": msg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
