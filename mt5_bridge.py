#import MetaTrader5 as mt5
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime
from typing import List
import pandas as pd
import uvicorn

# ============= CONFIG =============
#API_KEY = "neura_bridge_token_2025"   # must match the key in Render .env
#app = FastAPI(title="Neura.AI MT5 Bridge")
import requests

def get_mt5_data():
    try:
        response = requests.get("http://localhost:5001/data")
        return response.json()
    except:
        return {"error": "Cannot connect to MT5 bridge"}


# ==================================

security = APIKeyHeader(name="X-API-KEY")

def verify_token(token: str = Depends(security)):
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@app.get("/connect", dependencies=[Depends(verify_token)])
def connect(login: int, password: str, server: str):
    if not mt5.initialize(login=login, password=password, server=server):
        raise HTTPException(status_code=500, detail=f"MT5 init failed: {mt5.last_error()}")
    return {"status": "connected"}


@app.get("/candles", dependencies=[Depends(verify_token)], response_model=List[Candle])
def get_candles(symbol: str = "XAUUSD", timeframe: str = "M15", count: int = 100):
    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
    if timeframe not in tf_map:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
    if rates is None:
        raise HTTPException(status_code=404, detail="No data returned from MT5")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    candles = [
        Candle(
            time=row["time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["real_volume"]),
        )
        for _, row in df.iterrows()
    ]
    return candles


@app.on_event("shutdown")
def shutdown():
    mt5.shutdown()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
