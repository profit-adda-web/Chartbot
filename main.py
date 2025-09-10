import requests
import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf
import indilib   # 👈 import your indicator functions

# --- API fetch ---
base_url = "https://api.india.delta.exchange"
url = f"{base_url}/v2/history/candles"

symbol = "BTCUSD"
interval = "15m"
end = int(datetime.now().timestamp())
start = int((datetime.now() - timedelta(days=1)).timestamp())

params = {
    "symbol": symbol,
    "resolution": interval,
    "start": start,
    "end": end
}

r = requests.get(url, params=params)
data = r.json().get("result", [])

df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"]).sort_values("time")
df["time"] = pd.to_datetime(df["time"], unit="s")
df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
df.set_index("time", inplace=True)

# --- Apply indicator(s) ---
df = indilib.MA(df, 20)   # Add 20-period MA
df = indilib.MA(df, 50)   # Add 50-period MA

# --- Plot ---
apds = [
    mpf.make_addplot(df["MA_20"], color="blue"),
    mpf.make_addplot(df["MA_50"], color="red"),
]

mpf.plot(
    df,
    type="candle",
    style="charles",
    title=f"{symbol} {interval} Candlestick Chart",
    addplot=apds,
    volume=False
)
