import requests
import pandas as pd
from datetime import datetime, timedelta
import techindilib   # technical indicator library

# --- API fetch ---
base_url = "https://api.india.delta.exchange"
url = f"{base_url}/v2/history/candles"

symbol = "BTCUSD"
interval = "15m"
days=1
to_date=datetime.now()
from_date=datetime.now() - timedelta(days=days)
end = int(to_date.timestamp())
start = int(from_date.timestamp())

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
print(df)
csv_path=f"{symbol}_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}_{interval}.csv"
df.to_csv(f"CSV/{csv_path}")
print(f"{csv_path} Generated SuccessFully")

