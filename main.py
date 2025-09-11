import pandas as pd
import mplfinance as mpf
import techindilib   # technical indicator library

#input_csv_path
symbol = "BTCUSD"
interval = "15m"
to_date="20250910"
from_date="20250909"

csv_path=f"CSV/{symbol}_{from_date}_{to_date}_{interval}.csv"
df=pd.read_csv(csv_path)
 #Convert timestamp column to datetime
df["time"] = pd.to_datetime(df["time"])
# Set as index
df.set_index("time", inplace=True)
print(df)

# --- Apply indicator(s) ---
df = techindilib.MA(df, 20)   # Add 20-period MA
df = techindilib.MA(df, 50)   # Add 50-period MA

# --- Plot ---
apds = [
    mpf.make_addplot(df["MA_20"], color="blue"),
    mpf.make_addplot(df["MA_50"], color="red"),
]

mpf.plot(
    df,
    type="candle",
    style="charles",
    volume=False,
    addplot=apds,
    title=f"{symbol} {interval} Candlestick Chart"
)

