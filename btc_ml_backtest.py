import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import mplfinance as mpf

# --- Parameters ---
base_url = "https://api.india.delta.exchange"
url = f"{base_url}/v2/history/candles"
symbol = 'BTCUSD'
resolution = '15m'
lookback_days = 7  # more data for meaningful ML/MA

# --- Get Data ---
end = int(datetime.now().timestamp())
start = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
params = {
    'symbol': symbol,
    'resolution': resolution,
    'start': start,
    'end': end
}
r = requests.get(url, params=params)
if r.status_code != 200:
    raise Exception(f"API Error: {r.status_code} {r.text}")
data = r.json().get('result', [])
if not data:
    raise Exception("No data returned from API.")

df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume']).sort_values('time')
df['time'] = pd.to_datetime(df['time'], unit='s')
df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
df.set_index('time', inplace=True)
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# --- Feature Engineering: Moving Averages ---
df['sma10'] = df['close'].rolling(window=10).mean()
df['sma50'] = df['close'].rolling(window=50).mean()
df['return'] = df['close'].pct_change().shift(-1)  # Next period return

df = df.dropna()

# --- ML Prep ---
features = df[['sma10', 'sma50']]
target = np.where(df['return'] > 0, 1, -1)

split = int(0.7 * len(df))
X_train, X_test = features[:split], features[split:]
y_train, y_test = target[:split], target[split:]

# --- Train Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)
signals = model.predict(X_test)
df_test = df.iloc[split:].copy()
df_test['signal'] = np.where(signals > 0, 1, -1)

# --- Backtest ---
df_test['strategy_ret'] = df_test['return'] * df_test['signal']
df_test['equity_curve'] = (1 + df_test['strategy_ret']).cumprod()

# --- Plotting ---
apds = [
    mpf.make_addplot(df_test['sma10'], color='blue'),
    mpf.make_addplot(df_test['sma50'], color='orange'),
    mpf.make_addplot(df_test['equity_curve'], panel=1, color='green', ylabel='Equity')
]

mpf.plot(
    df_test,
    type='candle',
    addplot=apds,
    style='charles',
    title=f'{symbol} ML+MA Backtest',
    ylabel='Price',
    volume=True
)
