from fastapi import FastAPI, Query, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import io
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import marketfeed  
import indicators as indi 
from sklearn.linear_model import LinearRegression
import numpy as np
import os, base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def create_chart_image(df: pd.DataFrame, view: str = "candles") -> io.BytesIO:
    """Create an mplfinance PNG in-memory and return BytesIO.
    view options:
      - candles           : plain candlestick with volume
      - ma_rsi            : candlestick + MA(3,6,9) and RSI in separate panel
      - lr_ma             : candlestick + MA features and linear-regression predicted close line
    """
    # Ensure time index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]) 
        df.set_index("time", inplace=True)

    # compute indicators
    df['MA_3'] = indi.MA(df, 3)
    df['MA_6'] = indi.MA(df, 6)
    df['MA_9'] = indi.MA(df, 9)
    df['RSI'] = indi.RSI(df, 14)

    add_plots = []

    # conditional addplots depending on `view`
    if view == "ma_rsi":
        add_plots = [
            mpf.make_addplot(df['MA_3'],color='darkorange',label='MA 3'),
            mpf.make_addplot(df['MA_6'],color='green',label='MA 6'),
            mpf.make_addplot(df['MA_9'],color='blue',label='MA 9'),
            mpf.make_addplot(df['RSI'], panel=2, width=2.0,color='purple', ylabel="RSI"),
        ]
    elif view == "lr_ma":
        # Build feature matrix using available MAs (drop NaNs)
        feat_cols = ['MA_3', 'MA_9']
        trnd_df = df.dropna(subset=feat_cols + ['close']).copy()
        if len(trnd_df) >= 10:
            X = trnd_df[feat_cols].values
            y = trnd_df['close'].values
            model = LinearRegression()
            model.fit(X, y)
            # Predict for whole dataset where MAs are available
            preds = np.full(len(df), np.nan)
            mask = ~df[feat_cols].isnull().any(axis=1)
            preds[mask] = model.predict(df.loc[mask, feat_cols].values)
            df['LR_PRED'] = preds
            add_plots = [
                mpf.make_addplot(df['MA_3'],color='darkorange'),
                mpf.make_addplot(df['MA_9'],color='blue'),
                mpf.make_addplot(df['LR_PRED'], width=2.0,color='purple'),
            ]
        else:
            # fallback to MA-only if not enough data to train
            add_plots = [
            mpf.make_addplot(df['MA_3'],color='darkorange'),
            mpf.make_addplot(df['MA_6'],color='green'),
            mpf.make_addplot(df['MA_9'],color='blue'),
            ]
    else:
        add_plots = []

    # style
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        facecolor="black",
        figcolor="black",
        edgecolor="black",
        gridcolor="white",
        rc={
            "axes.labelcolor": "darkorange",
            "xtick.color": "darkorange",
            "ytick.color": "darkorange",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    buf = io.BytesIO()

    kwargs = {}
    if len(add_plots) > 0:
        kwargs["addplot"] = add_plots  


    mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        figsize=(19.2, 10.8),
        title="Candlestick Chart",
        savefig=dict(fname=buf, format="png"),
        **kwargs
    )
    buf.seek(0)
    return buf


@app.get("/chart")
def chart_endpoint(
    request: Request,
    symbol: str = Query("BTCUSD"),
    interval: str = Query("15m"),
    days: int = Query(1),
    view: str = Query("candles")
):

    csv_path, dframe = marketfeed.delta_fetch_data(symbol=symbol, interval=interval, days=days)
    df = pd.read_csv(csv_path)
    # Keep a copy of original index column if present
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    buf = create_chart_image(df, view=view)
    
    # Return HTML page with the chart image and controls
    return templates.TemplateResponse("chart.html", {
        "request": request,
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "view": view,
        "chart_image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    })


@app.post("/update_chart")
async def update_chart(
    symbol: str = Form("BTCUSD"),
    interval: str = Form("15m"),
    days: int = Form(1),
    view: str = Form("candles")
):
    """Endpoint to update chart with new parameters"""
    csv_path, dframe = marketfeed.delta_fetch_data(symbol=symbol, interval=interval, days=days)
    df = pd.read_csv(csv_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    buf = create_chart_image(df, view=view)
    return {"image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}


@app.get("/", response_class=HTMLResponse)
def home():
    # HTML contains a small MCQ (radio buttons) table; JS will build the /chart URL
    return """
    <html>
      <head>
        <title>Chartbot — One Click Links</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="font-family: Arial; background: #111; color: #ddd;">
        <div style="max-width:900px; margin:40px auto;">
          <h2 style="text-align:center; color:lightgreen;">Chartbot — One Click Links</h2>

          <table style="width:100%; border-collapse:collapse; margin-bottom:18px;">
            <tr style="background:#222;"><th style="padding:10px; text-align:left;">Field</th><th style="padding:10px; text-align:left;">Value / Options</th></tr>
            <tr><td style="padding:8px;">Symbol</td><td style="padding:8px;"><input id="symbol" value="BTCUSD"/></td></tr>
            <tr><td style="padding:8px;">Interval</td><td style="padding:8px;"><input id="interval" value="15m"/></td></tr>
            <tr><td style="padding:8px;">Days</td><td style="padding:8px;"><input id="days" type="number" value="1" min="1"/></td></tr>
          </table>

          <h3 style="color:#fff;">Choose Chart Option:</h3>
          <table style="width:100%; border-collapse:collapse; background:#222; padding:8px;">
            <tr><th style="padding:8px; text-align:left;">Option</th><th style="padding:8px; text-align:left;">Description</th></tr>
            <tr><td style="padding:8px;"><input type="radio" name="view" value="candles" checked> Candlestick Chart</td><td style="padding:8px;">Candlestick with Volume</td></tr>
            <tr><td style="padding:8px;"><input type="radio" name="view" value="ma_rsi"> MA + RSI</td><td style="padding:8px;">MA(3,6,9) and RSI panel</td></tr>
            <tr><td style="padding:8px;"><input type="radio" name="view" value="lr_ma"> Linear Regression (MA features)</td><td style="padding:8px;">Train LR on MA(3,9) Features and Overlay Predicted Close</td></tr>
          </table>

          <p style="text-align:center; margin-top:14px;">
            <button id="openChart" style="padding:10px 16px; font-size:16px;">Open Chart in new tab</button>
          </p>

        </div>

        <script>
          document.getElementById('openChart').addEventListener('click', function(){
            const symbol = encodeURIComponent(document.getElementById('symbol').value || 'BTCUSD');
            const interval = encodeURIComponent(document.getElementById('interval').value || '15m');
            const days = encodeURIComponent(document.getElementById('days').value || '1');
            const view = document.querySelector('input[name="view"]:checked').value;
            const url = `/chart?symbol=${symbol}&interval=${interval}&days=${days}&view=${view}`;
            window.open(url, '_blank');
          });
        </script>
      </body>
    </html>
    """




# If you run this file with: uvicorn dashboard:app --reload
# Visit http://127.0.0.1:8000/ to use the interface.
