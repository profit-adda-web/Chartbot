from fastapi import FastAPI, Query, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import io
import pandas as pd
import mplfinance as mpf
import matplotlib
from datetime import datetime, timedelta
import marketfeed  
import indicators as indi 
from sklearn.linear_model import LinearRegression
import numpy as np
import os, base64
import warnings 
matplotlib.use('Agg')
warnings.simplefilter('ignore')

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

    # Always compute basic MAs for fallback
    df['MA_3'] = indi.MA(df, 3)
    df['MA_6'] = indi.MA(df, 6)
    df['MA_9'] = indi.MA(df, 9)
    df['RSI'] = indi.RSI(df, 14)

    add_plots = []
    # Plot logic for each indicator
    if view == "candles":
        add_plots = []
    elif view == "ma_rsi":
        add_plots = [
            mpf.make_addplot(df['MA_3'],color='darkorange',label='MA 3'),
            mpf.make_addplot(df['MA_6'],color='green',label='MA 6'),
            mpf.make_addplot(df['MA_9'],color='blue',label='MA 9'),
            mpf.make_addplot(df['RSI'], panel=2, width=2.0,color='purple', ylabel="RSI")
        ]
    elif view == "lr_ma":
        feat_cols = ['MA_3', 'MA_9']
        trnd_df = df.dropna(subset=feat_cols + ['close']).copy()
        if len(trnd_df) >= 10:
            X = trnd_df[feat_cols].values
            y = trnd_df['close'].values
            model = LinearRegression()
            model.fit(X, y)
            preds = np.full(len(df), np.nan)
            mask = ~df[feat_cols].isnull().any(axis=1)
            preds[mask] = model.predict(df.loc[mask, feat_cols].values)
            df['LR_PRED'] = preds
            add_plots = [
                mpf.make_addplot(df['MA_3'],color='darkorange'),
                mpf.make_addplot(df['MA_9'],color='blue'),
                mpf.make_addplot(df['LR_PRED'], width=2.0,color='purple')
            ]
        else:
            add_plots = [
                mpf.make_addplot(df['MA_3'],color='darkorange'),
                mpf.make_addplot(df['MA_6'],color='green'),
                mpf.make_addplot(df['MA_9'],color='blue')
            ]
    elif view == "MA":
        df['MA_14'] = indi.MA(df, 14)
        add_plots = [mpf.make_addplot(df['MA_14'], color='orange', width=2.0, label='MA 14')]
    elif view == "EMA":
        df['EMA_14'] = indi.EMA(df, 14)
        add_plots = [mpf.make_addplot(df['EMA_14'], color='cyan', width=2.0, label='EMA 14')]
    elif view == "MOM":
        df['MOM_10'] = indi.MOM(df, 10)
        add_plots = [mpf.make_addplot(df['MOM_10'], color='magenta', width=2.0, panel=2, label='Momentum')]
    elif view == "ROC":
        df['ROC_10'] = indi.ROC(df, 10)
        add_plots = [mpf.make_addplot(df['ROC_10'], color='yellow', width=2.0, panel=2, label='Rate of Change')]
    elif view == "TRIX":
        df['TRIX_15'] = indi.TRIX(df, 15)
        add_plots = [mpf.make_addplot(df['TRIX_15'], color='lime', width=2.0, panel=2, label='Trix')]
    elif view == "ATR":
        df['ATR_14'] = indi.ATR(df, 14)
        add_plots = [mpf.make_addplot(df['ATR_14'], color='red', width=2.0, panel=2, label='ATR')]
    elif view == "BBANDS":
        bands = indi.BBANDS(df, 20)
        add_plots = [
            mpf.make_addplot(bands['upper'], color='orange', width=1.5, label='BB Upper'),
            mpf.make_addplot(bands['middle'], color='white', width=1.5, label='BB Middle'),
            mpf.make_addplot(bands['lower'], color='orange', width=1.5, label='BB Lower')
        ]
    elif view == "KELCH":
        kelch = indi.KELCH(df, 20)
        add_plots = [
            mpf.make_addplot(kelch['upper'], color='green', width=1.5, label='Keltner Upper'),
            mpf.make_addplot(kelch['middle'], color='white', width=1.5, label='Keltner Middle'),
            mpf.make_addplot(kelch['lower'], color='green', width=1.5, label='Keltner Lower')
        ]
    elif view == "DONCH":
        df['DONCH_20'] = indi.DONCH(df, 20)
        add_plots = [mpf.make_addplot(df['DONCH_20'], color='blue', width=2.0, panel=2, label='Donchian Channel')]
    elif view == "STDDEV":
        df['STDDEV_20'] = indi.STDDEV(df, 20)
        add_plots = [mpf.make_addplot(df['STDDEV_20'], color='purple', width=2.0, panel=2, label='Std Dev')]
    elif view == "PPSR":
        ppsr = indi.PPSR(df)
        add_plots = [
            mpf.make_addplot(ppsr['PP'], color='white', width=1.5, label='Pivot'),
            mpf.make_addplot(ppsr['R1'], color='green', width=1.0, label='R1'),
            mpf.make_addplot(ppsr['S1'], color='red', width=1.0, label='S1')
        ]
    elif view == "STOK":
        df['STOK'] = indi.STOK(df)
        add_plots = [mpf.make_addplot(df['STOK'], color='orange', width=2.0, panel=2, label='Stochastic %K')]
    elif view == "STO_EMA":
        sto_ema = indi.STO_EMA(df)
        add_plots = [
            mpf.make_addplot(sto_ema.iloc[:,0], color='orange', width=2.0, panel=2, label='Stoch %K EMA'),
            mpf.make_addplot(sto_ema.iloc[:,1], color='purple', width=2.0, panel=2, label='Stoch %D EMA')
        ]
    elif view == "STO_SMA":
        sto_sma = indi.STO_SMA(df)
        add_plots = [
            mpf.make_addplot(sto_sma.iloc[:,0], color='orange', width=2.0, panel=2, label='Stoch %K SMA'),
            mpf.make_addplot(sto_sma.iloc[:,1], color='purple', width=2.0, panel=2, label='Stoch %D SMA')
        ]
    elif view == "ADX":
        df['ADX'] = indi.ADX(df)
        add_plots = [mpf.make_addplot(df['ADX'], color='yellow', width=2.0, panel=2, label='ADX')]
    elif view == "MACD":
        macd = indi.MACD(df)
        add_plots = [
            mpf.make_addplot(macd.iloc[:,0], color='orange', width=2.0, panel=2, label='MACD'),
            mpf.make_addplot(macd.iloc[:,1], color='purple', width=2.0, panel=2, label='Signal'),
            mpf.make_addplot(macd.iloc[:,2], color='green', width=2.0, panel=2, label='Histogram')
        ]
    elif view == "MassI":
        df['MassI'] = indi.MassI(df)
        add_plots = [mpf.make_addplot(df['MassI'], color='cyan', width=2.0, panel=2, label='Mass Index')]
    elif view == "Vortex":
        vortex = indi.Vortex(df)
        add_plots = [
            mpf.make_addplot(vortex.iloc[:,0], color='orange', width=2.0, panel=2, label='Vortex+'),
            mpf.make_addplot(vortex.iloc[:,1], color='purple', width=2.0, panel=2, label='Vortex-')
        ]
    elif view == "KST":
        df['KST'] = indi.KST(df, 10, 15, 20, 30, 10, 10, 10, 15)
        add_plots = [mpf.make_addplot(df['KST'], color='lime', width=2.0, panel=2, label='KST Oscillator')]
    elif view == "RSI":
        df['RSI_14'] = indi.RSI(df, 14)
        add_plots = [mpf.make_addplot(df['RSI_14'], color='purple', width=2.0, panel=2, label='RSI')]
    elif view == "TSI":
        df['TSI'] = indi.TSI(df)
        add_plots = [mpf.make_addplot(df['TSI'], color='orange', width=2.0, panel=2, label='TSI')]
    elif view == "ACCDIST":
        df['ACCDIST'] = indi.ACCDIST(df)
        add_plots = [mpf.make_addplot(df['ACCDIST'], color='green', width=2.0, panel=2, label='Accum/Dist')]
    elif view == "Chaikin":
        df['Chaikin'] = indi.Chaikin(df)
        add_plots = [mpf.make_addplot(df['Chaikin'], color='blue', width=2.0, panel=2, label='Chaikin Osc')]
    elif view == "MFI":
        df['MFI'] = indi.MFI(df, 14)
        add_plots = [mpf.make_addplot(df['MFI'], color='magenta', width=2.0, panel=2, label='MFI')]
    elif view == "OBV":
        df['OBV'] = indi.OBV(df)
        add_plots = [mpf.make_addplot(df['OBV'], color='yellow', width=2.0, panel=2, label='OBV')]
    elif view == "FORCE":
        df['FORCE'] = indi.FORCE(df)
        add_plots = [mpf.make_addplot(df['FORCE'], color='red', width=2.0, panel=2, label='Force Index')]
    elif view == "EOM":
        df['EOM'] = indi.EOM(df)
        add_plots = [mpf.make_addplot(df['EOM'], color='cyan', width=2.0, panel=2, label='Ease of Movement')]
    elif view == "CCI":
        df['CCI'] = indi.CCI(df, 20)
        add_plots = [mpf.make_addplot(df['CCI'], color='orange', width=2.0, panel=2, label='CCI')]
    elif view == "COPP":
        df['COPP'] = indi.COPP(df, 14)
        add_plots = [mpf.make_addplot(df['COPP'], color='purple', width=2.0, panel=2, label='Coppock Curve')]
    elif view == "ULTOSC":
        df['ULTOSC'] = indi.ULTOSC(df)
        add_plots = [mpf.make_addplot(df['ULTOSC'], color='lime', width=2.0, panel=2, label='Ultimate Oscillator')]
    # fallback: no extra plots
    # ...existing code...

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
        # xlim=(0, len(df) + 25),
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
        df.set_index("time", inplace=True)

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
    indicators = [
        ("candles", "Candlestick Chart", "Candlestick with Volume"),
        ("ma_rsi", "MA + RSI", "MA(3,6,9) and RSI"),
        ("lr_ma", "Linear Regression (MA features)", "Train LR on MA(3,9) Features and Overlay Predicted Close"),
        ("MA", "Moving Average", "Simple moving average of close price"),
        ("EMA", "Exponential Moving Average", "Exponentially weighted moving average of close price"),
        ("MOM", "Momentum", "Difference in close price over n periods"),
        ("ROC", "Rate of Change", "Percentage change in close price over n periods"),
        ("TRIX", "Trix", "Triple-smoothed exponential moving average"),
        ("ATR", "Average True Range", "Volatility indicator based on true range"),
        ("BBANDS", "Bollinger Bands", "Upper, middle, and lower bands around moving average"),
        ("KELCH", "Keltner Channel", "Volatility-based envelope set above and below EMA"),
        ("DONCH", "Donchian Channel", "Difference between highest high and lowest low over n periods"),
        ("STDDEV", "Standard Deviation", "Standard deviation of close price over n periods"),
        ("PPSR", "Pivot Points", "Pivot points, supports, and resistances"),
        ("STOK", "Fast Stochastic %K", "Fast stochastic oscillator %K"),
        ("STO_EMA", "Stochastic Oscillator EMA", "Stochastic oscillator with EMA smoothing"),
        ("STO_SMA", "Stochastic Oscillator SMA", "Stochastic oscillator with SMA smoothing"),
        ("ADX", "Average Directional Movement Index", "Trend strength indicator"),
        ("MACD", "MACD", "Moving Average Convergence Divergence"),
        ("MassI", "Mass Index", "Detects trend reversals by measuring range expansions"),
        ("Vortex", "Vortex Indicator", "Identifies trend reversals and confirmations"),
        ("KST", "KST Oscillator", "Know Sure Thing momentum oscillator"),
        ("RSI", "Relative Strength Index", "Momentum oscillator measuring speed and change of price movements"),
        ("TSI", "True Strength Index", "Momentum oscillator"),
        ("ACCDIST", "Accumulation/Distribution", "Volume-based indicator"),
        ("Chaikin", "Chaikin Oscillator", "Volume-based indicator"),
        ("MFI", "Money Flow Index", "Volume-weighted RSI"),
        ("OBV", "On-balance Volume", "Cumulative volume indicator"),
        ("FORCE", "Force Index", "Combines price and volume to identify strength"),
        ("EOM", "Ease of Movement", "Volume-based oscillator"),
        ("CCI", "Commodity Channel Index", "Deviation from average price"),
        ("COPP", "Coppock Curve", "Long-term momentum indicator"),
        ("ULTOSC", "Ultimate Oscillator", "Combines short, intermediate, and long-term price action"),
    ]
    
    # Build the radio buttons HTML correctly
    radio_html = """
    <table style='width:100%; border-collapse:collapse; background:#222; padding:8px;'>
      <tr><th style='padding:8px; text-align:left;'>Option</th><th style='padding:8px; text-align:left;'>Description</th></tr>
    """
    
    for value, label, desc in indicators:
        radio_html += f"<tr><td style='padding:8px;'><input type='radio' name='view' value='{value}' {'checked' if value=='candles' else ''}> <b>{label}</b></td><td style='padding:8px;'>{desc}</td></tr>"
    
    radio_html += "</table>"
    
    return f"""
    <html>
        <head>
            <title>Chartbot — One Click Links</title>
            <meta name='viewport' content='width=device-width, initial-scale=1' />
        </head>
        <body style='font-family: Arial; background: #181c20; color: #eee;'>
            <div style='max-width:900px; margin:40px auto; background:#23272b; border-radius:16px; box-shadow:0 2px 16px #0006; padding:32px;'>
                <h2 style='text-align:center; color:#7fffd4;'>Chartbot — One Click Links</h2>
                <table style='width:100%; border-collapse:collapse; margin-bottom:18px;'>
                    <tr style='background:#222;'><th style='padding:10px; text-align:left;'>Field</th><th style='padding:10px; text-align:left;'>Value / Options</th></tr>
                    <tr><td style='padding:8px;'>Symbol</td><td style='padding:8px;'><input id='symbol' value='BTCUSD'/></td></tr>
                    <tr><td style='padding:8px;'>Interval</td><td style='padding:8px;'><input id='interval' value='15m'/></td></tr>
                    <tr><td style='padding:8px;'>Days</td><td style='padding:8px;'><input id='days' type='number' value='1' min='1'/></td></tr>
                </table>
                <h3 style='color:#fff;'>Choose Indicator:</h3>
                {radio_html}
                <p style='text-align:center; margin-top:14px;'>
                    <button id='openChart' style='padding:10px 16px; font-size:16px; background:#7fffd4; color:#23272b; border:none; border-radius:8px;'>Open Chart in new tab</button>
                </p>
            </div>
            <script>
                document.getElementById('openChart').addEventListener('click', function(){{
                    const symbol = encodeURIComponent(document.getElementById('symbol').value || 'BTCUSD');
                    const interval = encodeURIComponent(document.getElementById('interval').value || '15m');
                    const days = encodeURIComponent(document.getElementById('days').value || '1');
                    const view = document.querySelector('input[name="view"]:checked').value;
                    const url = `/chart?symbol=${{symbol}}&interval=${{interval}}&days=${{days}}&view=${{view}}`;
                    window.open(url, '_blank');
                }});
            </script>
        </body>
    </html>
    """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# If you run this file with: uvicorn dashboard:app --reload
# Visit http://127.0.0.1:8000/ to use the interface.
