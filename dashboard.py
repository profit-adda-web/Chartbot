# import libraries
import pandas as pd
import mplfinance as mpf
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
import indicators as indi
import io, requests, marketfeed, matplotlib
matplotlib.use('Agg')

app = FastAPI()

@app.get("/chart")
def get_chart(
    symbol: str = Query("BTCUSD"),
    interval: str = Query("15m"),
    days: int = Query(1)
):
    csv_path,dframe=marketfeed.delta_fetch_data(symbol=symbol, interval=interval, days=days)
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    # Set as index
    df.set_index("time", inplace=True)
    df['MA_3']=indi.MA(df, 3)
    df['MA_6']=indi.MA(df, 6)
    df['MA_9']=indi.MA(df, 9)
    df['RSI']=indi.RSI(df,14)
    df.to_csv(csv_path)
    
    buf = io.BytesIO()
    # prepare addplots
    add_plots = [
        mpf.make_addplot(df['MA_3'],color="darkorange"),
        mpf.make_addplot(df['MA_6'], color="green"),
        mpf.make_addplot(df['MA_9'], color="blue"),
        mpf.make_addplot(df['RSI'],panel=2, color="gold",width=2.0)
    ]


    # main chart
    style = mpf.make_mpf_style(
    base_mpf_style="charles",
    facecolor="black",   # axes background
    figcolor="black",    # outer figure background
    edgecolor="black",   # edge around the chart
    gridcolor="white" ,    # optional: grid color
        rc={
        "axes.labelcolor": "darkorange",   # axis labels
        "xtick.color": "darkorange",       # x-axis tick numbers
        "ytick.color": "darkorange" ,       # y-axis tick numbers (right side)
        "xtick.labelsize": 15,        # font size for x-axis numbers
        "ytick.labelsize": 15        # font size for y-axis numbers
    }
)
    mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        addplot=add_plots,
    figsize=(19.2, 10.8),  # 1920/100 = 19.2, 1080/100 = 10.8 (assuming 100 DPI)
    title=f"{symbol} {interval} Candlestick Chart",
    savefig=dict(fname=buf, format="png"))

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/")
def home():
    return {"msg": "Go to /chart?symbol=BTCUSD&interval=15m&days=1"}
