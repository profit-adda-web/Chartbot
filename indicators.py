import pandas as pd
import numpy as np

# ------------------- Moving Averages -------------------
def MA(df, n):
    """Moving Average - returns Series"""
    return df["close"].rolling(window=n).mean()

def EMA(df, n):
    """Exponential Moving Average - returns Series"""
    return df["close"].ewm(span=n, adjust=False).mean()

# ------------------- Momentum Indicators -------------------
def MOM(df, n):
    """Momentum - returns Series"""
    return df["close"].diff(n)

def ROC(df, n):
    """Rate of Change - returns Series"""
    return df["close"].pct_change(n)

def TRIX(df, n):
    """Trix - returns Series"""
    ex1 = df["close"].ewm(span=n, adjust=False).mean()
    ex2 = ex1.ewm(span=n, adjust=False).mean()
    ex3 = ex2.ewm(span=n, adjust=False).mean()
    return ex3.pct_change()

# ------------------- Volatility Indicators -------------------
def ATR(df, n=14):
    """Average True Range - returns Series"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    return tr.rolling(n).mean()

def BBANDS(df, n=20):
    """Bollinger Bands - returns DataFrame with upper, middle, lower bands"""
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return pd.DataFrame({'upper': upper, 'middle': ma, 'lower': lower})

def KELCH(df, n=20):
    """Keltner Channel - returns DataFrame with middle, upper, lower channels"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    middle = typical.rolling(n).mean()
    upper = ((4*df["high"] - 2*df["low"] + df["close"]) / 3).rolling(n).mean()
    lower = ((-2*df["high"] + 4*df["low"] + df["close"]) / 3).rolling(n).mean()
    return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

def DONCH(df, n=20):
    """Donchian Channel - returns Series"""
    return df["high"].rolling(n).max() - df["low"].rolling(n).min()

def STDDEV(df, n=20):
    """Standard Deviation - returns Series"""
    return df["close"].rolling(n).std()

# ------------------- Pivot Points -------------------
def PPSR(df):
    """Pivot Points, Supports and Resistances - returns DataFrame"""
    pp = (df["high"] + df["low"] + df["close"]) / 3
    result = pd.DataFrame({
        "PP": pp,
        "R1": 2*pp - df["low"],
        "S1": 2*pp - df["high"],
        "R2": pp + (df["high"] - df["low"]),
        "S2": pp - (df["high"] - df["low"]),
        "R3": df["high"] + 2*(pp - df["low"]),
        "S3": df["low"] - 2*(df["high"] - pp)
    })
    return result

# ------------------- Stochastic Oscillator -------------------
def STOK(df):
    """Fast Stochastic %K - returns Series"""
    return (df["close"] - df["low"]) / (df["high"] - df["low"])

def STO_EMA(df, nK=14, nD=3, nS=1):
    """Stochastic Oscillator with EMA smoothing - returns DataFrame with %K and %D"""
    lowest_low = df["low"].rolling(nK).min()
    highest_high = df["high"].rolling(nK).max()

    SOk = (df["close"] - lowest_low) / (highest_high - lowest_low)
    
    if nS > 1:
        SOk = SOk.ewm(span=nS, adjust=False).mean()
    
    SOd = SOk.ewm(span=nD, adjust=False).mean()

    return pd.DataFrame({f"SO%k_EMA{nK}": SOk, f"SO%d_EMA{nD}": SOd})

def STO_SMA(df, nK=14, nD=3, nS=1):
    """Stochastic Oscillator with SMA smoothing - returns DataFrame with %K and %D"""
    lowest_low = df["low"].rolling(nK).min()
    highest_high = df["high"].rolling(nK).max()

    SOk = (df["close"] - lowest_low) / (highest_high - lowest_low)
    
    if nS > 1:
        SOk = SOk.rolling(nS).mean()
    
    SOd = SOk.rolling(nD).mean()

    return pd.DataFrame({f"SO%k_SMA{nK}": SOk, f"SO%d_SMA{nD}": SOd})

# ------------------- Trend Indicators -------------------
def ADX(df, n=14, n_adx=14):
    """Average Directional Movement Index - returns Series"""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(n_adx).mean()

def MACD(df, n_fast=12, n_slow=26, n_signal=9):
    """MACD - returns DataFrame with MACD, signal, and histogram"""
    ema_fast = df["close"].ewm(span=n_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=n_signal, adjust=False).mean()
    histogram = macd - signal
    
    return pd.DataFrame({
        f"MACD_{n_fast}_{n_slow}": macd,
        f"MACDsign_{n_fast}_{n_slow}": signal,
        f"MACDdiff_{n_fast}_{n_slow}": histogram
    })

def MassI(df):
    """Mass Index - returns Series"""
    rng = df["high"] - df["low"]
    ex1 = rng.ewm(span=9, adjust=False).mean()
    ex2 = ex1.ewm(span=9, adjust=False).mean()
    mass = ex1 / ex2
    return mass.rolling(25).sum()

def Vortex(df, n=14):
    """Vortex Indicator - returns DataFrame with VIP and VIN"""
    tr = (df["high"].combine(df["close"].shift(), max) -
          df["low"].combine(df["close"].shift(), min))
    vm_plus = (df["high"] - df["low"].shift()).abs()
    vm_minus = (df["low"] - df["high"].shift()).abs()
    vip = vm_plus.rolling(n).sum() / tr.rolling(n).sum()
    vin = vm_minus.rolling(n).sum() / tr.rolling(n).sum()
    return pd.DataFrame({f"Vortex+_{n}": vip, f"Vortex-_{n}": vin})

def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    """KST Oscillator - returns Series"""
    roc1 = df["close"].pct_change(r1)
    roc2 = df["close"].pct_change(r2)
    roc3 = df["close"].pct_change(r3)
    roc4 = df["close"].pct_change(r4)
    kst = (roc1.rolling(n1).sum() +
           2*roc2.rolling(n2).sum() +
           3*roc3.rolling(n3).sum() +
           4*roc4.rolling(n4).sum())
    return kst

def RSI(df, n=14):
    """Relative Strength Index - returns Series"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=n, adjust=False).mean()
    avg_loss = loss.ewm(span=n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def TSI(df, r=25, s=13):
    """True Strength Index - returns Series"""
    m = df["close"].diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    aema1 = m.abs().ewm(span=r, adjust=False).mean()
    aema2 = aema1.ewm(span=s, adjust=False).mean()
    return 100 * ema2 / aema2

# ------------------- Volume Indicators -------------------
def ACCDIST(df, n=14):
    """Accumulation/Distribution - returns Series"""
    ad = ((2*df["close"] - df["high"] - df["low"]) /
          (df["high"] - df["low"])) * df["volume"]
    return ad.pct_change(n)

def Chaikin(df):
    """Chaikin Oscillator - returns Series"""
    ad = ((2*df["close"] - df["high"] - df["low"]) /
          (df["high"] - df["low"])) * df["volume"]
    return ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()

def MFI(df, n=14):
    """Money Flow Index - returns Series"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp > tp.shift(), 0)
    neg_mf = mf.where(tp < tp.shift(), 0)
    mr = pos_mf.rolling(n).sum() / neg_mf.rolling(n).sum()
    return 100 - (100 / (1 + mr))

def OBV(df):
    """On-balance Volume - returns Series"""
    return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

def FORCE(df, n=1):
    """Force Index - returns Series"""
    return df["close"].diff(n) * df["volume"].diff(n)

def EOM(df, n=14):
    """Ease of Movement - returns Series"""
    em = (df["high"].diff() + df["low"].diff()) * (df["high"] - df["low"]) / (2*df["volume"])
    return em.rolling(n).mean()

# ------------------- Other Indicators -------------------
def CCI(df, n=20):
    """Commodity Channel Index - returns Series"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - ma) / (0.015 * md)

def COPP(df, n=14):
    """Coppock Curve - returns Series"""
    roc1 = df["close"].pct_change(int(n*11/10))
    roc2 = df["close"].pct_change(int(n*14/10))
    return (roc1 + roc2).ewm(span=n, adjust=False).mean()

def ULTOSC(df):
    """Ultimate Oscillator - returns Series"""
    bp = df["close"] - df[["low", "close"]].shift().min(axis=1)
    tr = df[["high", "close"]].shift().max(axis=1) - df[["low", "close"]].shift().min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return 100 * (4*avg7 + 2*avg14 + avg28) / 7
