import pandas as pd
import numpy as np

# ------------------- Moving Averages -------------------
#Moving Average 
def MA(df, n):
    df[f"MA_{n}"] = df["Close"].rolling(n).mean()
    return df
#Exponential Moving Average 
def EMA(df, n):
    df[f"EMA_{n}"] = df["Close"].ewm(span=n, adjust=False).mean()
    return df

# ------------------- Momentum Indicators -------------------
#Momentum  
def MOM(df, n):
    df[f"Momentum_{n}"] = df["Close"].diff(n)
    return df
#Rate of Change  
def ROC(df, n):
    df[f"ROC_{n}"] = df["Close"].pct_change(n)
    return df
#Trix  
def TRIX(df, n):
    ex1 = df["Close"].ewm(span=n, adjust=False).mean()
    ex2 = ex1.ewm(span=n, adjust=False).mean()
    ex3 = ex2.ewm(span=n, adjust=False).mean()
    df[f"Trix_{n}"] = ex3.pct_change()
    return df

# ------------------- Volatility Indicators -------------------
#Average True Range  
def ATR(df, n=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df[f"ATR_{n}"] = tr.rolling(n).mean()
    return df
#Bollinger Bands  
def BBANDS(df, n=20):
    ma = df["Close"].rolling(n).mean()
    std = df["Close"].rolling(n).std()
    df[f"BollingerB_{n}"] = 4 * std / ma
    df[f"Bollinger%b_{n}"] = (df["Close"] - ma + 2 * std) / (4 * std)
    return df
#Keltner Channel  
def KELCH(df, n=20):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df[f"KelChM_{n}"] = typical.rolling(n).mean()
    df[f"KelChU_{n}"] = ((4*df["High"] - 2*df["Low"] + df["Close"]) / 3).rolling(n).mean()
    df[f"KelChD_{n}"] = ((-2*df["High"] + 4*df["Low"] + df["Close"]) / 3).rolling(n).mean()
    return df
#Donchian Channel  
def DONCH(df, n=20):
    df[f"Donchian_{n}"] = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    return df
#Standard Deviation  
def STDDEV(df, n=20):
    df[f"STD_{n}"] = df["Close"].rolling(n).std()
    return df

# ------------------- Pivot Points -------------------
#Pivot Points, Supports and Resistances 
def PPSR(df):
    pp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["PP"] = pp
    df["R1"] = 2*pp - df["Low"]
    df["S1"] = 2*pp - df["High"]
    df["R2"] = pp + (df["High"] - df["Low"])
    df["S2"] = pp - (df["High"] - df["Low"])
    df["R3"] = df["High"] + 2*(pp - df["Low"])
    df["S3"] = df["Low"] - 2*(df["High"] - pp)
    return df

# ------------------- Stochastic Oscillator -------------------
# ------------------- Fast %K (no smoothing) -------------------
def STOK(df):
    """
    Fast Stochastic %K = (Close - Low) / (High - Low)
    Single-bar stochastic, no rolling window.
    """
    df["SO%k"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    return df


# ------------------- Stochastic Oscillator (EMA smoothing) -------------------
def STO_EMA(df, nK=14, nD=3, nS=1):
    """
    Stochastic Oscillator with EMA smoothing.
    nK = lookback period for %K
    nD = smoothing period for %D
    nS = slowing factor (1 = no extra smoothing)
    """
    lowest_low = df["Low"].rolling(nK).min()
    highest_high = df["High"].rolling(nK).max()

    SOk = (df["Close"] - lowest_low) / (highest_high - lowest_low)
    SOd = SOk.ewm(span=nD, adjust=False).mean()

    if nS > 1:
        SOk = SOk.ewm(span=nS, adjust=False).mean()
        SOd = SOd.ewm(span=nS, adjust=False).mean()

    df[f"SO%k_EMA{nK}"] = SOk
    df[f"SO%d_EMA{nD}"] = SOd
    return df


# ------------------- Stochastic Oscillator (SMA smoothing) -------------------
def STO_SMA(df, nK=14, nD=3, nS=1):
    """
    Stochastic Oscillator with SMA smoothing.
    nK = lookback period for %K
    nD = smoothing period for %D
    nS = slowing factor (1 = no extra smoothing)
    """
    lowest_low = df["Low"].rolling(nK).min()
    highest_high = df["High"].rolling(nK).max()

    SOk = (df["Close"] - lowest_low) / (highest_high - lowest_low)
    SOd = SOk.rolling(nD).mean()

    if nS > 1:
        SOk = SOk.rolling(nS).mean()
        SOd = SOd.rolling(nS).mean()

    df[f"SO%k_SMA{nK}"] = SOk
    df[f"SO%d_SMA{nD}"] = SOd
    return df


# ------------------- Trend Indicators -------------------
#Average Directional Movement Index  
def ADX(df, n=14, n_adx=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df[f"ADX_{n}_{n_adx}"] = dx.rolling(n_adx).mean()
    return df
#MACD, MACD Signal and MACD difference 
def MACD(df, n_fast=12, n_slow=26, n_signal=9):
    ema_fast = df["Close"].ewm(span=n_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=n_signal, adjust=False).mean()
    df[f"MACD_{n_fast}_{n_slow}"] = macd
    df[f"MACDsign_{n_fast}_{n_slow}"] = signal
    df[f"MACDdiff_{n_fast}_{n_slow}"] = macd - signal
    return df
#Mass Index  
def MassI(df):
    rng = df["High"] - df["Low"]
    ex1 = rng.ewm(span=9, adjust=False).mean()
    ex2 = ex1.ewm(span=9, adjust=False).mean()
    mass = ex1 / ex2
    df["Mass Index"] = mass.rolling(25).sum()
    return df
#Vortex Indicator
def Vortex(df, n=14):
    tr = (df["High"].combine(df["Close"].shift(), max) -
          df["Low"].combine(df["Close"].shift(), min))
    vm_plus = (df["High"] - df["Low"].shift()).abs()
    vm_minus = (df["Low"] - df["High"].shift()).abs()
    vip = vm_plus.rolling(n).sum() / tr.rolling(n).sum()
    vin = vm_minus.rolling(n).sum() / tr.rolling(n).sum()
    df[f"Vortex+_{n}"] = vip
    df[f"Vortex-_{n}"] = vin
    return df
#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    roc1 = df["Close"].pct_change(r1)
    roc2 = df["Close"].pct_change(r2)
    roc3 = df["Close"].pct_change(r3)
    roc4 = df["Close"].pct_change(r4)
    kst = (roc1.rolling(n1).sum() +
           2*roc2.rolling(n2).sum() +
           3*roc3.rolling(n3).sum() +
           4*roc4.rolling(n4).sum())
    df[f"KST_{r1}_{r2}_{r3}_{r4}"] = kst
    return df
#Relative Strength Index  
def RSI(df, n=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=n, adjust=False).mean()
    avg_loss = loss.ewm(span=n, adjust=False).mean()
    rs = avg_gain / avg_loss
    df[f"RSI_{n}"] = 100 - (100 / (1 + rs))
    return df
#True Strength Index  
def TSI(df, r=25, s=13):
    m = df["Close"].diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    aema1 = m.abs().ewm(span=r, adjust=False).mean()
    aema2 = aema1.ewm(span=s, adjust=False).mean()
    df[f"TSI_{r}_{s}"] = 100 * ema2 / aema2
    return df

# ------------------- Volume Indicators -------------------
#Accumulation/Distribution  
def ACCDIST(df, n=14):
    ad = ((2*df["Close"] - df["High"] - df["Low"]) /
          (df["High"] - df["Low"])) * df["Volume"]
    df[f"Acc/Dist_ROC_{n}"] = ad.pct_change(n)
    return df
#Chaikin Oscillator  
def Chaikin(df):
    ad = ((2*df["Close"] - df["High"] - df["Low"]) /
          (df["High"] - df["Low"])) * df["Volume"]
    df["Chaikin"] = ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()
    return df
#Money Flow Index and Ratio  
def MFI(df, n=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos_mf = mf.where(tp > tp.shift(), 0)
    neg_mf = mf.where(tp < tp.shift(), 0)
    mr = pos_mf.rolling(n).sum() / neg_mf.rolling(n).sum()
    df[f"MFI_{n}"] = 100 - (100 / (1 + mr))
    return df
#On-balance Volume
def OBV(df):
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    return df
#Force Index  
def FORCE(df, n=1):
    df[f"Force_{n}"] = df["Close"].diff(n) * df["Volume"].diff(n)
    return df
#Ease of Movement  
def EOM(df, n=14):
    em = (df["High"].diff() + df["Low"].diff()) * (df["High"] - df["Low"]) / (2*df["Volume"])
    df[f"EoM_{n}"] = em.rolling(n).mean()
    return df

# ------------------- Other Indicators -------------------
#Commodity Channel Index  
def CCI(df, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.fabs(x - x.mean()).mean())
    df[f"CCI_{n}"] = (tp - ma) / (0.015 * md)
    return df
#Coppock Curve  
def COPP(df, n=14):
    roc1 = df["Close"].pct_change(int(n*11/10))
    roc2 = df["Close"].pct_change(int(n*14/10))
    copp = (roc1 + roc2).ewm(span=n, adjust=False).mean()
    df[f"Copp_{n}"] = copp
    return df
#Ultimate Oscillator  
def ULTOSC(df):
    bp = df["Close"] - df[["Low", "Close"]].shift().min(axis=1)
    tr = df[["High", "Close"]].shift().max(axis=1) - df[["Low", "Close"]].shift().min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    df["Ultimate_Osc"] = 100 * (4*avg7 + 2*avg14 + avg28) / 7
    return df
