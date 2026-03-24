import pandas as pd
import ta
import numpy as np

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates required indicators: ATR, RSI, and Log Returns
    """
    if df.empty or len(df) < 15:
        return df

    # Calculate Log Return (OHLCV)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate ATR (Average True Range)
    # length=14 is standard for ATR
    indicator_atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = indicator_atr.average_true_range()

    # Calculate RSI (Relative Strength Index)
    indicator_rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
    df['rsi'] = indicator_rsi.rsi()

    # Calculate 5 MA and 10 MA for the new technical strategy
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
        
    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    return df
