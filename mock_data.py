import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent dir to path
sys.path.append(os.getcwd())

from data.database import TimescaleDB

def generate_ohlcv(symbol, days=30):
    start_price = 60000 if "BTC" in symbol else (3500 if "ETH" in symbol else 150)
    volatility = 0.005 if "BTC" in symbol else 0.008
    
    rows = days * 24
    end_date = datetime.now()
    dates = [end_date - timedelta(hours=i) for i in range(rows)]
    dates.reverse()
    
    data = []
    current_price = start_price
    for date in dates:
        change = current_price * np.random.normal(0, volatility)
        open_p = current_price
        close_p = open_p + change
        high_p = max(open_p, close_p) + (abs(change) * 0.2)
        low_p = min(open_p, close_p) - (abs(change) * 0.2)
        volume = np.random.uniform(10, 100)
        
        data.append([date, open_p, high_p, low_p, close_p, volume])
        current_price = close_p
        
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('timestamp', inplace=True)
    return df

async def main():
    dsn = "postgresql://postgres:postgres@db:5432/crypto"
    db = TimescaleDB(dsn)
    await db.connect()
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for symbol in symbols:
        print(f"Generating mock data for {symbol}...")
        df = generate_ohlcv(symbol)
        await db.insert_data(df, symbol)
        print(f"Inserted {len(df)} mock rows for {symbol}")
            
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
