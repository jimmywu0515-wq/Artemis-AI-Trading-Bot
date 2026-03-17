import ccxt.async_support as ccxt
import pandas as pd
import asyncio
from typing import Optional

class BinanceDataFetcher:
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        self.symbol = symbol
        self.timeframe = timeframe

    async def fetch_historical_data(self, since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """Fetches historical OHLCV data from Binance asynchronously."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=since,
                limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
        finally:
            await self.exchange.close()
