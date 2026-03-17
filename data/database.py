import asyncpg
import pandas as pd
from typing import List, Dict, Any, Optional

class TimescaleDB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn=self.dsn)

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    async def create_tables(self):
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        query = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION
        );
        -- Convert to hypertable if not already
        SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
        
        -- Create unique index
        CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_time_symbol_idx ON ohlcv (time, symbol);
        """
        async with self.pool.acquire() as connection:
            await connection.execute(query)

    async def insert_data(self, df: pd.DataFrame, symbol: str):
        if not self.pool:
            raise Exception("Database pool not initialized")
        if df.empty:
            return

        # Prepare records for insertion
        records = []
        for index, row in df.iterrows():
            records.append((
                index.to_pydatetime(),
                symbol,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ))

        query = """
        INSERT INTO ohlcv (time, symbol, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (time, symbol) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """
        async with self.pool.acquire() as connection:
            await connection.executemany(query, records)

    async def fetch_data(self, symbol: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        query = "SELECT * FROM ohlcv WHERE symbol = $1"
        args: List[Any] = [symbol]
        
        if start_time:
            query += f" AND time >= ${len(args) + 1}"
            args.append(start_time)
            
        if end_time:
            query += f" AND time <= ${len(args) + 1}"
            args.append(end_time)
            
        query += " ORDER BY time ASC"

        async with self.pool.acquire() as connection:
            records = await connection.fetch(query, *args)
            
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('time', inplace=True)
        return df
