import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add root folder to sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.grid_trading_env import GridTradingEnv
from data.indicators import calculate_features
from data.database import TimescaleDB
import asyncio


async def load_data(dsn: str, symbol: str) -> pd.DataFrame:
    db = TimescaleDB(dsn)
    await db.connect()
    # Fetch all data for training
    df = await db.fetch_data(symbol)
    await db.disconnect()
    
    if not df.empty:
        df = calculate_features(df)
    return df

def train_agent(df: pd.DataFrame, model_path: str = "models/ppo_grid_bot"):
    if df.empty:
        print("No data available for training.")
        return

    # Initialize environment
    env = DummyVecEnv([lambda: GridTradingEnv(df)])

    # Initialize PPO Agent with typical MLP policy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1
    )

    print("Starting Training...")
    # Train for 1,000,000 timesteps as requested
    model.learn(total_timesteps=1_000_000)
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    # Example usage for manual testing
    # Requires a running database at this DSN
    dsn = "postgresql://postgres:postgres@localhost:5432/crypto"
    
    try:
        df = asyncio.run(load_data(dsn, "BTC/USDT"))
        if not df.empty:
           train_agent(df)
    except Exception as e:
        print(f"Failed to load data or train: {e}")
