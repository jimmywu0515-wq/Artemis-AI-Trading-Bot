import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn as linear_schedule

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_env.triple_barrier_env import TripleBarrierTradingEnv
from data.database import TimescaleDB
import asyncio
from datetime import datetime, timedelta

async def load_training_data(dsn, symbol):
    db = TimescaleDB(dsn)
    await db.connect()
    df = await db.fetch_data(symbol)
    await db.disconnect()
    return df

def make_env(df, seed=0):
    def _init():
        env = TripleBarrierTradingEnv(df)
        env = Monitor(env)
        return env
    return _init

def generate_ohlcv(symbol="BTC/USDT", days=100):
    start_price = 65000
    volatility = 0.005
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

def train_artemis_tb(df, model_path="models/artemis_tb_v1", total_timesteps=50000, n_envs=4):
    if df.empty:
        print("No data for training.")
        return

    # Parallel environments
    env_fns = [make_env(df, seed=i) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    # Eval environment
    eval_env = DummyVecEnv([make_env(df, seed=99)])

    # Callbacks
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=50, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./{model_path}_best/",
        log_path="./logs/",
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=5,
        callback_after_eval=stop_cb,
        verbose=1
    )

    # MLP Policy with Tanh activation
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.Tanh,
        ortho_init=True
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(5e-4, 1e-5, 1.0), # slightly higher LR for discrete
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03, # Higher entropy to encourage exploring discrete actions
        vf_coef=0.75,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        seed=42
    )

    print(f"Starting Artemis Triple-Barrier Training ({total_timesteps} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ Artemis Triple-Barrier saved to {model_path}.zip")

if __name__ == "__main__":
    dsn = "postgresql://postgres:postgres@localhost:5432/crypto"
    try:
        # Check if database is accessible, if not, use mock_data or fallback
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(load_training_data(dsn, "BTC/USDT"))
        if not df.empty:
            train_artemis_tb(df)
        else:
            print("DB empty. Training aborted.")
    except Exception as e:
        print(f"Database connection failed: {e}. Falling back to Generated Mock Data.")
        df = generate_ohlcv("BTC/USDT", days=150)
        train_artemis_tb(df)
