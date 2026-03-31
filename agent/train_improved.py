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
from trading_env.improved_trading_env import ImprovedTradingEnv
from data.indicators import calculate_features
from data.database import TimescaleDB
import asyncio

async def load_training_data(dsn, symbol):
    db = TimescaleDB(dsn)
    await db.connect()
    df = await db.fetch_data(symbol)
    await db.disconnect()
    return df

def make_env(df, seed=0):
    def _init():
        env = ImprovedTradingEnv(df)
        env = Monitor(env)
        return env
    return _init

def train_artemis_v2(df, model_path="models/artemis_v2", total_timesteps=1000000, n_envs=4):
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
        learning_rate=linear_schedule(3e-4, 1e-5, 1.0),
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.75,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        seed=42
    )

    print(f"Starting Artemis V2 Training ({total_timesteps} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ Artemis V2 saved to {model_path}.zip")

if __name__ == "__main__":
    dsn = "postgresql://postgres:postgres@localhost:5432/crypto"
    try:
        df = asyncio.run(load_training_data(dsn, "BTC/USDT"))
        if not df.empty:
            train_artemis_v2(df)
    except Exception as e:
        print(f"Training failed: {e}")
