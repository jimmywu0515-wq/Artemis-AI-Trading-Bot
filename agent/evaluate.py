import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.grid_trading_env import GridTradingEnv

def evaluate_agent(df: pd.DataFrame, model_path: str = "models/ppo_grid_bot.zip"):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return [], [], [], []

    # Load Model
    model = PPO.load(model_path)
    
    # Setup environments
    rl_env = GridTradingEnv(df)
    static_env = GridTradingEnv(df)

    # Evaluate RL Agent
    obs, _ = rl_env.reset()
    rl_net_worths = []
    trades = [] # Track trade indices and actions
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = rl_env.step(action)
        rl_net_worths.append(info['net_worth'])
        trades = info.get('trades', []) # The env keeps the full list
        
        if done or truncated:
            break

    # Evaluate Static Grid
    obs, _ = static_env.reset()
    static_net_worths = []
    
    done = False
    while not done:
        # Action [0, 0] means no change to grid width and no shift to grid center
        obs, reward, done, truncated, info = static_env.step([0.0, 0.0])
        static_net_worths.append(info['net_worth'])
        
        if done or truncated:
            break

    # Plot Comparison (Legacy)
    plt.figure(figsize=(12, 6))
    plt.plot(rl_net_worths, label='RL Dynamic Grid Bot')
    plt.plot(static_net_worths, label='Static Grid Bot')
    plt.title('RL Grid Bot vs Static Grid Bot Performance')
    plt.legend()
    plt.grid(True)
    
    plot_path = "models/evaluation_plot.png"
    plt.savefig(plot_path)
    plt.close() # Clean up
    
    # Return all info for frontend
    # price_history is useful for the chart
    price_history = df['close'].tolist()
    # Also need time indices for plotting trades
    time_indices = df.index.tolist()
    
    return rl_net_worths, static_net_worths, price_history, trades, time_indices

if __name__ == "__main__":
    # Needs a valid dataframe for testing
    pass
