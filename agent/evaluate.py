import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_env.grid_trading_env import GridTradingEnv

def calculate_ma_strategy(df: pd.DataFrame, buffer_pct: float = 0.0):
    """
    Implements a 5MA/10MA Cross Strategy with user-defined buffer.
    - Buy (all in) when 5MA > 10MA (Golden Cross)
    - Sell 50% when Price < 5MA * (1 - buffer)
    - Sell 100% when Price < 10MA * (1 - buffer)
    """
    net_worths = []
    trades = []
    
    balance = 10000.0 # Starting USDT
    holdings = 0.0    # Starting Crypto
    trading_fee = 0.001 # 0.1% fee
    
    # Pre-calculate signals
    ma5 = df['ma5']
    ma10 = df['ma10']
    prices = df['close']
    
    for i in range(len(df)):
        price = prices.iloc[i]
        m5 = ma5.iloc[i]
        m10 = ma10.iloc[i]
        
        # Check Buy Signal: 5MA > 10MA
        if m5 > m10 and balance > 0:
            buy_amount = balance
            fee = buy_amount * trading_fee
            holdings += (buy_amount - fee) / price
            balance = 0
            trades.append({'step': i, 'type': 'buy', 'price': price, 'fee': fee})
            
        # Check Sell Signal
        elif holdings > 0 and price < m5 * (1 - buffer_pct):
            if price < m10 * (1 - buffer_pct):
                sell_val = holdings * price
                fee = sell_val * trading_fee
                balance += (sell_val - fee)
                holdings = 0
                trades.append({'step': i, 'type': 'sell', 'price': price, 'fee': fee})
            else:
                sell_amount = holdings * 0.5
                sell_val = sell_amount * price
                fee = sell_val * trading_fee
                balance += (sell_val - fee)
                holdings -= sell_amount
                trades.append({'step': i, 'type': 'sell', 'price': price, 'fee': fee})
                
        net_worths.append(balance + holdings * price)
        
    return net_worths, trades

def run_sensitivity_analysis(df: pd.DataFrame, start_pct: float = 0.0, end_pct: float = 0.05, steps: int = 11):
    """
    Evaluates the MA strategy across a range of buffer percentages.
    Returns a list of {'buffer': pct, 'net_worth': final_value}
    """
    results = []
    import numpy as np
    
    # Generate range from start to end (e.g., 0.0 to 0.05)
    buffers = np.linspace(start_pct, end_pct, steps)
    
    for b in buffers:
        worths, _ = calculate_ma_strategy(df, b)
        final_worth = worths[-1] if worths else 10000.0
        results.append({"buffer": float(round(b * 100, 2)), "net_worth": float(final_worth)})
        
    return results

def evaluate_agent(df: pd.DataFrame, buffer_pct: float = 0.0, model_path: str = "models/ppo_grid_bot.zip"):
    # Load Model
    model = None
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    
    rl_net_worths = []
    rl_trades = []
    
    if model:
        # Evaluate RL Agent
        rl_env = GridTradingEnv(df)
        obs, _ = rl_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = rl_env.step(action)
            rl_net_worths.append(info['net_worth'])
            rl_trades = info.get('trades', [])
            if done or truncated: break
    
    # Evaluate Static Grid
    static_env = GridTradingEnv(df)
    static_net_worths = []
    static_trades = []
    obs, _ = static_env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = static_env.step([0.0, 0.0])
        static_net_worths.append(info['net_worth'])
        static_trades = info.get('trades', [])
        if done or truncated: break

    # Evaluate MA Strategy
    ma_worths, ma_trades = calculate_ma_strategy(df, buffer_pct)
    
    # Run Sensitivity Analysis
    sensitivity_data = run_sensitivity_analysis(df)
    
    # Return all info for frontend
    price_history = df['close'].tolist()
    time_indices = df.index.tolist()
    ma5_history = df['ma5'].tolist() if 'ma5' in df else []
    ma10_history = df['ma10'].tolist() if 'ma10' in df else []
    
    return (rl_net_worths, static_net_worths, price_history, rl_trades, 
            time_indices, ma_worths, ma_trades, ma5_history, ma10_history, 
            sensitivity_data, static_trades)

if __name__ == "__main__":
    pass
