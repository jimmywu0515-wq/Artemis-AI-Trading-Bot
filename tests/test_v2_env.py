import sys
import os
import numpy as np
import pandas as pd

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_env.improved_trading_env import ImprovedTradingEnv

def test_v2_env():
    print("Testing Artemis V2 Environment...")
    np.random.seed(42)
    n = 100
    returns = np.random.normal(0.0003, 0.012, n)
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n)
    })
    
    env = ImprovedTradingEnv(df)
    obs, _ = env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (18,), "Observation shape should be 18"
    
    # Take a random action
    action = np.array([0.5, 0.1], dtype=np.float32) # Buy 10% notional
    next_obs, reward, done, truncated, info = env.step(action)
    
    print(f"Next Obs Shape: {next_obs.shape}")
    print(f"Reward: {reward}")
    print(f"Portfolio Value: {info['portfolio_value']}")
    
    assert next_obs.shape == (18,), "Next observation shape should be 18"
    print("✅ Artemis V2 Environment Verification Passed!")

if __name__ == "__main__":
    test_v2_env()
