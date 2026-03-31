import sys
import os
import pandas as pd
import numpy as np

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.evaluate import evaluate_agent
from data.indicators import calculate_features

def test_evaluation_pipeline():
    print("Testing Multi-Strategy Evaluation Pipeline...")
    np.random.seed(42)
    n = 100
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.rand(n) * 100
    })
    
    # Calculate indicators
    df = calculate_features(df)
    
    # Test case: No model found (should return base wealth)
    print(f"Running evaluation on {len(df)} rows without model file...")
    results = evaluate_agent(df, buffer_pct=0.01, model_path="non_existent.zip")
    
    # rl_worth, static_worth, prices, rl_trades, time_indices, ma_worth, ma_trades, ma5, ma10, sensitivity, static_trades
    rl_worth = results[0]
    ma_worth = results[5]
    
    print(f"RL Worth Count: {len(rl_worth) if rl_worth else 0}")
    print(f"MA Worth Count: {len(ma_worth)}")
    
    assert len(ma_worth) == len(df), "MA results should match input length"
    print("✅ Evaluation Pipeline Baseline Passed!")

if __name__ == "__main__":
    test_evaluation_pipeline()
