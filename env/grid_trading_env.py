import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class GridTradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates a grid trading bot on historical OHLCV data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, grid_count: int = 10):
        super(GridTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.grid_count = grid_count
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # State/Observation Space
        # Current Price, ATR, RSI, Log Return, Current Balance, Current Crypto Holding, Grid Profit, Sentiment Score
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # Action Space:
        # [0] Change grid width percentage (e.g., -0.05 to +0.05)
        # [1] Shift grid center percentage (e.g., -0.02 to +0.02)
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.05], dtype=np.float32), 
            high=np.array([0.1, 0.05], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize internal state
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.grid_profit = 0.0
        
        # Current Grid Parameters
        self.current_price = 0.0
        self.grid_center = 0.0
        self.grid_width_pct = 0.05 # 5% default
        self.trade_history = [] # List of {'step': int, 'type': 'buy'/'sell', 'price': float}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # In a real live scenario, sentiment would be fetched actively per step.
        # For historical training/backtesting, we either mock it or compute historical sentiment.
        # Here we add a neutral default or mock historical pattern for testing.
        mock_historical_sentiment = 0.0 # Neutral 
        
        obs = np.array([
            row['close'],          # Current Price
            row['atr'],            # Volatility indicator
            row['rsi'],            # Momentum indicator
            row['log_return'],     # Price momentum
            self.balance,          # Available USDT
            self.crypto_held,      # Current holding
            self.grid_profit,      # Accumulated grid profit
            mock_historical_sentiment # Sentiment Score from RAG
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly select a starting point if we want to train robustly,
        # but for simplicity let's start at a fixed offset to allow indicator computation.
        self.current_step = 0
        
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.grid_profit = 0.0
        
        row = self.df.iloc[self.current_step]
        self.current_price = row['close']
        self.grid_center = self.current_price
        self.grid_width_pct = 0.05
        self.trade_history = []
        
        return self._get_obs(), {}

    def _simulate_grid_trading(self, low: float, high: float):
        """
        Simulate the profit of grid trading during the current interval.
        Explicitly tracks buy/sell cross events.
        """
        upper_bound = self.grid_center * (1 + self.grid_width_pct)
        lower_bound = self.grid_center * (1 - self.grid_width_pct)
        grid_step = (upper_bound - lower_bound) / self.grid_count
        
        if grid_step <= 0: return

        # Calculate hits based on previous price vs current low/high
        # If price moves across a grid level, we trigger a trade.
        # This is still a simulation, but more explicit.
        levels = [lower_bound + i * grid_step for i in range(self.grid_count + 1)]
        
        for level in levels:
            # If price crossed from above to below level -> BUY
            if self.current_price > level and low <= level:
                self.trade_history.append({'step': self.current_step, 'type': 'buy', 'price': level})
                # In real grid, we'd allocate capital here.
            # If price crossed from below to above level -> SELL
            if self.current_price < level and high >= level:
                self.trade_history.append({'step': self.current_step, 'type': 'sell', 'price': level})
                # Record profit (simplified)
                profit = (self.balance / self.grid_count) * (grid_step / level)
                self.grid_profit += profit
                self.balance += profit

    def step(self, action):
        self.previous_net_worth = self.net_worth
        
        # Apply actions
        width_change, center_shift = action
        self.grid_width_pct = max(0.01, self.grid_width_pct + width_change) # Enforce min width
        self.grid_center = self.grid_center * (1 + center_shift)
        
        # Advance time
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        if not done:
            row = self.df.iloc[self.current_step]
            self.current_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Simulate trading
            self._simulate_grid_trading(low_price, high_price)
            
            # Calculate new net worth
            # Simplified: assuming we hold a dynamic amount of crypto based on grid position
            # This needs to be correctly modeled in a full bot.
            self.net_worth = self.balance + (self.crypto_held * self.current_price)
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
            
            # Calculate Drawdown
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            
            # Reward Calculation: Sharpe Ratio variant
            # Reward = Profit - Penalty * Drawdown
            profit = self.net_worth - self.previous_net_worth
            lambda_penalty = 1000.0 # High penalty for drawdown
            
            reward = profit - (lambda_penalty * drawdown)
            
        else:
            reward = 0.0

        info = {
            'net_worth': self.net_worth,
            'grid_profit': self.grid_profit,
            'grid_center': self.grid_center,
            'grid_width_pct': self.grid_width_pct,
            'trades': self.trade_history
        }

        return self._get_obs(), float(reward), done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Grid Profit: {self.grid_profit:.2f}")

