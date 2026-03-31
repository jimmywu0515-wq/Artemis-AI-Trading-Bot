import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────

def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index, returns values 0-100 normalized to [-1, 1]."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50  # normalize to [-1, 1]

def compute_macd(prices: np.ndarray, fast=12, slow=26, signal=9):
    """MACD line and signal line, normalized by price."""
    s = pd.Series(prices)
    ema_fast = s.ewm(span=fast, adjust=False).mean().values
    ema_slow = s.ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - signal_line
    price_scale = prices + 1e-9
    return macd_line / price_scale, signal_line / price_scale, hist / price_scale

def compute_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Returns %B indicator in [-1, 1] range."""
    s = pd.Series(prices)
    mid = s.rolling(period, min_periods=1).mean().values
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0).values
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = upper - lower + 1e-9
    pct_b = (prices - lower) / band_width  # 0=lower, 1=upper
    return np.clip(pct_b * 2 - 1, -1, 1)  # normalize to [-1, 1]

def compute_atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average True Range, normalized by close price."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr / (close + 1e-9)

# ─────────────────────────────────────────────
#  IMPROVED TRADING ENVIRONMENT
# ─────────────────────────────────────────────

class ImprovedTradingEnv(gym.Env):
    """
    Artemis V2 - Advanced Continuous Action Trading Environment

    Observation (18 features):
      [0]  log return (t)
      [1]  log return (t-1)
      [2]  log return (t-2)
      ... [RSI, MACD, Bollinger, ATR, Momentum, Volatility, Drawdown]

    Action (2 continuous values):
      [0]  target position change direction ∈ [-1, 1]
      [1]  size fraction ∈ [0, 1]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost_bps: float = 5.0,
        max_position: float = 1.0,
        reward_window: int = 20,
        max_hold_steps: int = 50,
        drawdown_penalty: float = 0.5,
    ):
        super().__init__()

        self.df = df
        self.prices = df['close'].values.astype(np.float32)
        self.highs = df['high'].values.astype(np.float32) if 'high' in df else self.prices * 1.005
        self.lows = df['low'].values.astype(np.float32) if 'low' in df else self.prices * 0.995
        self.volumes = df['volume'].values.astype(np.float32) if 'volume' in df else np.ones_like(self.prices)
        self.n = len(self.prices)

        self.initial_capital = initial_capital
        self.tc_bps = transaction_cost_bps / 10000
        self.max_position = max_position
        self.reward_window = reward_window
        self.max_hold_steps = max_hold_steps
        self.drawdown_penalty_coef = drawdown_penalty

        # Pre-compute technicals
        self._log_returns = np.log(self.prices[1:] / self.prices[:-1] + 1e-9)
        self._log_returns = np.concatenate([[0.0], self._log_returns])
        self._rsi = compute_rsi(self.prices)
        self._macd, self._macd_sig, self._macd_hist = compute_macd(self.prices)
        self._bband = compute_bollinger(self.prices)
        self._atr = compute_atr(self.highs, self.lows, self.prices)
        self._vol20 = pd.Series(self._log_returns).rolling(20, min_periods=1).std(ddof=0).values.astype(np.float32)
        self._mom5 = np.concatenate([[0.0]*5, (self.prices[5:] - self.prices[:-5]) / (self.prices[:-5] + 1e-9)])
        self._mom10 = np.concatenate([[0.0]*10, (self.prices[10:] - self.prices[:-10]) / (self.prices[:-10] + 1e-9)])
        vol_roll = pd.Series(self.volumes)
        self._vol_z = ((vol_roll - vol_roll.rolling(20, min_periods=1).mean()) / (vol_roll.rolling(20, min_periods=1).std(ddof=0) + 1e-9)).values.astype(np.float32)

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32))

        self.reset()

    def _get_obs(self) -> np.ndarray:
        t = self._t
        obs = np.array([
            self._log_returns[t],
            self._log_returns[max(t-1, 0)],
            self._log_returns[max(t-2, 0)],
            self._rsi[t],
            self._macd[t],
            self._macd_sig[t],
            self._macd_hist[t],
            self._bband[t],
            self._atr[t],
            np.clip(self._position / (self._portfolio_value * self.max_position + 1e-9), -1, 1),
            np.clip(self._unrealised_pnl / (self.initial_capital * 0.1 + 1e-9), -1, 1),
            np.clip(self._mom5[t], -0.2, 0.2) / 0.2,
            np.clip(self._mom10[t], -0.3, 0.3) / 0.3,
            np.clip(self._vol_z[t], -3, 3) / 3,
            np.clip(self._vol20[t] * 100, 0, 5) / 5,
            min(self._steps_in_position / self.max_hold_steps, 1.0),
            np.clip(self._portfolio_value / self.initial_capital, 0.5, 2.0) - 1.0,
            np.clip(self._max_drawdown, 0, 0.5) / 0.5,
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 30
        self._position = 0.0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._max_drawdown = 0.0
        self._entry_price = None
        self._unrealised_pnl = 0.0
        self._steps_in_position = 0
        self._step_returns = []
        self._trades = []
        return self._get_obs(), {}

    def step(self, action):
        direction = float(np.clip(action[0], -1, 1))
        size_frac  = float(np.clip(action[1], 0, 1))
        delta_pos_raw = direction * size_frac

        price = self.prices[self._t]
        max_dollar = self._portfolio_value * self.max_position
        target_position = delta_pos_raw * max_dollar
        trade_size = target_position - self._position
        tc = abs(trade_size) * self.tc_bps

        if abs(trade_size) > 1e-6:
            trade_type = 'buy' if trade_size > 0 else 'sell'
            self._trades.append({'step': self._t, 'type': trade_type, 'price': price, 'fee': tc})

        self._position = target_position
        if abs(self._position) > 1e-6:
            if self._entry_price is None: self._entry_price = price
            self._steps_in_position += 1
        else:
            self._entry_price = None
            self._steps_in_position = 0

        self._t += 1
        done = self._t >= self.n - 1

        next_price = self.prices[self._t]
        price_return = (next_price - price) / (price + 1e-9)
        raw_pnl = self._position * price_return - tc
        self._portfolio_value += raw_pnl
        self._unrealised_pnl = self._position * (next_price - price) / (price + 1e-9) * next_price

        if self._portfolio_value > self._peak_value: self._peak_value = self._portfolio_value
        dd = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-9)
        self._max_drawdown = max(self._max_drawdown, dd)

        step_return = raw_pnl / (self.initial_capital + 1e-9)
        self._step_returns.append(step_return)
        if len(self._step_returns) > self.reward_window: self._step_returns.pop(0)

        if len(self._step_returns) >= 2:
            mu, sigma = np.mean(self._step_returns), np.std(self._step_returns) + 1e-8
            reward = mu / sigma
        else:
            reward = step_return * 100

        reward -= self.drawdown_penalty_coef * dd
        if self._portfolio_value <= self.initial_capital * 0.5:
            reward -= 5.0
            done = True

        obs = self._get_obs()
        info = { "portfolio_value": self._portfolio_value, "position": self._position, "max_drawdown": self._max_drawdown, "trades": self._trades, "net_worth": self._portfolio_value}
        return obs, float(reward), done, False, info
