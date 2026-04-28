import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────

def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50  # normalize to [-1, 1]

def compute_macd(prices: np.ndarray, fast=12, slow=26, signal=9):
    s = pd.Series(prices)
    ema_fast = s.ewm(span=fast, adjust=False).mean().values
    ema_slow = s.ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - signal_line
    price_scale = prices + 1e-9
    return macd_line / price_scale, signal_line / price_scale, hist / price_scale

def compute_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    s = pd.Series(prices)
    mid = s.rolling(period, min_periods=1).mean().values
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0).values
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = upper - lower + 1e-9
    pct_b = (prices - lower) / band_width
    return np.clip(pct_b * 2 - 1, -1, 1)

def compute_atr(high, low, close, period: int = 14) -> np.ndarray:
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
#  TRIPLE BARRIER TRADING ENVIRONMENT
# ─────────────────────────────────────────────

class TripleBarrierTradingEnv(gym.Env):
    """
    Artemis Triple Barrier - Discrete Action Trading Environment

    Observation (20 features):
      [0..14] Technicals: log_returns, RSI, MACD, Bollinger, ATR, Momentum, Volatility
      [15] Current Position (-1, 0, 1)
      [16] Unrealised PnL (normalized)
      [17] Distance to TP (normalized)
      [18] Distance to SL (normalized)
      [19] Steps remaining in vertical barrier (normalized)

    Action (Discrete 3):
      0: Flat / Close Position
      1: Go Long
      2: Go Short
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost_bps: float = 5.0,
        max_hold_steps: int = 50,
        atr_multiplier: float = 2.0,
        min_distance_pct: float = 0.01,
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
        self.max_hold_steps = max_hold_steps
        self.atr_multiplier = atr_multiplier
        self.min_distance_pct = min_distance_pct

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _get_obs(self) -> np.ndarray:
        t = self._t
        price = self.prices[t]
        
        pos_val = 0.0
        if self._position_type == 1: pos_val = 1.0
        elif self._position_type == 2: pos_val = -1.0

        unrealised_pnl = 0.0
        dist_tp = 0.0
        dist_sl = 0.0
        steps_rem = 0.0

        if self._position_type != 0:
            if self._position_type == 1: # Long
                unrealised_pnl = (price - self._entry_price) / self._entry_price
                dist_tp = (self._tp_price - price) / self._entry_price
                dist_sl = (price - self._sl_price) / self._entry_price
            else: # Short
                unrealised_pnl = (self._entry_price - price) / self._entry_price
                dist_tp = (price - self._tp_price) / self._entry_price
                dist_sl = (self._sl_price - price) / self._entry_price
            
            steps_rem = max(0, self.max_hold_steps - self._steps_in_position) / self.max_hold_steps

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
            np.clip(self._mom5[t], -0.2, 0.2) / 0.2,
            np.clip(self._mom10[t], -0.3, 0.3) / 0.3,
            np.clip(self._vol_z[t], -3, 3) / 3,
            np.clip(self._vol20[t] * 100, 0, 5) / 5,
            np.clip(self._portfolio_value / self.initial_capital, 0.5, 2.0) - 1.0,
            np.clip(self._max_drawdown, 0, 0.5) / 0.5,
            pos_val,
            np.clip(unrealised_pnl * 10, -1, 1),
            np.clip(dist_tp * 10, 0, 1),
            np.clip(dist_sl * 10, 0, 1),
            steps_rem
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 30
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._max_drawdown = 0.0
        
        self._position_type = 0 # 0: Flat, 1: Long, 2: Short
        self._position_size = 0.0
        self._entry_price = 0.0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._steps_in_position = 0
        
        self._trades = []
        return self._get_obs(), {}

    def _close_position(self, current_price, reason):
        if self._position_type == 1:
            gross_pnl = (current_price - self._entry_price) * self._position_size
            trade_type = 'sell'
        else:
            gross_pnl = (self._entry_price - current_price) * self._position_size
            trade_type = 'buy'
            
        fee = (current_price * self._position_size) * self.tc_bps
        net_pnl = gross_pnl - fee
        
        self._portfolio_value += net_pnl
        
        t_info = {'step': self._t, 'type': trade_type, 'price': current_price, 'fee': fee, 'pnl': net_pnl, 'reason': reason}
        self._trades.append(t_info)
        
        self._position_type = 0
        self._position_size = 0.0
        self._entry_price = 0.0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._steps_in_position = 0
        
        return net_pnl, t_info

    def _open_position(self, p_type, current_price, distance):
        self._position_type = p_type
        self._entry_price = current_price
        
        # Allocate 95% of portfolio to the trade
        invest_amount = self._portfolio_value * 0.95
        self._position_size = invest_amount / current_price
        
        entry_fee = invest_amount * self.tc_bps
        self._portfolio_value -= entry_fee
        
        if p_type == 1: # Long
            self._tp_price = current_price + distance
            self._sl_price = current_price - distance
            self._trades.append({'step': self._t, 'type': 'buy', 'price': current_price, 'fee': entry_fee, 'reason': 'open_long'})
        else: # Short
            self._tp_price = current_price - distance
            self._sl_price = current_price + distance
            self._trades.append({'step': self._t, 'type': 'sell', 'price': current_price, 'fee': entry_fee, 'reason': 'open_short'})
            
        self._steps_in_position = 0

    def step(self, action):
        price = self.prices[self._t]
        atr_val = self._atr[self._t] * price
        distance = max(atr_val * self.atr_multiplier, price * self.min_distance_pct)

        reward = 0.0
        trade_occurred = False

        # Handle agent requests
        if action == 0: # Request Flat
            if self._position_type != 0:
                net_pnl, _ = self._close_position(price, reason="early_exit")
                reward += (net_pnl / self.initial_capital) * 100.0
                trade_occurred = True
                
        elif action == 1: # Request Long
            if self._position_type == 2: # Close short first
                net_pnl, _ = self._close_position(price, reason="early_exit_reverse")
                reward += (net_pnl / self.initial_capital) * 100.0
                self._open_position(1, price, distance)
                trade_occurred = True
            elif self._position_type == 0:
                self._open_position(1, price, distance)
                trade_occurred = True
                
        elif action == 2: # Request Short
            if self._position_type == 1: # Close long first
                net_pnl, _ = self._close_position(price, reason="early_exit_reverse")
                reward += (net_pnl / self.initial_capital) * 100.0
                self._open_position(2, price, distance)
                trade_occurred = True
            elif self._position_type == 0:
                self._open_position(2, price, distance)
                trade_occurred = True

        # Process Active Position Barriers
        if self._position_type != 0 and not trade_occurred:
            hit_tp = False
            hit_sl = False
            
            if self._position_type == 1: # Long
                hit_tp = price >= self._tp_price
                hit_sl = price <= self._sl_price
            elif self._position_type == 2: # Short
                hit_tp = price <= self._tp_price
                hit_sl = price >= self._sl_price
                
            hit_time = self._steps_in_position >= self.max_hold_steps

            if hit_tp:
                self._close_position(price, reason="tp")
                reward += 2.0
            elif hit_sl:
                self._close_position(price, reason="sl")
                reward += -2.0
            elif hit_time:
                net_pnl, _ = self._close_position(price, reason="timeout")
                reward += (net_pnl / self.initial_capital) * 100.0
            else:
                self._steps_in_position += 1
                reward -= 0.01 # Time penalty

        # Drawdown Tracking
        if self._portfolio_value > self._peak_value: 
            self._peak_value = self._portfolio_value
        dd = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-9)
        self._max_drawdown = max(self._max_drawdown, dd)

        self._t += 1
        done = self._t >= self.n - 1

        if self._portfolio_value <= self.initial_capital * 0.5:
            reward -= 5.0
            done = True

        obs = self._get_obs()
        info = { 
            "portfolio_value": self._portfolio_value, 
            "net_worth": self._portfolio_value,
            "max_drawdown": self._max_drawdown, 
            "trades": self._trades 
        }
        return obs, float(reward), done, False, info
