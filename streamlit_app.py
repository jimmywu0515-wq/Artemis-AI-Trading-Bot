import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
from datetime import datetime, timedelta
import numpy as np
import ccxt

# Path setup to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent.evaluate import evaluate_agent, calculate_ma_strategy, run_sensitivity_analysis
from data.indicators import calculate_features
from data.database import TimescaleDB

st.set_page_config(page_title="Artemis AI - Trading Dashboard", layout="wide", page_icon="🤖")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    .stButton>button { background-image: linear-gradient(to right, #3b82f6, #8b5cf6); color: white; border: none; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Artemis AI Trading Dashboard")
st.markdown("Professional Reinforcement Learning & Technical Strategy Analysis")
st.info("💡 **Economic Mode:** Real-world 0.1% Trading Fees are now enabled for all strategies to ensure fair comparison.")

# Sidebar Controls
st.sidebar.header("🕹️ Control Center")
symbol = st.sidebar.selectbox("Market Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
lookback = st.sidebar.slider("Lookback Period (Hours)", 100, 2000, 1500, 100)
buffer_pct = st.sidebar.slider("Sell Buffer %", 0.0, 5.0, 1.0, 0.1) / 100.0
show_ma = st.sidebar.checkbox("Show Moving Averages (5/10)", value=True)

@st.cache_data(ttl=300)
def get_data(symbol, limit=1000):
    """Fetches data from DB or falls back to live Kraken/Binance API."""
    dsn = "postgresql://postgres:postgres@db:5432/crypto"
    db = TimescaleDB(dsn)
    
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def fetch():
            await db.connect()
            df = await db.fetch_data(symbol)
            await db.disconnect()
            return df
        
        df_db = loop.run_until_complete(fetch())
        if df_db is not None and not df_db.empty:
            return df_db.tail(limit)
    except Exception:
        pass 

    exchanges_to_try = [ccxt.kraken(), ccxt.binance(), ccxt.kucoin()]
    for exchange in exchanges_to_try:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
            df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms')
            df_live.set_index('timestamp', inplace=True)
            if not df_live.empty:
                st.sidebar.success(f"✅ Live Data: {exchange.id.upper()}")
                return df_live
        except Exception:
            continue

    st.sidebar.error("⚠️ All Live Data APIs Restricted. Using Simulation.")
    dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
    base_price = 65000 if "BTC" in symbol else (3500 if "ETH" in symbol else 150)
    prices = base_price + np.cumsum(np.random.randn(limit) * (base_price * 0.01))
    df_mock = pd.DataFrame({'open': prices * 0.999, 'high': prices * 1.005, 'low': prices * 0.995, 'close': prices, 'volume': np.random.rand(limit) * 100}, index=dates)
    return df_mock

# Execution
df = get_data(symbol, lookback)
df = calculate_features(df)

if st.sidebar.button("🚀 Run Full Evaluation"):
    with st.spinner("Analyzing Market Dynamics..."):
        # Run Evaluation
        (rl_worth, static_worth, prices, rl_trades, time_indices, 
         ma_worth, ma_trades, ma5, ma10, sensitivity, static_trades) = evaluate_agent(df, buffer_pct)
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        rl_final = rl_worth[-1] if rl_worth else 10000
        ma_final = ma_worth[-1] if ma_worth else 10000
        static_final = static_worth[-1] if static_worth else 10000
        
        col1.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}")
        
        if not rl_worth:
            col2.metric("RL Bot (Smart Aggressive)", "$10,000.00", "OFFLINE")
        else:
            col2.metric("RL Bot (Smart Aggressive)", f"${rl_final:,.2f}", f"{((rl_final/10000)-1)*100:.2f}%")
            
        col3.metric("MA Cross Strategy", f"${ma_final:,.2f}", f"{((ma_final/10000)-1)*100:.2f}%")
        col4.metric("Static Grid Baseline", f"${static_final:,.2f}", f"{((static_final/10000)-1)*100:.2f}%")

        # Main Candlestick Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Market"), row=1, col=1)
        
        if show_ma:
            fig.add_trace(go.Scatter(x=df.index, y=df['ma5'], line=dict(color='orange', width=1, dash='dot'), name="5 MA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ma10'], line=dict(color='blue', width=1, dash='dot'), name="10 MA"), row=1, col=1)

        # Plot All Trades
        for t in rl_trades:
            if t['step'] < len(df):
                color = "#00FFFF" if t['type'] == 'buy' else "#FF00FF"
                fig.add_trace(go.Scatter(x=[df.index[t['step']]], y=[t['price']], mode="markers", 
                             marker=dict(symbol="triangle-up" if t['type']=='buy' else "triangle-down", color=color, size=12), 
                             name=f"RL {t['type'].upper()}"), row=1, col=1)

        for t in ma_trades:
            if t['step'] < len(df):
                color = "#F59E0B" if t['type'] == 'buy' else "#FBBF24"
                fig.add_trace(go.Scatter(x=[df.index[t['step']]], y=[t['price']], mode="markers", 
                             marker=dict(symbol="diamond", color=color, size=8), 
                             name=f"MA {t['type'].upper()}"), row=1, col=1)

        # Performance Chart
        fig.add_trace(go.Scatter(x=df.index, y=rl_worth, line=dict(color='#8b5cf6', width=2), name="RL Strategy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma_worth, line=dict(color='#f59e0b', width=1), name="MA Strategy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=static_worth, line=dict(color='#64748b', width=1, dash='dot'), name="Static Baseline"), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Combined Trade History Table
        st.subheader("📝 Master Trade History (Fee Aware)")
        
        combined_trades = []
        def add_to_combined(trades, strategy_name):
            for t in trades:
                if t['step'] < len(df):
                    fee_val = t.get('fee', 0.0)
                    combined_trades.append({
                        "Time": df.index[t['step']],
                        "Strategy": strategy_name,
                        "Action": t['type'].upper(),
                        "Price": t['price'],
                        "Fee Paid": fee_val
                    })
        
        add_to_combined(rl_trades, "RL Agent")
        add_to_combined(ma_trades, "MA Cross")
        add_to_combined(static_trades, "Static Grid")
        
        if combined_trades:
            trade_df = pd.DataFrame(combined_trades).sort_values(by="Time", ascending=False)
            trade_df["Time"] = trade_df["Time"].dt.strftime('%Y-%m-%d %H:%M')
            trade_df["Price"] = trade_df["Price"].map("${:,.2f}".format)
            trade_df["Fee Paid"] = trade_df["Fee Paid"].map("${:,.2f}".format)
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.info("No trades executed.")

        # Aggressive Mode Insights
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🛠️ Optimization Active")
        st.sidebar.write("- Risk Reward Ratio refined")
        st.sidebar.write("- Transaction Costs modelled")
        st.sidebar.write("- Drawdown tolerance increased")

        # Sensitivity Analysis
        st.subheader("📊 MA Strategy Buffer Sensitivity")
        sens_df = pd.DataFrame(sensitivity)
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(x=sens_df['buffer'], y=sens_df['net_worth'], marker_color='orange', name="MA Performance"))
        fig_sens.add_trace(go.Scatter(x=[sens_df['buffer'].min(), sens_df['buffer'].max()], y=[rl_final, rl_final], mode="lines", 
                                    line=dict(color='purple', width=2, dash='dash'), name="RL Benchmark"))
        fig_sens.update_layout(template="plotly_dark", height=400, title="Final Net Worth vs Buffer %")
        st.plotly_chart(fig_sens, use_container_width=True)

else:
    st.info("👈 Use the Sidebar to configure your strategy and click 'Run Full Evaluation' to begin!")
    st.image("https://images.unsplash.com/photo-1611974717484-2a62372f4f2c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", caption="Artemis AI - Multi-Strategy Platform Ready")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Artemis AI Systems. Aggressive Optimization Active.")
