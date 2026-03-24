from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys
import os
import asyncio
import pandas as pd
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.train import train_agent, load_data
from agent.evaluate import evaluate_agent
from data.database import TimescaleDB
try:
    from data.fetcher import BinanceDataFetcher
    from rag.chat import RAGChatbot
    from rag.sentiment import get_market_sentiment_score
except ImportError:
    # Fallbacks for local testing if pathing is slightly different
    pass

# Initialize RAG
chatbot = RAGChatbot()

# In-memory simple bot state
bot_state = {
    "is_training": False,
    "last_training_error": None,
    "status": "idle",
    "current_price": 65000.0,
    "grid_width_pct": 0.05
}

async def price_poller():
    """Simulates/Fetches market data updates for the UI"""
    while True:
        try:
            import random
            bot_state["current_price"] += random.uniform(-10, 10)
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            break

async def startup_db_and_data():
    """Initializes DB and fetches initial data for a better user experience."""
    print("STARTING DATABASE INITIALIZATION...")
    dsn = "postgresql://postgres:postgres@db:5432/crypto"
    db = TimescaleDB(dsn)
    try:
        await db.connect()
        await db.create_tables()
        
        # Prefetch data if table is empty
        import time
        since = int((time.time() - 30 * 24 * 60 * 60) * 1000)
        for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            df = await db.fetch_data(symbol)
            if df.empty:
                print(f"Fetching initial data for {symbol}...")
                fetcher = BinanceDataFetcher(symbol=symbol, timeframe="1h")
                initial_df = await fetcher.fetch_historical_data(since=since, limit=1000)
                if not initial_df.empty:
                    await db.insert_data(initial_df, symbol)
        
        await db.disconnect()
        print("Database initialized and ready.")
    except Exception as e:
        print(f"Startup DB error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    task = asyncio.create_task(price_poller())
    asyncio.create_task(startup_db_and_data())
    yield
    # Shutdown
    task.cancel()

app = FastAPI(title="RL Grid Trading Bot API", version="1.0.0", lifespan=lifespan)

# Mount statics AFTER app initialization
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../frontend/static")), name="static")
app.mount("/models", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../models")), name="models")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../frontend/templates"))

class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    dsn: str
    symbol: str = "BTC/USDT"

async def background_train(dsn: str, symbol: str):
    bot_state["is_training"] = True
    bot_state["status"] = "fetching_data"
    try:
        df = await load_data(dsn, symbol)
        if df.empty:
             bot_state["status"] = "error"
             bot_state["last_training_error"] = "No data fetched from DB."
             return
             
        bot_state["status"] = "training"
        await asyncio.to_thread(train_agent, df)
        bot_state["status"] = "completed"
    except Exception as e:
        bot_state["status"] = "error"
        bot_state["last_training_error"] = str(e)
    finally:
        bot_state["is_training"] = False

@app.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if bot_state["is_training"]:
        return {"message": "Training is already in progress"}
    background_tasks.add_task(background_train, req.dsn, req.symbol)
    return {"message": "Training started in background"}

@app.get("/status")
async def get_status():
    return bot_state

@app.get("/evaluate")
async def get_evaluation(dsn: str, symbol: str = "BTC/USDT", buffer_pct: float = 0.0):
    try:
        df = await load_data(dsn, symbol)
        if df.empty:
            return {"error": "No data available for evaluation"}
        
        # evaluation now returns 10 values
        rl_worth, static_worth, prices, rl_trades, times, ma_worth, ma_trades, ma5, ma10, sensitivity = evaluate_agent(df, buffer_pct)
        
        # Format OHLCV for Lightweight Charts
        ohlcv_data = []
        for i, (idx, row) in enumerate(df.iterrows()):
            ohlcv_data.append({
                "time": int(idx.timestamp()),
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close']
            })

        # Format RL Trades
        formatted_rl_trades = []
        for t in rl_trades:
            if t['step'] < len(times):
                formatted_rl_trades.append({
                    "time": int(times[t['step']].timestamp()),
                    "type": t['type'],
                    "price": t['price']
                })
        
        # Format MA Trades
        formatted_ma_trades = []
        for t in ma_trades:
            if t['step'] < len(times):
                formatted_ma_trades.append({
                    "time": int(times[t['step']].timestamp()),
                    "type": t['type'],
                    "price": t['price']
                })

        return {
            "rl_net_worth": rl_worth,
            "static_net_worth": static_worth,
            "ma_net_worth": ma_worth,
            "ohlcv": ohlcv_data,
            "rl_trades": formatted_rl_trades,
            "ma_trades": formatted_ma_trades,
            "ma5": ma5,
            "ma10": ma10,
            "sensitivity": sensitivity,
            "plot_saved_at": "models/evaluation_plot.png"
        }
    except Exception as e:
         import traceback
         traceback.print_exc()
         return {"error": str(e)}

@app.get("/sentiment")
async def get_sentiment():
    try:
        score = get_market_sentiment_score()
        return {"score": score}
    except:
        return {"score": 0.0}

@app.post("/chat")
async def handle_chat(req: ChatRequest):
    msg = req.message
    try:
        reply = chatbot.ask(msg)
        return {"reply": reply}
    except Exception as e:
        return {"reply": "Sorry, I am currently facing issues retrieving knowledge computations."}

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
