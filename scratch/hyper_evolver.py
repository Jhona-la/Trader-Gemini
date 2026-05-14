import os
import sys
import logging
import random
import optuna
import pandas as pd
import time
from datetime import datetime, timezone

# Disable logging spam for speed
logging.getLogger().setLevel(logging.CRITICAL)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from queue import Queue
from core.events import FillEvent, OrderSide, EventType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from strategies.technical import HybridScalpingStrategy
from core.backtest_infra import fetch_binance_data, BacktestDataProvider, COMMISSION_PCT, COMMISSION_MAKER
from scripts.quick_diagnostic_bt import SimpleExecutor

# 1. Fetch data ONCE for speed
print("Fetching 1 day of historical data for optimization...")
df = fetch_binance_data('BTC/USDT', days=1)
print(f"Data ready: {len(df)} bars.")
all_data = {'BTC/USDT': df}

def objective(trial):
    # --- Mutate Hyperparameters ---
    min_atr = trial.suggest_float("min_atr", 0.0005, 0.0020, log=True)
    adx_thresh = trial.suggest_int("adx", 10, 30)
    max_tp = trial.suggest_float("max_tp", 0.0010, 0.0040)
    sl_mult = trial.suggest_float("sl_mult", 0.5, 3.0)
    
    # Inject into Config to be read by technical.py
    if not hasattr(Config, 'Mutations'):
        Config.Mutations = {}
        
    Config.Mutations['min_atr_required'] = min_atr
    Config.Mutations['adx_threshold'] = adx_thresh
    Config.Mutations['max_tp_cap'] = max_tp
    Config.Mutations['sl_multiplier'] = sl_mult
    
    # Disable Observability to avoid Telegram crashes
    if hasattr(Config, 'Observability'):
        Config.Observability.TELEGRAM_ENABLED = False
        Config.Observability.DISCORD_ENABLED = False
        Config.Observability.EMAIL_ENABLED = False
        
    # --- Setup Backtest ---
    events_queue = Queue()
    dp = BacktestDataProvider(events_queue, ['BTC/USDT'], all_data.copy())
    
    portfolio = Portfolio(
        initial_capital=13.0,
        csv_path='scratch/dummy_trades.csv',
        status_path='scratch/dummy_status.csv',
        auto_save=False
    )
    portfolio.data_provider = dp
    rm = RiskManager(max_concurrent_positions=2, portfolio=portfolio)
    tech = HybridScalpingStrategy(dp, events_queue, horizon='SCALPING')
    executor = SimpleExecutor(seed=42)
    
    epochs = dp.total_epochs
    warmup = min(100, epochs // 20)
    
    portfolio.trade_history = []
    portfolio.current_cash = 13.0
    
    # --- Execution Loop ---
    for epoch in range(epochs):
        dp.update_bars()
        
        market_events = []
        while not events_queue.empty():
            evt = events_queue.get()
            if evt.type == EventType.MARKET:
                market_events.append(evt)
                
        bar_time = pd.to_datetime(dp.current_time_ms, unit='ms', utc=True)
        if bar_time:
            executor.current_bar_time = bar_time
            
        for evt in market_events:
            portfolio.update_market_price(evt.symbol, evt.close_price)
            
        if epoch < warmup:
            continue
            
        # check stops
        try:
            for evt in market_events:
                bar = dp.get_latest_bars(evt.symbol, n=1)
                if bar is not None and len(bar) > 0:
                    stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
                    if stop_sigs:
                        for sig in stop_sigs:
                            events_queue.put(sig)
        except Exception:
            pass

        # strategy
        try:
            class DummyEvent: pass
            dummy = DummyEvent()
            if bar_time: dummy.timestamp = bar_time
            tech.generate_signals(event=dummy)
        except Exception:
            pass
            
        # execution
        while not events_queue.empty():
            evt = events_queue.get()
            etype = evt.type.name if hasattr(evt.type, 'name') else str(evt.type)
            if etype == 'SIGNAL':
                bar = dp.get_latest_bars(evt.symbol, n=1)
                if bar is not None and len(bar) > 0:
                    price = float(bar['close'][-1])
                    order = rm.generate_order(evt, price)
                    if order:
                        fill = executor.execute(order, price)
                        if fill:
                            portfolio.update_fill(fill)
            elif etype == 'ORDER':
                bar = dp.get_latest_bars(evt.symbol, n=1)
                if bar is not None and len(bar) > 0:
                    fill = executor.execute(evt, float(bar['close'][-1]))
                    if fill: events_queue.put(fill)
            elif etype == 'FILL':
                portfolio.update_fill(evt)

    # Calculate metrics
    trades = len(portfolio.trade_history)
    wins = sum(1 for t in portfolio.trade_history if t['net_pnl'] > 0)
    wr = (wins / trades * 100) if trades > 0 else 0
    net_pnl = sum(t['net_pnl'] for t in portfolio.trade_history)
    
    score = (wr * 50) + (trades * 2) + (net_pnl * 500)
    if trades < 50: score -= 1000.0 - trades
    if wr < 70.0: score -= 500.0 - wr
    if net_pnl <= 0: score -= 100.0 - (net_pnl * 10)
    
    trial.set_user_attr("trades", trades)
    trial.set_user_attr("wr", wr)
    trial.set_user_attr("net_pnl", net_pnl)
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1)
    
    print("\n" + "="*50)
    print("🏆 BEST GENOTYPE FOUND:")
    best = study.best_trial
    print(f"  Score: {best.value}")
    print(f"  Trades: {best.user_attrs['trades']} | WR: {best.user_attrs['wr']:.1f}% | Net PnL: ${best.user_attrs['net_pnl']:.4f}")
    print("  Parameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    print("="*50)
