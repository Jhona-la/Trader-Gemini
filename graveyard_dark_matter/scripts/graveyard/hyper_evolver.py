#!/usr/bin/env python3
"""
Hyper-Evolver: Optuna-driven parameter optimization for HybridScalpingStrategy.

QUÉ: Muta parámetros de la estrategia técnica (strength, ADX, ATR, TP, SL)
     y ejecuta un backtest completo por cada mutación para encontrar el genotipo óptimo.
POR QUÉ: Los parámetros actuales generan demasiados trades zombie (600s timeout).
PARA QUÉ: Encontrar la combinación que maximiza Win Rate y Net PnL con capital de $13.
CÓMO: Usa Optuna (optimización bayesiana) para explorar el espacio de parámetros,
      ejecutando un backtest idéntico a quick_diagnostic_bt.py por cada trial.
CUÁNDO: Se ejecuta manualmente antes de producción para calibrar parámetros.
DÓNDE: scripts/hyper_evolver.py
QUIÉN: Quant Developer + QA Engineer
"""
import os, sys, time, io, contextlib, random, logging, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'

import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from queue import Queue

from config import Config
from core.backtest_infra import fetch_binance_data, BacktestDataProvider, COMMISSION_PCT, COMMISSION_MAKER
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from utils.cooldown_manager import cooldown_manager

# Suppress logging spam for speed
logging.getLogger().setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Disable notifications
Config.TELEGRAM_ENABLED = False
Config.EMAIL_ENABLED = False
Config.DISCORD_ENABLED = False
if hasattr(Config, 'Observability'):
    Config.Observability.TELEGRAM_ENABLED = False
    Config.Observability.DISCORD_ENABLED = False
    Config.Observability.EMAIL_ENABLED = False

# Remove stale lock
lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'STOP_TRADING.LOCK')
if os.path.exists(lock_file):
    os.remove(lock_file)

# ── DATA: Fetch ONCE, reuse across all trials ──
print("Fetching 1 day of historical data for optimization...")
symbols = ['BTC/USDT']
from core.backtest_infra import fetch_multi_symbol_data
all_data = fetch_multi_symbol_data(symbols, days=1)
n_bars = len(list(all_data.values())[0])
print(f"Data ready: {n_bars} bars for {len(symbols)} symbols.")


# ── EXECUTOR: Reuse from quick_diagnostic_bt ──
from scripts.quick_diagnostic_bt import SimpleExecutor


def run_single_backtest(mutations: dict) -> dict:
    """
    Execute a full backtest with the given mutations.
    Returns dict with trades, wins, wr, net_pnl, trade_list.
    
    This mirrors the EXACT loop from quick_diagnostic_bt.py for production parity.
    """
    # ── 1. RESET ALL GLOBAL STATE ──
    cooldown_manager.reset()
    if hasattr(cooldown_manager, 'custom_cooldowns'):
        cooldown_manager.custom_cooldowns.clear()
    
    random.seed(42)
    np.random.seed(42)
    
    # Inject mutations into Config
    if not hasattr(Config, 'Mutations'):
        Config.Mutations = {}
    Config.Mutations.update(mutations)
    
    # ── 2. FRESH INSTANCES PER TRIAL ──
    # CRITICAL: BacktestDataProvider DESTROYS the input dict (sets values to None)
    # We must deep-copy the data for each trial
    trial_data = copy.deepcopy(all_data)
    events_queue = Queue()
    dp = BacktestDataProvider(events_queue, symbols, trial_data)
    
    bt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'dashboard', 'data', 'evo_temp')
    os.makedirs(bt_dir, exist_ok=True)
    
    portfolio = Portfolio(
        initial_capital=13.0,
        csv_path=os.path.join(bt_dir, 'evo_trades.csv'),
        status_path=os.path.join(bt_dir, 'evo_status.csv'),
        auto_save=False
    )
    portfolio.data_provider = dp
    
    rm = RiskManager(max_concurrent_positions=2, portfolio=portfolio)
    
    from strategies.technical import HybridScalpingStrategy
    tech = HybridScalpingStrategy(dp, events_queue, horizon='SCALPING')
    
    executor = SimpleExecutor(seed=42)
    
    total = dp.total_epochs
    warmup = min(100, total // 20)
    signals_total = 0
    fills_total = 0
    
    # ── 3. MAIN LOOP (identical to quick_diagnostic_bt.py) ──
    for epoch in range(total):
        dp.update_bars()
        
        # Drain market events
        market_events = []
        while not events_queue.empty():
            evt = events_queue.get()
            if evt.type == EventType.MARKET:
                market_events.append(evt)
        
        bar_time = pd.to_datetime(dp.current_time_ms, unit='ms', utc=True)
        executor.current_bar_time = bar_time
        cooldown_manager.set_virtual_time(bar_time)
        
        if epoch < warmup:
            continue
        
        # ── CHECK STOPS (identical to quick_diagnostic_bt) ──
        for evt in market_events:
            bar = dp.get_latest_bars(evt.symbol, n=1)
            if bar is not None and len(bar) > 0:
                # Intra-bar adverse price (SL)
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos['quantity']
                    if abs(qty) < 1e-8:
                        continue
                    pos_sym = v_key.split('_SCALPING')[0].split('_SWING')[0]
                    if pos_sym != evt.symbol:
                        continue
                    if qty > 0:
                        portfolio.update_market_price(evt.symbol, float(bar['low'][-1]))
                    else:
                        portfolio.update_market_price(evt.symbol, float(bar['high'][-1]))
                
                stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
                if stop_sigs:
                    for sig in stop_sigs:
                        order = rm.generate_order(sig, float(bar['close'][-1]))
                        if order:
                            fill = executor.execute(order, float(bar['close'][-1]))
                            if fill:
                                portfolio.update_fill(fill)
                                fills_total += 1
                
                # Intra-bar favorable price (TP)
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos['quantity']
                    if abs(qty) < 1e-8:
                        continue
                    pos_sym = v_key.split('_SCALPING')[0].split('_SWING')[0]
                    if pos_sym != evt.symbol:
                        continue
                    if qty > 0:
                        portfolio.update_market_price(evt.symbol, float(bar['high'][-1]))
                    else:
                        portfolio.update_market_price(evt.symbol, float(bar['low'][-1]))
                
                stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
                if stop_sigs:
                    for sig in stop_sigs:
                        order = rm.generate_order(sig, float(bar['close'][-1]))
                        if order:
                            fill = executor.execute(order, float(bar['close'][-1]))
                            if fill:
                                portfolio.update_fill(fill)
                                fills_total += 1
                
                # Restore close price
                portfolio.update_market_price(evt.symbol, evt.close_price)
            
            # Check at close
            stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
            if stop_sigs:
                for sig in stop_sigs:
                    order = rm.generate_order(sig, evt.close_price)
                    if order:
                        fill = executor.execute(order, evt.close_price)
                        if fill:
                            portfolio.update_fill(fill)
                            fills_total += 1
        
        # ── GENERATE SIGNALS (identical to quick_diagnostic_bt) ──
        try:
            class DummyEvent:
                pass
            dummy = DummyEvent()
            if bar_time:
                dummy.timestamp = bar_time
            tech.generate_signals(event=dummy)
        except:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
        
        # ── PROCESS SIGNALS → ORDERS → FILLS ──
        while not events_queue.empty():
            evt = events_queue.get()
            etype = evt.type.name if hasattr(evt.type, 'name') else str(evt.type)
            
            if etype == 'SIGNAL':
                signals_total += 1
                price = dp.get_latest_price(evt.symbol)
                if not price:
                    continue
                
                # Suppress stdout from generate_order
                capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(capture):
                        order = rm.generate_order(evt, price)
                except:
                    order = None
                
                if order is None:
                    continue
                
                fill = executor.execute(order, price)
                if fill:
                    portfolio.update_fill(fill)
                    fills_total += 1
    
    # ── CLOSE REMAINING POSITIONS ──
    for v_key, vpos in list(portfolio.virtual_ledger.items()):
        qty = vpos['quantity']
        if qty == 0:
            continue
        horizon = vpos['horizon']
        parts = v_key.rsplit(f'_{horizon}', 1)
        symbol = parts[0] if len(parts) > 1 else v_key
        price = dp.get_latest_price(symbol)
        if not price:
            continue
        direction = OrderSide.SELL if qty > 0 else OrderSide.BUY
        close_fill = FillEvent(
            timeindex=bar_time if bar_time else datetime.now(timezone.utc),
            symbol=symbol,
            exchange='BT_EVO',
            quantity=abs(qty),
            direction=direction,
            fill_cost=abs(qty) * price,
            commission=abs(qty) * price * COMMISSION_MAKER,
            strategy_id='EVO_CLOSE',
            fill_price=price,
            horizon=horizon,
            metadata={'is_close': True}
        )
        portfolio.update_fill(close_fill)
    
    # ── RESULTS ──
    trades = len(portfolio.trade_history)
    wins = sum(1 for t in portfolio.trade_history if t['net_pnl'] > 0)
    wr = (wins / trades * 100) if trades > 0 else 0
    net_pnl = sum(t['net_pnl'] for t in portfolio.trade_history)
    eq = portfolio.get_total_equity()
    
    return {
        'trades': trades,
        'wins': wins,
        'wr': wr,
        'net_pnl': net_pnl,
        'equity': eq,
        'signals': signals_total,
        'fills': fills_total,
    }


def objective(trial):
    """Optuna objective function: maximize score = f(WR, trades, PnL)."""
    
    # ── MUTATE HYPERPARAMETERS ──
    mutations = {
        'min_atr_required': trial.suggest_float("min_atr", 0.0003, 0.0020, log=True),
        'adx_threshold': trial.suggest_int("adx", 8, 30),
        'strength_threshold': trial.suggest_float("strength", 0.25, 0.65),
        'max_tp_cap': trial.suggest_float("max_tp", 0.0008, 0.0040),
        'sl_multiplier': trial.suggest_float("sl_mult", 0.5, 3.0),
    }
    
    # ── RUN BACKTEST ──
    result = run_single_backtest(mutations)
    
    trades = result['trades']
    wr = result['wr']
    net_pnl = result['net_pnl']
    
    # Store for reporting
    trial.set_user_attr("trades", trades)
    trial.set_user_attr("wr", wr)
    trial.set_user_attr("net_pnl", net_pnl)
    trial.set_user_attr("equity", result['equity'])
    trial.set_user_attr("signals", result['signals'])
    trial.set_user_attr("fills", result['fills'])
    
    # ── FITNESS FUNCTION ──
    if trades < 3:
        return -1000.0 + trades
    if wr < 40.0:
        return -500.0 + wr
    if net_pnl <= -0.05:
        return -100.0 + (net_pnl * 10)
    
    # Composite score: WR is king, then PnL, then trade frequency
    score = (wr * 50) + (trades * 2) + (net_pnl * 500)
    return score


if __name__ == "__main__":
    print("=" * 60)
    print("🧬 HYPER-EVOLVER: Optuna Parameter Optimization")
    print(f"   Symbols: {symbols}")
    print(f"   Bars: {n_bars}")
    print(f"   Capital: $13.00")
    print("=" * 60)
    
    study = optuna.create_study(direction="maximize")
    
    # Callback to print progress
    def trial_callback(study, trial):
        attrs = trial.user_attrs
        t = attrs['trades']
        w = attrs['wr']
        p = attrs['net_pnl']
        s = attrs['signals']
        print(f"  Trial {trial.number:3d} | Score: {trial.value or -9999:8.1f} | "
              f"Trades: {t:3d} | WR: {w:5.1f}% | PnL: ${p:+.4f} | Signals: {s}")
    
    print("\nStarting evolution (50 trials)...\n")
    study.optimize(objective, n_trials=50, n_jobs=1, callbacks=[trial_callback])
    
    print("\n" + "=" * 60)
    print("🏆 BEST GENOTYPE FOUND:")
    best = study.best_trial
    print(f"  Score: {best.value:.1f}")
    print(f"  Trades: {best.user_attrs['trades']} | WR: {best.user_attrs['wr']:.1f}% | "
          f"Net PnL: ${best.user_attrs['net_pnl']:.4f}")
    print(f"  Equity: ${best.user_attrs['equity']:.2f}")
    print("  Parameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    print("=" * 60)
    
    # Show top 5 trials
    print("\n📊 TOP 5 TRIALS:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value or -9999, reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        attrs = t.user_attrs
        print(f"  #{i+1} | Score: {t.value or -9999:8.1f} | "
              f"Trades: {attrs['trades']:3d} | WR: {attrs['wr']:5.1f}% | "
              f"PnL: ${attrs['net_pnl']:+.4f}")
