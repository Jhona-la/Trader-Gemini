#!/usr/bin/env python3
"""Quick diagnostic backtest — Technical Strategy only, 1 day, 1 symbol."""
import os, sys, time, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'

from config import Config
from core.backtest_infra import fetch_binance_data, BacktestDataProvider, COMMISSION_PCT, COMMISSION_MAKER
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from queue import Queue
from utils.cooldown_manager import cooldown_manager
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import random

Config.TELEGRAM_ENABLED = False
Config.EMAIL_ENABLED = False
Config.DISCORD_ENABLED = False

# Silence Observability layer notifications (the real gate for Notifier)
if hasattr(Config, 'Observability'):
    Config.Observability.TELEGRAM_ENABLED = False
    Config.Observability.DISCORD_ENABLED = False
    Config.Observability.EMAIL_ENABLED = False

random.seed(42)
np.random.seed(42)

# Remove stale lock
lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'STOP_TRADING.LOCK')
if os.path.exists(lock_file):
    os.remove(lock_file)
    print("Removed STOP_TRADING.LOCK")

from core.backtest_infra import fetch_multi_symbol_data

# Download 1 day data for top 5 symbols
print("Downloading 1 day Top 5 symbols...")
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
all_data = fetch_multi_symbol_data(symbols, days=1)
events_queue = Queue()
dp = BacktestDataProvider(events_queue, symbols, all_data)

bt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'data', 'diag_temp')
os.makedirs(bt_dir, exist_ok=True)

portfolio = Portfolio(
    initial_capital=13.0,
    csv_path=os.path.join(bt_dir, 'diag_trades.csv'),
    status_path=os.path.join(bt_dir, 'diag_status.csv'),
    auto_save=False
)
portfolio.data_provider = dp
from core.strategy_registry import UniversalStrategyRegistry
from core.engine import Engine

registry = UniversalStrategyRegistry()
# Simulate loading them via Engine
class MockRiskManager:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.max_concurrent_positions = 5
        self.prediction_tracker = None
        self.kill_switch = None
        # Must have evaluate to bypass actual generating orders if we don't want to test execution?
        # Let's just use the real RiskManager.
        
rm = RiskManager(max_concurrent_positions=5, portfolio=portfolio)

engine = Engine(events_queue=events_queue)
engine.portfolio = portfolio
engine.risk_manager = rm
engine.data_handlers = [dp] # data_provider is accessed through portfolio/handlers mostly, but engine holds references too
# Force discover and load:
engine.strategies = []
engine._active_strategies = {'ALL': []}

from strategies.technical import HybridScalpingStrategy
from strategies.swing_motor import SwingMotor

tech = HybridScalpingStrategy(dp, events_queue, horizon='SCALPING')
swing = SwingMotor(dp, events_queue)

engine.strategies.append(tech)
engine.strategies.append(swing)
engine._active_strategies['ALL'] = [tech, swing]


# Simple BacktestExecutor
class SimpleExecutor:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self.fills = 0
        self.current_bar_time = datetime.now(timezone.utc)  # Will be updated per epoch
    
    def execute(self, order, price):
        # FORENSIC-V60 FIX: Ignore resting limit orders in naive backtest executor
        # QUÉ: Prevent immediate market execution of resting TP limits.
        # POR QUÉ: SimpleExecutor has no orderbook. When PREDICTIVE_TP was placed,
        #   the executor immediately filled it at `price` (current market price),
        #   closing the trade at breakeven/loss instantly.
        # PARA QUÉ: Rely on risk_manager.py `check_stops` fallback to trigger
        #   TAKE_PROFIT when price actually hits the target.
        if getattr(order, 'strategy_id', '') in ('PREDICTIVE_TP', 'PLACE_TP_LIMIT', 'PLACE_SL_LIMIT'):
            return None
            
        slip = self._rng.uniform(0.0, 0.0002)
        if order.direction == OrderSide.BUY:
            fp = price * (1 + slip)
        else:
            fp = price * (1 - slip)
        qty = order.quantity
        fc = fp * qty
        comm = fc * COMMISSION_MAKER
        self.fills += 1
        
        meta = order.metadata.copy() if order.metadata else {}
        meta['actual_order_type'] = 'limit'
        meta['is_close'] = getattr(order, 'is_close', False)
        meta['is_exit'] = getattr(order, 'is_exit', False)
        
        return FillEvent(
            timeindex=self.current_bar_time,  # P2-FIX #9: Use candle timestamp, not wall-clock
            symbol=order.symbol,
            exchange='BT_DIAG',
            quantity=qty,
            direction=order.direction,
            fill_cost=fc,
            commission=comm,
            strategy_id=order.strategy_id,
            fill_price=fp,
            order_id=f'DIAG_{self.fills}',
            sl_pct=order.sl_pct,
            tp_pct=order.tp_pct,
            horizon=order.horizon,
            leverage=order.leverage,
            metadata=meta,
        )

executor = SimpleExecutor()

total = dp.total_epochs
warmup = min(100, total // 20)
signals_total = 0
fills_total = 0
rejected_total = 0
rejection_reasons = {}

t_start = time.time()
print(f"Total epochs: {total} | Warmup: {warmup}")
print(f"Starting simulation...")

for epoch in range(total):
    dp.update_bars()
    
    # Drain market events
    market_events = []
    while not events_queue.empty():
        evt = events_queue.get()
        if evt.type == EventType.MARKET:
            market_events.append(evt)
    
    # P2-FIX #9: Track the bar timestamp for time-based logic fidelity
    bar_time = pd.to_datetime(dp.current_time_ms, unit='ms', utc=True)
    
    for evt in market_events:
        portfolio.update_market_price(evt.symbol, evt.close_price)

    if bar_time:
        executor.current_bar_time = bar_time
        cooldown_manager.set_virtual_time(bar_time)
    
    if epoch < warmup:
        continue
    
    # Check stops with candle timestamp and intra-bar evaluation
    # FORENSIC-V81: Process EXIT signals IMMEDIATELY (not batched)
    # QUÉ: Exit signals from check_stops are now executed inline.
    # POR QUÉ: Batching them to the queue caused re-entry loops:
    #   check_stops → EXIT signal → queue → strategy generates RE-ENTRY → drain processes both
    #   → net result: position closes then immediately re-opens → infinite zombie cycle.
    # PARA QUÉ: Guarantee position closure before any new entry signal is generated.
    def _process_exit_signals_immediately(stop_sigs, price):
        """Execute EXIT signals inline without going through the event queue."""
        if not stop_sigs:
            return
        for sig in stop_sigs:
            strategy_id = getattr(sig, 'strategy_id', '')
            sym = getattr(sig, 'symbol', '')
            hor = getattr(sig, 'horizon', 'SCALPING')
            
            # Pre-check: does position exist?
            pre_pos = portfolio.get_horizon_position(sym, hor)
            pre_qty = pre_pos.get('quantity', 0) if pre_pos else 0
            
            order = rm.generate_order(sig, price)
            if order:
                fill = executor.execute(order, price)
                if fill:
                    portfolio.update_fill(fill)
                    
                    # Post-check: did position close?
                    post_pos = portfolio.get_horizon_position(sym, hor)
                    post_qty = post_pos.get('quantity', 0) if post_pos else 0
                    
                    if strategy_id == 'TIME_STOP_ZOMBIE':
                        print(f"  [DEBUG-ZOMBIE] {sym} {hor} pre={pre_qty:.8f} post={post_qty:.8f} close?={fill.metadata.get('is_close', False) if hasattr(fill, 'metadata') and fill.metadata else getattr(fill, 'is_close', False)}")
                    
                    if strategy_id == 'TIME_STOP_ZOMBIE' and abs(post_qty) > 1e-8:
                        print(f"  ⚠️ ZOMBIE EXIT FAILED: {sym} {hor} | pre_qty={pre_qty:.8f} -> post_qty={post_qty:.8f} | order.dir={order.direction} order.qty={order.quantity:.8f}")
            elif strategy_id == 'TIME_STOP_ZOMBIE':
                print(f"  ⚠️ ZOMBIE ORDER=None: {sym} {hor} | pre_qty={pre_qty:.8f} | pos_exists={pre_pos is not None}")

    
    try:
        for evt in market_events:
            bar = dp.get_latest_bars(evt.symbol, n=1)
            if bar is not None and len(bar) > 0:
                # --- INTRA-BAR ADVERSE PRICE (SL check) ---
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos.get('quantity', 0)
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
                _process_exit_signals_immediately(stop_sigs, float(bar['close'][-1]))
                
                # --- INTRA-BAR FAVORABLE PRICE (TP check) ---
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos.get('quantity', 0)
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
                _process_exit_signals_immediately(stop_sigs, float(bar['close'][-1]))
                
                # Restore close price
                portfolio.update_market_price(evt.symbol, evt.close_price)
            
            # Also check at close
            stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
            _process_exit_signals_immediately(stop_sigs, evt.close_price)
        
        # FORENSIC-V81: Global check_stops (no symbol_filter) to catch zombies
        # on symbols that have no MarketEvent in this epoch.
        # QUÉ: Evalúa TODAS las posiciones abiertas, incluso si su símbolo
        #   no tiene datos de mercado en este epoch.
        # POR QUÉ: SOL/BNB zombies vivían 11+ horas porque solo se evaluaban
        #   cuando tenían market events, pero la timeline global avanzaba sin ellos.
        if bar_time:
            global_stops = rm.check_stops(portfolio, dp, now=bar_time)
            if global_stops:
                for sig in global_stops:
                    price = dp.get_latest_price(sig.symbol) or 0
                    if price:
                        _process_exit_signals_immediately([sig], price)
    except Exception as e:
        print(f"[WARN] check_stops error: {e}")


    # Dispatch MarketEvents through the engine routing
    for evt in market_events:
        try:
            engine._process_market_event(evt)
        except Exception as e:
            print(f"[WARN] Engine processing error: {e}")
    
    # Drain and process signals
    while not events_queue.empty():
        evt = events_queue.get()
        etype = evt.type.name if hasattr(evt.type, 'name') else str(evt.type)
        
        if etype == 'SIGNAL':
            signals_total += 1
            price = dp.get_latest_price(evt.symbol)
            if not price:
                continue
            
            capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(capture):
                    order = rm.generate_order(evt, price)
            except Exception as e:
                import traceback
                import sys
                print(f"[FATAL ERROR] generate_order threw exception: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                order = None
                if evt.metadata is None:
                    object.__setattr__(evt, 'metadata', {})
                evt.metadata["rejection_reason"] = f"FATAL_ERROR:{type(e).__name__}"
            
            if order is None:
                rejected_total += 1
                reason = evt.metadata.get("rejection_reason", "UNKNOWN")
                if hasattr(reason, 'name'):
                    reason = reason.name
                else:
                    reason = str(reason)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            
            orders = order if isinstance(order, list) else [order]
            
            for ord in orders:
                fill = executor.execute(ord, price)
                if fill:
                    portfolio.update_fill(fill)
                    fills_total += 1
                    # Handle flip/exit
                    if ord.direction == OrderSide.SELL and getattr(ord, 'setup_type', '') == 'FLIP_EXIT':
                        # Simplistic PNL calc
                        pass
                    # Just count fills for now.
    
    # Progress
    if epoch % 300 == 0:
        elapsed = time.time() - t_start
        eq = portfolio.get_total_equity()
        open_pos = sum(1 for v in portfolio.virtual_ledger.values() if v.get('quantity', 0) != 0)
        print(f"  [{epoch}/{total}] Equity: ${eq:.2f} | Sig: {signals_total} | Fill: {fills_total} | Rej: {rejected_total} | Open: {open_pos} | {elapsed:.1f}s")

# Close remaining
for v_key, vpos in list(portfolio.virtual_ledger.items()):
    qty = vpos.get('quantity', 0)
    if qty == 0:
        continue
    horizon = vpos.get('horizon', 'SCALPING')
    parts = v_key.rsplit(f'_{horizon}', 1)
    symbol = parts[0] if len(parts) > 1 else v_key
    price = dp.get_latest_price(symbol)
    if not price:
        continue
    direction = OrderSide.SELL if qty > 0 else OrderSide.BUY
    close_fill = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange='BT_DIAG',
        quantity=abs(qty),
        direction=direction,
        fill_cost=abs(qty) * price,
        commission=abs(qty) * price * COMMISSION_MAKER,
        strategy_id='DIAG_CLOSE',
        fill_price=price,
        horizon=horizon,
        metadata={'is_close': True}
    )
    portfolio.update_fill(close_fill)
    fills_total += 1

elapsed = time.time() - t_start
eq = portfolio.get_total_equity()
ret = (eq - 13.0) / 13.0 * 100

print(f"\n{'='*60}")
print(f"DIAGNOSTIC RESULTS")
print(f"{'='*60}")
print(f"  Duration: {elapsed:.1f}s")
print(f"  Epochs: {total}")
print(f"  Signals: {signals_total}")
print(f"  Fills: {fills_total}")
print(f"  Rejected: {rejected_total}")
print(f"  Final Equity: ${eq:.2f}")
print(f"  Return: {ret:.2f}%")

if rejection_reasons:
    print(f"\n  REJECTION BREAKDOWN:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {reason}: {count}")

# Trade breakdown
if portfolio.trade_history:
    wins = sum(1 for t in portfolio.trade_history if t['net_pnl'] > 0)
    losses = sum(1 for t in portfolio.trade_history if t['net_pnl'] <= 0)
    total_trades = len(portfolio.trade_history)
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t['net_pnl'] for t in portfolio.trade_history)
    total_fees = sum(t['fees_paid'] for t in portfolio.trade_history)
    
    print(f"\n  TRADE BREAKDOWN:")
    print(f"    Total: {total_trades} | Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")
    print(f"    Net PnL: ${total_pnl:.4f} | Fees: ${total_fees:.4f}")
    
    # Aggregate exit reasons
    exit_counts = {}
    for t in portfolio.trade_history:
        reason = t.get('exit_reason', 'UNKNOWN')
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
        
    print(f"    EXIT REASONS:")
    for reason, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {reason}: {count}")

    print(f"\n  LOSING TRADES ANATOMY (Top 10 Worst):")
    losing_trades = sorted([t for t in portfolio.trade_history if t['net_pnl'] < 0], key=lambda x: x['net_pnl'])
    for idx, t in enumerate(losing_trades[:10]):
        print(f"    [{idx+1}] {t['direction']} {t['symbol']} | Reason: {t['exit_reason']}")
        print(f"        Entry: ${t['entry_price']:.2f} | Exit: ${t['exit_price']:.2f} | Setup: {t['setup_type']}")
        print(f"        Gross PnL: ${t['gross_pnl']:.4f} | Fees: ${t['fees_paid']:.4f} | Net: ${t['net_pnl']:.4f}")
        print(f"        Duration: {t['duration_seconds']}s")
    
    print(f"\n  WINNING TRADES ANATOMY (Top 5 Best):")
    winning_trades = sorted([t for t in portfolio.trade_history if t['net_pnl'] > 0], key=lambda x: -x['net_pnl'])
    for idx, t in enumerate(winning_trades[:5]):
        print(f"    [{idx+1}] {t['direction']} {t['symbol']} | Reason: {t['exit_reason']}")
        print(f"        Entry: ${t['entry_price']:.2f} | Exit: ${t['exit_price']:.2f} | Setup: {t['setup_type']}")
        print(f"        Gross PnL: ${t['gross_pnl']:.4f} | Fees: ${t['fees_paid']:.4f} | Net: ${t['net_pnl']:.4f}")
        print(f"        Duration: {t['duration_seconds']}s")

print(f"{'='*60}")
