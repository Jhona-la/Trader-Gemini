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
from datetime import datetime, timezone
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# Remove stale lock
lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'STOP_TRADING.LOCK')
if os.path.exists(lock_file):
    os.remove(lock_file)
    print("Removed STOP_TRADING.LOCK")

# Download 1 day BTC
print("Downloading 1 day BTC/USDT...")
df = fetch_binance_data('BTC/USDT', days=1)
print(f"Got {len(df)} bars")

symbols = ['BTC/USDT']
all_data = {'BTC/USDT': df}
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
rm = RiskManager(max_concurrent_positions=2, portfolio=portfolio)

from strategies.technical import HybridScalpingStrategy
tech = HybridScalpingStrategy(dp, events_queue, horizon='SCALPING')

# Simple BacktestExecutor
class SimpleExecutor:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self.fills = 0
    
    def execute(self, order, price):
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
        
        return FillEvent(
            timeindex=datetime.now(timezone.utc),
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
    
    for evt in market_events:
        portfolio.update_market_price(evt.symbol, evt.close_price)
    
    if epoch < warmup:
        continue
    
    # Check stops
    try:
        stop_sigs = rm.check_stops(portfolio, dp)
        if stop_sigs:
            for sig in stop_sigs:
                events_queue.put(sig)
    except:
        pass
    
    # Run technical strategy
    try:
        tech.generate_signals()
    except:
        pass
    
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
            except:
                order = None
            
            if order is None:
                rejected_total += 1
                captured = capture.getvalue()
                reason = 'UNKNOWN'
                for line in captured.strip().split('\n'):
                    if '[RISK] Rejected' in line:
                        reason = line.strip()
                        break
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            
            fill = executor.execute(order, price)
            if fill:
                portfolio.update_fill(fill)
                fills_total += 1
    
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
    print(f"\n  TRADE BREAKDOWN:")
    print(f"    Total: {total_trades} | Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")
    total_pnl = sum(t['net_pnl'] for t in portfolio.trade_history)
    total_fees = sum(t['fees_paid'] for t in portfolio.trade_history)
    print(f"    Net PnL: ${total_pnl:.4f} | Fees: ${total_fees:.4f}")
    for t in portfolio.trade_history[:5]:
        print(f"    {t['symbol']} {t['direction']} | Net: ${t['net_pnl']:.4f} | Dur: {t['duration_seconds']}s | Exit: {t['exit_reason']}")

print(f"{'='*60}")
