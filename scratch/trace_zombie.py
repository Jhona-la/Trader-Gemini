"""
FORENSIC-V81: Trace a single zombie trade lifecycle to find the EXIT failure point.
Runs the same backtest but prints detailed tracing for the FIRST zombie exit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib, io
from datetime import datetime, timezone
from queue import Queue

from config import Config
Config.BINANCE_USE_FUTURES = True
Config.BINANCE_LEVERAGE = 10

from core.events import SignalEvent
from core.enums import SignalType, EventType
from core.backtest_infra import BacktestDataProvider, BacktestPortfolio, SimpleExecutor
from strategies.technical import HybridScalpingStrategy
from risk.risk_manager import RiskManager

# Single symbol for simplicity
SYMBOL = "ETH/USDT"
SYMBOLS = [SYMBOL]

events_queue = Queue()
dp = BacktestDataProvider(symbols=SYMBOLS, resolution="1m", lookback_bars=500, events_queue=events_queue)
portfolio = BacktestPortfolio(initial_capital=13.0, csv_path="scratch/_zombie_trades.csv", status_path="scratch/_zombie_status.csv")
portfolio.data_provider = dp
rm = RiskManager(portfolio=portfolio, data_provider=dp)
executor = SimpleExecutor(events_queue=events_queue, portfolio=portfolio)

tech = HybridScalpingStrategy(
    data_provider=dp, events_queue=events_queue,
    symbol_list=SYMBOLS, horizon="SCALPING"
)

print(f"=== ZOMBIE TRACE: {SYMBOL} ===")
print(f"TP cap in technical.py: Should be 0.20% (check line 1880)")
print(f"TP cap in risk_manager.py: Should be 0.25% (check line 2315)")
print(f"Zombie timeout: Should be 1800s (check line 2374)")
print()

zombie_traced = False
fills_total = 0
total = len(dp._global_timeline)

for epoch, bar_time in enumerate(dp._global_timeline):
    dp._current_epoch = epoch
    dp.current_time_ms = int(bar_time.timestamp() * 1000)
    market_events = dp._emit_market_events(epoch)
    
    # Skip empty
    if not market_events:
        continue
    
    # Check for open positions and trace their age
    for v_key, vpos in list(portfolio.virtual_ledger.items()):
        qty = vpos.get('quantity', 0)
        if abs(qty) < 1e-8:
            continue
        entry_time_val = vpos.get('entry_time')
        if entry_time_val and hasattr(entry_time_val, 'timestamp'):
            seconds_held = bar_time.timestamp() - entry_time_val.timestamp()
        elif entry_time_val:
            seconds_held = bar_time.timestamp() - entry_time_val
        else:
            seconds_held = 0
        
        # If this position should be zombie-killed (>1800s), trace it
        if seconds_held > 1800 and not zombie_traced:
            print(f"\n🧟 ZOMBIE DETECTED at epoch {epoch}:")
            print(f"   v_key: {v_key}")
            print(f"   qty: {qty}")
            print(f"   entry_price: {vpos.get('avg_price')}")
            print(f"   current_price: {vpos.get('current_price')}")
            print(f"   entry_time: {entry_time_val}")
            print(f"   seconds_held: {seconds_held:.0f}")
            print(f"   tp_pct: {vpos.get('tp_pct')}")
            print(f"   sl_pct: {vpos.get('sl_pct')}")
            
            # Now manually call check_stops and trace
            print(f"\n   Calling check_stops(symbol_filter={SYMBOL})...")
            stops = rm.check_stops(portfolio, dp, symbol_filter=SYMBOL, now=bar_time)
            print(f"   check_stops returned {len(stops) if stops else 0} signals")
            
            if stops:
                for sig in stops:
                    print(f"   Signal: strategy_id={sig.strategy_id}, type={sig.signal_type}, symbol={sig.symbol}, horizon={getattr(sig, 'horizon', '?')}")
                    
                    # Try generate_order
                    price = dp.get_latest_price(sig.symbol) or vpos.get('current_price', 0)
                    print(f"   Calling generate_order with price={price}...")
                    order = rm.generate_order(sig, price)
                    print(f"   generate_order returned: {order}")
                    
                    if order:
                        print(f"   Order: type={order.order_type}, direction={order.direction}, qty={order.quantity}, is_exit={getattr(order, 'is_exit', '?')}")
                        
                        # Execute
                        fill = executor.execute(order, price)
                        print(f"   executor.execute returned: {fill}")
                        
                        if fill:
                            print(f"   Fill: qty={fill.quantity}, price={fill.fill_price}")
                            portfolio.update_fill(fill)
                            print(f"   After update_fill:")
                            # Check if position is closed
                            remaining = portfolio.virtual_ledger.get(v_key, {}).get('quantity', 0)
                            print(f"   Remaining qty in {v_key}: {remaining}")
                    else:
                        # WHY was order None?
                        print(f"   ⚠️ ORDER IS NONE - checking why...")
                        pos_check = portfolio.get_horizon_position(sig.symbol, getattr(sig, 'horizon', 'SCALPING'))
                        print(f"   get_horizon_position returned: {pos_check}")
            else:
                # check_stops returned nothing - WHY?
                print(f"   ⚠️ NO STOP SIGNALS - Position should be zombie-killed but check_stops returned nothing")
                print(f"   Calling check_stops WITHOUT symbol_filter...")
                stops2 = rm.check_stops(portfolio, dp, now=bar_time)
                print(f"   Global check_stops returned {len(stops2) if stops2 else 0} signals")
                if stops2:
                    for s in stops2:
                        print(f"     Signal: {s.strategy_id} {s.symbol}")
            
            zombie_traced = True
            break
    
    # Process stops inline
    for evt in market_events:
        bar = dp.get_latest_bars(evt.symbol, n=1)
        if bar is not None and len(bar) > 0:
            stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
            if stop_sigs:
                for sig in stop_sigs:
                    order = rm.generate_order(sig, dp.get_latest_price(sig.symbol) or evt.close_price)
                    if order:
                        fill = executor.execute(order, evt.close_price)
                        if fill:
                            portfolio.update_fill(fill)
                            fills_total += 1
            portfolio.update_market_price(evt.symbol, evt.close_price)
    
    # Run strategy
    try:
        class DummyEvent: pass
        dummy = DummyEvent()
        dummy.timestamp = bar_time
        with contextlib.redirect_stdout(io.StringIO()):
            tech.generate_signals(event=dummy)
    except:
        pass
    
    # Drain queue
    while not events_queue.empty():
        evt = events_queue.get()
        etype = evt.type.name if hasattr(evt.type, 'name') else str(evt.type)
        if etype == 'SIGNAL':
            price = dp.get_latest_price(evt.symbol)
            if not price: continue
            with contextlib.redirect_stdout(io.StringIO()):
                order = rm.generate_order(evt, price)
            if order:
                fill = executor.execute(order, price)
                if fill:
                    portfolio.update_fill(fill)
                    fills_total += 1
    
    if zombie_traced:
        # Run 5 more epochs to see if position closes
        if epoch > dp._global_timeline.index(bar_time) + 5:
            break

print(f"\n=== Done. Fills: {fills_total}, Equity: ${portfolio.get_total_equity():.2f} ===")
