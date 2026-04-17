"""
🔬 OPTIMIZADOR ITERATIVO - Trader Gemini
Aplica optimizaciones una por una y mide el impacto en PnL.
Usa backtest de 1 día (rápido) para iteración veloz.
"""
import sys, os, io, contextlib, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_infra import fetch_binance_data, calculate_metrics, BacktestPortfolio, INITIAL_CAPITAL, LEVERAGE
import core.backtest_infra as bt_module

print("="*60)
print("🔬 OPTIMIZADOR ITERATIVO - BUSCANDO RENDIMIENTO MÁXIMO")
print("="*60)

# Download data once
symbol = 'BTC/USDT'
data_1d = fetch_binance_data(symbol, days=1)
data_15d = fetch_binance_data(symbol, days=15)

results_log = []

def run_test(label, days_key='1d'):
    """Ejecuta backtest y captura métricas"""
    data = data_1d if days_key == '1d' else data_15d
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = run_backtest(data, symbol)
    p = res['portfolio']
    m = calculate_metrics(p)
    pnl = p.current_capital - p.initial_capital
    entry = {
        'label': label,
        'days': days_key,
        'pnl': pnl,
        'return_pct': m['total_return'],
        'sharpe': m['sharpe_ratio'],
        'win_rate': m['win_rate'],
        'max_dd': m['max_drawdown_pct'],
        'trades': m['total_trades'],
        'capital_final': p.current_capital
    }
    results_log.append(entry)
    print(f"\n{'='*55}")
    print(f"📊 {label}")
    print(f"{'='*55}")
    print(f"   PnL:        ${pnl:+.4f} ({m['total_return']:+.2f}%)")
    print(f"   Sharpe:     {m['sharpe_ratio']:.2f}")
    print(f"   Win Rate:   {m['win_rate']:.1f}%")
    print(f"   Max DD:     {m['max_drawdown_pct']:.2f}%")
    print(f"   Trades:     {m['total_trades']}")
    print(f"   Capital:    ${p.current_capital:.4f}")
    return entry

# ============================================================
# FASE 0: BASELINE (estado actual)
# ============================================================
print("\n⏳ FASE 0: BASELINE (configuración actual)...")
original_commission = bt_module.COMMISSION_PCT
baseline = run_test("BASELINE (actual)", '1d')
baseline_15d = run_test("BASELINE 15D", '15d')

# ============================================================
# FASE 1: FILTRO DE CONFIANZA (P(Win) > 60%)
# ============================================================
print("\n⏳ FASE 1: Aplicando filtro de confianza mínima P(Win) > 60%...")

# Monkey-patch the signal processing in run_backtest
# We need to modify the signal entry logic to check confidence
# The signal's strength attribute contains SOPHIA's confidence
original_run_backtest = bt_module.run_backtest.__code__

# Instead of monkey-patching complex code, we'll modify the constants
# and re-import. Let's use a simpler approach: modify the event processing
# by adding a confidence filter wrapper.

# Save original
import types

_orig_run = bt_module.run_backtest

def run_backtest_with_conf_filter(data, symbol='BTC/USDT', min_confidence=0.60):
    """Wrapper that filters low-confidence signals"""
    from queue import Queue
    from core.backtest_infra import (BacktestDataProvider, BacktestPortfolio, 
                                     INITIAL_CAPITAL, LEVERAGE, COMMISSION_PCT)
    from strategies.technical import HybridScalpingStrategy
    from core.events import MarketEvent, SignalEvent
    from core.enums import SignalType
    from config import Config
    import pandas as pd
    
    events_queue = Queue()
    historical_data = {symbol: data}
    data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
    portfolio = BacktestPortfolio(INITIAL_CAPITAL, LEVERAGE)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    
    warmup_bars = 100
    signals_generated = 0
    trades_executed = 0
    signals_filtered = 0
    bar_count = 0
    total_bars = len(data)
    
    while data_provider.continue_backtest:
        data_provider.update_bars()
        bar_count += 1
        if bar_count < warmup_bars:
            continue
        
        bars = data_provider.get_latest_bars(symbol, 1)
        if not bars:
            continue
        
        current_bar = bars[-1]
        current_price = current_bar['close']
        current_time = pd.to_datetime(current_bar['timestamp'], unit='ms', utc=True)
        high = current_bar['high']
        low = current_bar['low']
        
        # Check SL/TP
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            entry = pos['entry']
            side = pos['side']
            stored_sl = pos.get('sl_price')
            stored_tp = pos.get('tp_price')
            
            if stored_sl is None:
                if side == 'LONG': stored_sl = entry * 0.985
                else: stored_sl = entry * 1.015
            if stored_tp is None:
                if side == 'LONG': stored_tp = entry * 1.01
                else: stored_tp = entry * 0.99
            
            # TRAILING STOP: Move SL to breakeven when profit > 0.5%
            if pos.get('trailing_active', False) == False:
                if side == 'LONG' and current_price > entry * 1.005:
                    stored_sl = entry * 1.001  # Breakeven + tiny buffer
                    portfolio.positions[symbol]['sl_price'] = stored_sl
                    portfolio.positions[symbol]['trailing_active'] = True
                elif side == 'SHORT' and current_price < entry * 0.995:
                    stored_sl = entry * 0.999
                    portfolio.positions[symbol]['sl_price'] = stored_sl
                    portfolio.positions[symbol]['trailing_active'] = True
            
            if side == 'LONG':
                if low <= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time)
                    if trade:
                        trades_executed += 1
                        if hasattr(strategy, 'process_reward'): strategy.process_reward(trade)
                elif high >= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time)
                    if trade:
                        trades_executed += 1
                        if hasattr(strategy, 'process_reward'): strategy.process_reward(trade)
            else:
                if high >= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time)
                    if trade:
                        trades_executed += 1
                        if hasattr(strategy, 'process_reward'): strategy.process_reward(trade)
                elif low <= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time)
                    if trade:
                        trades_executed += 1
                        if hasattr(strategy, 'process_reward'): strategy.process_reward(trade)
        
        # Generate signals
        strategy.bought[symbol] = symbol in portfolio.positions
        market_event = MarketEvent(symbol=symbol, close_price=current_price, timestamp=current_time)
        strategy.calculate_signals(market_event)
        
        while not events_queue.empty():
            event = events_queue.get()
            if not isinstance(event, SignalEvent):
                continue
            
            if event.signal_type == SignalType.EXIT:
                if symbol in portfolio.positions:
                    trade = portfolio.close_position(symbol, current_price, current_time)
                    if trade:
                        trades_executed += 1
                        if hasattr(strategy, 'process_reward'): strategy.process_reward(trade)
                continue
            
            if symbol not in portfolio.positions:
                # === CONFIDENCE FILTER ===
                signal_strength = getattr(event, 'strength', 0.5)
                meta_dict = event.metadata if event.metadata else {}
                sophia_conf = meta_dict.get('sophia_confidence', signal_strength)
                
                if sophia_conf < min_confidence:
                    signals_filtered += 1
                    continue
                
                signals_generated += 1
                
                metadata = {
                    'atr': getattr(event, 'atr', 0.0),
                    'confluence': meta_dict.get('multi_timeframe_score', 0.0)
                }
                
                peak = portfolio.peak_equity
                current_cap = portfolio.current_capital
                initial = portfolio.initial_capital
                dd = (peak - current_cap) / peak if peak > 0 else 0
                
                risk_pct = getattr(Config, 'MAX_RISK_PER_TRADE', 0.05)
                if dd > 0.05: risk_pct *= 0.5
                if dd > 0.10: risk_pct *= 0.25
                if peak >= (initial * 2.0): risk_pct *= 0.50
                elif peak >= (initial * 1.5): risk_pct *= 0.75
                
                risk_usd = current_cap * risk_pct
                
                raw_sl_pct = getattr(event, 'sl_pct', 1.5)
                sl_decimal = raw_sl_pct / 100.0 if raw_sl_pct > 0.1 else raw_sl_pct
                tp_pct = getattr(event, 'tp_pct', 2.0)
                
                size_usd = (risk_usd / sl_decimal) if sl_decimal > 0 else (current_cap * 0.1)
                max_size = current_cap * 10
                size_usd = min(size_usd, max_size)
                if size_usd < 5.0:
                    size_usd = 5.0
                
                side = 'LONG' if event.signal_type == SignalType.LONG else 'SHORT'
                tp_decimal = tp_pct / 100.0 if tp_pct > 0.1 else tp_pct
                
                if side == 'LONG':
                    entry_sl = current_price * (1 - sl_decimal)
                    entry_tp = current_price * (1 + tp_decimal)
                else:
                    entry_sl = current_price * (1 + sl_decimal)
                    entry_tp = current_price * (1 - tp_decimal)
                
                opened = portfolio.open_position_with_metadata(
                    symbol, side, current_price, size_usd, current_time, metadata, entry_sl, entry_tp
                )
                if opened:
                    trades_executed += 1
        
        if bar_count % 60 == 0:
            portfolio.update_equity(current_time)
    
    # Close remaining
    for sym in list(portfolio.positions.keys()):
        bars = data_provider.get_latest_bars(sym, 1)
        if bars is not None and len(bars) > 0:
            ts_ms = bars[-1]['timestamp']
            dt_close = pd.to_datetime(ts_ms, unit='ms', utc=True)
            trade = portfolio.close_position(sym, bars[-1]['close'], dt_close)
            if trade:
                trades_executed += 1
    
    return {
        'portfolio': portfolio,
        'trades_executed': trades_executed,
        'signals_generated': signals_generated,
        'signals_filtered': signals_filtered
    }


def run_optimized_test(label, days_key, min_conf=0.60, sl_pct_override=None, tp_pct_override=None, commission_override=None, trailing=False):
    """Run with specific optimizations"""
    data = data_1d if days_key == '1d' else data_15d
    
    # Override commission if requested
    if commission_override is not None:
        bt_module.COMMISSION_PCT = commission_override
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = run_backtest_with_conf_filter(data, symbol, min_confidence=min_conf)
    
    p = res['portfolio']
    m = calculate_metrics(p)
    pnl = p.current_capital - p.initial_capital
    
    # Restore commission
    bt_module.COMMISSION_PCT = original_commission
    
    entry = {
        'label': label,
        'days': days_key,
        'pnl': pnl,
        'return_pct': m['total_return'],
        'sharpe': m['sharpe_ratio'],
        'win_rate': m['win_rate'],
        'max_dd': m['max_drawdown_pct'],
        'trades': m['total_trades'],
        'signals_filtered': res.get('signals_filtered', 0),
        'capital_final': p.current_capital
    }
    results_log.append(entry)
    
    filtered = res.get('signals_filtered', 0)
    print(f"\n{'='*55}")
    print(f"📊 {label}")
    print(f"{'='*55}")
    print(f"   PnL:          ${pnl:+.4f} ({m['total_return']:+.2f}%)")
    print(f"   Sharpe:       {m['sharpe_ratio']:.2f}")
    print(f"   Win Rate:     {m['win_rate']:.1f}%")
    print(f"   Max DD:       {m['max_drawdown_pct']:.2f}%")
    print(f"   Trades:       {m['total_trades']}")
    print(f"   Filtered:     {filtered}")
    print(f"   Capital:      ${p.current_capital:.4f}")
    return entry

# ============================================================
# FASE 1: Confidence Filter (P(Win) > 55%, 60%, 65%, 70%)
# ============================================================
print("\n" + "="*60)
print("🧪 FASE 1: BARRIDO DE UMBRALES DE CONFIANZA")
print("="*60)

for conf_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    run_optimized_test(f"CONF>{conf_threshold*100:.0f}% (1D)", '1d', min_conf=conf_threshold)

# Run best on 15D
best_conf_1d = max([r for r in results_log if 'CONF>' in r['label'] and r['days'] == '1d'], key=lambda x: x['pnl'])
best_conf = float(best_conf_1d['label'].split('>')[1].split('%')[0]) / 100.0
print(f"\n🏆 Mejor umbral 1D: {best_conf*100:.0f}% → PnL: ${best_conf_1d['pnl']:+.4f}")

run_optimized_test(f"CONF>{best_conf*100:.0f}% (15D)", '15d', min_conf=best_conf)

# ============================================================
# FASE 2: Con Maker Fee (0.02% en vez de 0.04%)
# ============================================================
print("\n" + "="*60)
print("🧪 FASE 2: MAKER FEE (0.02%)")
print("="*60)

bt_module.COMMISSION_PCT = 0.0002  # Maker fee
run_optimized_test(f"CONF>{best_conf*100:.0f}% + MAKER_FEE (1D)", '1d', min_conf=best_conf)
run_optimized_test(f"CONF>{best_conf*100:.0f}% + MAKER_FEE (15D)", '15d', min_conf=best_conf)
bt_module.COMMISSION_PCT = original_commission

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*60)
print("🏆 TABLA COMPARATIVA FINAL")
print("="*60)
print(f"{'Optimización':<40} {'PnL':>10} {'WR':>7} {'Trades':>7} {'Sharpe':>8} {'MaxDD':>8}")
print("-"*80)
for r in results_log:
    tag = f"{r['label']}"
    print(f"{tag:<40} ${r['pnl']:>+8.4f} {r['win_rate']:>5.1f}% {r['trades']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.2f}%")

print("\n✅ OPTIMIZACIÓN ITERATIVA COMPLETADA")
