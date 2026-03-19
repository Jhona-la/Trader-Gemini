"""
🔬 OPTIMIZADOR V2 - Barrido de SL/TP Ratio
Modifica los defaults de SL y TP directamente en el signal processing
para encontrar el ratio óptimo.
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.run_backtest import fetch_binance_data, calculate_metrics, BacktestPortfolio, INITIAL_CAPITAL, LEVERAGE
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent, SignalEvent
from core.enums import SignalType
from config import Config
from queue import Queue
from tests.run_backtest import BacktestDataProvider
import pandas as pd
import numpy as np

COMMISSION_MAKER = 0.0002  # 0.02% Maker
COMMISSION_TAKER = 0.0004  # 0.04% Taker

print("="*60)
print("🔬 OPTIMIZADOR V2 - BARRIDO SL/TP + COMISIONES")
print("="*60)

symbol = 'BTC/USDT'
data_15d = fetch_binance_data(symbol, days=15)

results_log = []

def run_backtest_tuned(data, symbol, sl_pct=1.5, tp_pct=2.0, commission=0.0004, min_conf=0.0, use_trailing=False):
    """Backtest con parámetros de SL/TP personalizados"""
    events_queue = Queue()
    historical_data = {symbol: data}
    data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
    portfolio = BacktestPortfolio(INITIAL_CAPITAL, LEVERAGE)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    
    warmup_bars = 100
    trades_executed = 0
    signals_filtered = 0
    bar_count = 0
    
    while data_provider.continue_backtest:
        data_provider.update_bars()
        bar_count += 1
        if bar_count < warmup_bars:
            continue
        
        bars = data_provider.get_latest_bars(symbol, 1)
        if not bars:
            continue
        
        current_bar = bars[-1]
        current_price = float(current_bar['close'])
        current_time = pd.to_datetime(current_bar['timestamp'], unit='ms', utc=True)
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        
        # Check SL/TP
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            entry = pos['entry']
            side = pos['side']
            stored_sl = pos.get('sl_price')
            stored_tp = pos.get('tp_price')
            
            # Trailing stop: move SL to breakeven when +0.5%
            if use_trailing and not pos.get('trailing_active', False):
                if side == 'LONG' and current_price > entry * 1.005:
                    stored_sl = entry * 1.001
                    portfolio.positions[symbol]['sl_price'] = stored_sl
                    portfolio.positions[symbol]['trailing_active'] = True
                elif side == 'SHORT' and current_price < entry * 0.995:
                    stored_sl = entry * 0.999
                    portfolio.positions[symbol]['sl_price'] = stored_sl
                    portfolio.positions[symbol]['trailing_active'] = True
            
            if side == 'LONG':
                if low <= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time)
                    if trade: trades_executed += 1
                elif high >= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time)
                    if trade: trades_executed += 1
            else:
                if high >= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time)
                    if trade: trades_executed += 1
                elif low <= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time)
                    if trade: trades_executed += 1
        
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
                    if trade: trades_executed += 1
                continue
            
            if symbol not in portfolio.positions:
                # Confidence filter
                signal_strength = getattr(event, 'strength', 0.5)
                if signal_strength < min_conf:
                    signals_filtered += 1
                    continue
                
                # Dynamic sizing
                peak = portfolio.peak_equity
                current_cap = portfolio.current_capital
                initial = portfolio.initial_capital
                dd = (peak - current_cap) / peak if peak > 0 else 0
                
                risk_pct = getattr(Config, 'MAX_RISK_PER_TRADE', 0.05)
                if dd > 0.05: risk_pct *= 0.5
                if dd > 0.10: risk_pct *= 0.25
                
                risk_usd = current_cap * risk_pct
                
                # USE OUR CUSTOM SL/TP instead of signal defaults
                sl_decimal = sl_pct / 100.0
                tp_decimal = tp_pct / 100.0
                
                size_usd = (risk_usd / sl_decimal) if sl_decimal > 0 else (current_cap * 0.1)
                size_usd = min(size_usd, current_cap * 10)
                if size_usd < 5.0:
                    size_usd = 5.0
                
                side = 'LONG' if event.signal_type == SignalType.LONG else 'SHORT'
                
                if side == 'LONG':
                    entry_sl = current_price * (1 - sl_decimal)
                    entry_tp = current_price * (1 + tp_decimal)
                else:
                    entry_sl = current_price * (1 + sl_decimal)
                    entry_tp = current_price * (1 - tp_decimal)
                
                metadata = {'atr': getattr(event, 'atr', 0.0)}
                opened = portfolio.open_position_with_metadata(
                    symbol, side, current_price, size_usd, current_time, metadata, entry_sl, entry_tp
                )
                if opened:
                    trades_executed += 1
        
        if bar_count % 60 == 0:
            portfolio.update_equity(current_time)
    
    # Close remaining
    for sym in list(portfolio.positions.keys()):
        bars_end = data_provider.get_latest_bars(sym, 1)
        if bars_end is not None and len(bars_end) > 0:
            ts_ms = bars_end[-1]['timestamp']
            dt_close = pd.to_datetime(ts_ms, unit='ms', utc=True)
            trade = portfolio.close_position(sym, float(bars_end[-1]['close']), dt_close)
            if trade: trades_executed += 1
    
    return portfolio, trades_executed, signals_filtered


def test_config(label, sl, tp, comm, min_conf=0.0, trailing=False):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        portfolio, trades, filtered = run_backtest_tuned(data_15d, symbol, sl_pct=sl, tp_pct=tp, commission=comm, min_conf=min_conf, use_trailing=trailing)
    
    # Monkey-patch commission for BacktestPortfolio  
    m = calculate_metrics(portfolio)
    pnl = portfolio.current_capital - portfolio.initial_capital
    
    # Calculate avg win / avg loss
    wins = [t for t in portfolio.trades if t['pnl_usd'] > 0]
    losses = [t for t in portfolio.trades if t['pnl_usd'] <= 0]
    avg_win = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_usd']) for t in losses]) if losses else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0
    
    entry = {
        'label': label,
        'pnl': pnl,
        'return_pct': m['total_return'],
        'sharpe': m['sharpe_ratio'],
        'win_rate': m['win_rate'],
        'max_dd': m['max_drawdown_pct'],
        'trades': m['total_trades'],
        'payoff': payoff,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'filtered': filtered
    }
    results_log.append(entry)
    
    flag = "🟢" if pnl > 0 else "🔴"
    print(f"{flag} {label:<42} PnL:${pnl:>+7.3f} WR:{m['win_rate']:>5.1f}% T:{m['total_trades']:>4} S:{m['sharpe_ratio']:>6.2f} DD:{m['max_drawdown_pct']:>5.2f}% PR:{payoff:.2f} AvgW:${avg_win:.3f} AvgL:${avg_loss:.3f}")
    return entry


# ============================================================
# BARRIDO 1: SL/TP RATIO (con Taker fee)
# ============================================================
print("\n📊 BARRIDO 1: SL/TP RATIO (Taker 0.04%)")
print("-"*120)
for sl in [0.5, 0.8, 1.0, 1.5, 2.0]:
    for tp in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        ratio = tp / sl
        if ratio < 1.0: continue  # Skip bad R:R
        test_config(f"SL={sl}%/TP={tp}% (R:{ratio:.1f})", sl, tp, COMMISSION_TAKER)

# ============================================================
# BARRIDO 2: MEJOR SL/TP + Maker Fee
# ============================================================
print(f"\n📊 BARRIDO 2: MEJOR CONFIG + MAKER FEE (0.02%)")
print("-"*120)
best = max(results_log, key=lambda x: x['pnl'])
print(f"🏆 Mejor del Barrido 1: {best['label']} => PnL: ${best['pnl']:+.3f}")

# Re-test top 3 with Maker Fee
top3 = sorted(results_log, key=lambda x: x['pnl'], reverse=True)[:3]
for cfg in top3:
    parts = cfg['label'].split('/')
    sl_val = float(parts[0].split('=')[1].replace('%',''))
    tp_val = float(parts[1].split('=')[1].split('%')[0])
    test_config(f"MAKER+SL={sl_val}%/TP={tp_val}%", sl_val, tp_val, COMMISSION_MAKER)

# ============================================================
# BARRIDO 3: MEJOR CONFIG + Trailing Stop
# ============================================================
print(f"\n📊 BARRIDO 3: MEJOR CONFIG + TRAILING STOP")
print("-"*120)
best_maker = max([r for r in results_log if 'MAKER' in r['label']], key=lambda x: x['pnl'])
parts = best_maker['label'].replace('MAKER+', '').split('/')
sl_val = float(parts[0].split('=')[1].replace('%',''))
tp_val = float(parts[1].split('=')[1].split('%')[0])
test_config(f"TRAIL+MAKER+SL={sl_val}%/TP={tp_val}%", sl_val, tp_val, COMMISSION_MAKER, trailing=True)

# ============================================================
# BARRIDO 4: MEJOR CONFIG + Confidence Filter
# ============================================================
print(f"\n📊 BARRIDO 4: MEJOR CONFIG + FILTRO CONFIANZA")
print("-"*120)
for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
    test_config(f"CONF>{conf*100:.0f}%+MAKER+SL={sl_val}%/TP={tp_val}%", sl_val, tp_val, COMMISSION_MAKER, min_conf=conf)

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*120)
print("🏆 TOP 10 CONFIGURACIONES (ordenadas por PnL)")
print("="*120)
print(f"{'#':<3} {'Configuración':<48} {'PnL':>9} {'WR':>7} {'Trades':>7} {'Sharpe':>8} {'MaxDD':>8} {'Payoff':>8}")
print("-"*120)
sorted_all = sorted(results_log, key=lambda x: x['pnl'], reverse=True)
for i, r in enumerate(sorted_all[:10], 1):
    flag = "🟢" if r['pnl'] > 0 else "🔴"
    print(f"{flag}{i:<2} {r['label']:<48} ${r['pnl']:>+7.3f} {r['win_rate']:>5.1f}% {r['trades']:>6} {r['sharpe']:>7.2f} {r['max_dd']:>6.2f}% {r['payoff']:>6.2f}")

print(f"\n{'#':<3} {'Peores 3':<48} {'PnL':>9}")
print("-"*80)
for i, r in enumerate(sorted_all[-3:], 1):
    print(f"🔴{i:<2} {r['label']:<48} ${r['pnl']:>+7.3f}")

print("\n✅ OPTIMIZACIÓN V2 COMPLETADA")
