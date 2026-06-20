"""Deep diagnostic — understand WHY 52.8% WR with 1.8:1 ratio is still negative"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from scripts.run_multi_horizon_backtest import (
    fetch_data, compute_indicators, calibrate_sl_tp, signal_technical,
    SophiaClusterEngine, INITIAL_CAPITAL, LEVERAGE, HORIZON_PROFILES,
    COMMISSION_PCT, RISK_PER_TRADE
)
import numpy as np

df = fetch_data('BTC/USDT', 9)
print(f"Data: {len(df)} bars")

profile = HORIZON_PROFILES[1]
df2 = compute_indicators(df, horizon_days=1)
df2 = df2.dropna()

# Calibrate
closes = df2['close'].values
cal_sl, cal_tp = calibrate_sl_tp(closes, 60, sl_cap=profile['sl_cap'], tp_cap=profile['tp_cap'])
print(f"Calibrated SL={cal_sl*100:.3f}% TP={cal_tp*100:.3f}% Ratio={cal_tp/cal_sl:.2f}")

# Simulate a simplified backtest and track exit types
capital = INITIAL_CAPITAL
leverage = LEVERAGE 
position = None
trades = []
warmup = 200
rows = df2.reset_index()
total = len(rows)
rsi_window = []
sophia = SophiaClusterEngine(n_clusters=4, refit_interval=profile['sophia_refit'])

for i in range(warmup, total):
    row = rows.iloc[i]
    prev_row = rows.iloc[i-1]
    close = row['close']
    high = row['high']
    low = row['low']
    rsi_window.append(row['rsi'])
    if len(rsi_window) > 200: rsi_window.pop(0)
    sophia.update(row)
    
    # Check exits
    if position is not None:
        side = position['side']
        sl = position['sl']
        tp = position['tp']
        entry = position['entry']
        size_usd = position['size_usd']
        bars_held = i - position['entry_idx']
        commission = size_usd * COMMISSION_PCT * 2
        
        if side == 'long':
            if low <= sl:
                pnl_pct = (sl - entry) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'SL', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
            elif high >= tp:
                pnl_pct = (tp - entry) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'TP', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
            elif bars_held >= profile['max_hold_bars']:
                pnl_pct = (close - entry) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'TIME', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
        elif side == 'short':
            if high >= sl:
                pnl_pct = (entry - sl) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'SL', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
            elif low <= tp:
                pnl_pct = (entry - tp) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'TP', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
            elif bars_held >= profile['max_hold_bars']:
                pnl_pct = (entry - close) / entry
                pnl_usd = size_usd * pnl_pct - commission
                capital += pnl_usd
                trades.append({'pnl': pnl_usd, 'exit': 'TIME', 'bars': bars_held, 'pnl_pct': pnl_pct*100})
                position = None
    
    # Entry
    if position is None:
        if len(rsi_window) >= 50:
            rsi_buy = max(20, min(np.percentile(rsi_window, 15), 40))
            rsi_sell = min(80, max(np.percentile(rsi_window, 85), 60))
            params = {'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
                      'calibrated_sl': cal_sl, 'calibrated_tp': cal_tp}
        else:
            params = {'rsi_buy': 30, 'rsi_sell': 70,
                      'calibrated_sl': cal_sl, 'calibrated_tp': cal_tp}
        
        is_safe, regime, conf = sophia.is_safe_to_trade(row)
        in_bootstrap = len(trades) < 20
        if not is_safe and not in_bootstrap:
            continue
            
        direction, sl_pct, tp_pct = signal_technical(row, prev_row, params, regime=regime, horizon_profile=profile)
        
        if direction is not None:
            size_pct = RISK_PER_TRADE
            notional = capital * leverage * size_pct
            if notional < 10.0:
                notional = min(10.0, capital * leverage * 0.45)
            size_usd = min(notional, capital * leverage)
            
            if direction == 'long':
                position = {'side': 'long', 'entry': close, 'sl': close*(1-sl_pct),
                           'tp': close*(1+tp_pct), 'size_usd': size_usd, 'entry_idx': i}
            else:
                position = {'side': 'short', 'entry': close, 'sl': close*(1+sl_pct),
                           'tp': close*(1-tp_pct), 'size_usd': size_usd, 'entry_idx': i}

# Analysis
print(f"\n{'='*70}")
print(f"TRADE DISTRIBUTION ANALYSIS")
print(f"{'='*70}")
print(f"Total trades: {len(trades)}")

for exit_type in ['SL', 'TP', 'TIME']:
    subset = [t for t in trades if t['exit'] == exit_type]
    if subset:
        avg_pnl = np.mean([t['pnl'] for t in subset])
        avg_pct = np.mean([t['pnl_pct'] for t in subset])
        wins = len([t for t in subset if t['pnl'] > 0])
        avg_bars = np.mean([t['bars'] for t in subset])
        print(f"\n{exit_type}: {len(subset)} trades ({len(subset)/len(trades)*100:.0f}%)")
        print(f"  Avg PNL: ${avg_pnl:+.4f} ({avg_pct:+.2f}%)")
        print(f"  Wins: {wins}/{len(subset)} ({wins/len(subset)*100:.0f}%)")
        print(f"  Avg bars held: {avg_bars:.0f}")
        
total_pnl = sum(t['pnl'] for t in trades)
total_commissions = sum(t['size_usd'] * COMMISSION_PCT * 2 for t in [{'size_usd': 10.0}] * len(trades))
wins = len([t for t in trades if t['pnl'] > 0])
print(f"\nTotal PNL: ${total_pnl:+.4f}")
print(f"Final capital: ${capital:.4f}")
print(f"Win Rate: {wins/len(trades)*100:.1f}%")
print(f"Est. total commissions ({len(trades)} trades): ${len(trades) * 10.0 * COMMISSION_PCT * 2:.4f}")
