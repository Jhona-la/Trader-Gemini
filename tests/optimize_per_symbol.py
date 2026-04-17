"""
🔬 OPTIMIZADOR PER-SYMBOL: Perfiles Adaptativos Evolutivos
Barre SL/TP para cada símbolo y genera perfiles óptimos automáticos.
Usa subset de 6 símbolos representativos (BTC, ETH, SOL, XRP, BNB, DOGE)
para cubrir baja/media/alta volatilidad.
"""
import sys, os, io, contextlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_infra import fetch_binance_data, calculate_metrics, BacktestPortfolio, INITIAL_CAPITAL, LEVERAGE
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent, SignalEvent
from core.enums import SignalType
from config import Config
from queue import Queue
from core.backtest_infra import BacktestDataProvider
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
DAYS = 15
SL_RANGE = [0.5, 0.8, 1.0, 1.5, 2.0]
TP_RANGE = [1.0, 1.5, 2.0, 3.0, 4.0]
COMMISSION = 0.0002  # Maker

print("="*80)
print("🔬 OPTIMIZADOR PER-SYMBOL: PERFILES ADAPTATIVOS EVOLUTIVOS")
print("="*80)

# ============================================================
# DOWNLOAD ALL DATA FIRST
# ============================================================
all_data = {}
for sym in SYMBOLS:
    print(f"📡 Descargando {DAYS}D datos para {sym}...")
    try:
        data = fetch_binance_data(sym, days=DAYS)
        if data is not None and len(data) > 100:
            all_data[sym] = data
            print(f"   ✅ {len(data)} velas")
        else:
            print(f"   ⚠️ Datos insuficientes, saltando {sym}")
    except Exception as e:
        print(f"   ❌ Error: {e}, saltando {sym}")

def run_backtest_sym(data, symbol, sl_pct, tp_pct):
    """Backtest rápido con SL/TP específicos"""
    events_queue = Queue()
    historical_data = {symbol: data}
    data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
    portfolio = BacktestPortfolio(INITIAL_CAPITAL, LEVERAGE)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    
    warmup = 100
    trades = 0
    bar_count = 0
    
    while data_provider.continue_backtest:
        data_provider.update_bars()
        bar_count += 1
        if bar_count < warmup: continue
        
        bars = data_provider.get_latest_bars(symbol, 1)
        if not bars: continue
        
        current_bar = bars[-1]
        price = float(current_bar['close'])
        ts = pd.to_datetime(current_bar['timestamp'], unit='ms', utc=True)
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            entry = pos['entry']
            side = pos['side']
            sl_p = pos.get('sl_price')
            tp_p = pos.get('tp_price')
            
            if side == 'LONG':
                if low <= sl_p:
                    t = portfolio.close_position(symbol, sl_p, ts)
                    if t: trades += 1
                elif high >= tp_p:
                    t = portfolio.close_position(symbol, tp_p, ts)
                    if t: trades += 1
            else:
                if high >= sl_p:
                    t = portfolio.close_position(symbol, sl_p, ts)
                    if t: trades += 1
                elif low <= tp_p:
                    t = portfolio.close_position(symbol, tp_p, ts)
                    if t: trades += 1
        
        strategy.bought[symbol] = symbol in portfolio.positions
        market_event = MarketEvent(symbol=symbol, close_price=price, timestamp=ts)
        strategy.calculate_signals(market_event)
        
        while not events_queue.empty():
            event = events_queue.get()
            if not isinstance(event, SignalEvent): continue
            if event.signal_type == SignalType.EXIT:
                if symbol in portfolio.positions:
                    t = portfolio.close_position(symbol, price, ts)
                    if t: trades += 1
                continue
            
            if symbol not in portfolio.positions:
                peak = portfolio.peak_equity
                cap = portfolio.current_capital
                dd = (peak - cap) / peak if peak > 0 else 0
                risk_pct = getattr(Config, 'MAX_RISK_PER_TRADE', 0.05)
                if dd > 0.05: risk_pct *= 0.5
                if dd > 0.10: risk_pct *= 0.25
                
                sl_dec = sl_pct / 100.0
                tp_dec = tp_pct / 100.0
                risk_usd = cap * risk_pct
                size = (risk_usd / sl_dec) if sl_dec > 0 else (cap * 0.1)
                size = min(size, cap * 10)
                if size < 5.0: size = 5.0
                
                side = 'LONG' if event.signal_type == SignalType.LONG else 'SHORT'
                if side == 'LONG':
                    sl_price = price * (1 - sl_dec)
                    tp_price = price * (1 + tp_dec)
                else:
                    sl_price = price * (1 + sl_dec)
                    tp_price = price * (1 - tp_dec)
                
                meta = {'atr': getattr(event, 'atr', 0.0)}
                opened = portfolio.open_position_with_metadata(
                    symbol, side, price, size, ts, meta, sl_price, tp_price
                )
                if opened: trades += 1
        
        if bar_count % 60 == 0: portfolio.update_equity(ts)
    
    # Close remaining
    for sym_k in list(portfolio.positions.keys()):
        b = data_provider.get_latest_bars(sym_k, 1)
        if b and len(b) > 0:
            dt = pd.to_datetime(b[-1]['timestamp'], unit='ms', utc=True)
            t = portfolio.close_position(sym_k, float(b[-1]['close']), dt)
            if t: trades += 1
    
    return portfolio, trades

# ============================================================
# SWEEP PER SYMBOL
# ============================================================
optimal_profiles = {}

for sym in SYMBOLS:
    if sym not in all_data:
        continue
    
    data = all_data[sym]
    print(f"\n{'='*80}")
    print(f"🎯 BARRIDO DE {sym} ({len(data)} velas, {DAYS}D)")
    print(f"{'='*80}")
    
    best_pnl = -999
    best_config = None
    results = []
    
    for sl in SL_RANGE:
        for tp in TP_RANGE:
            if tp / sl < 1.0: continue  # Skip bad ratios
            
            f_out = io.StringIO()
            with contextlib.redirect_stdout(f_out):
                portfolio, num_trades = run_backtest_sym(data, sym, sl, tp)
            
            m = calculate_metrics(portfolio)
            pnl = portfolio.current_capital - portfolio.initial_capital
            
            wins = [t for t in portfolio.trades if t['pnl_usd'] > 0]
            losses = [t for t in portfolio.trades if t['pnl_usd'] <= 0]
            avg_w = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
            avg_l = np.mean([abs(t['pnl_usd']) for t in losses]) if losses else 0
            payoff = avg_w / avg_l if avg_l > 0 else 0
            
            flag = "🟢" if pnl > 0 else "🔴"
            print(f"  {flag} SL={sl}%/TP={tp}% R:{tp/sl:.1f}  PnL:${pnl:>+7.3f} WR:{m['win_rate']:>5.1f}% T:{num_trades:>3} S:{m['sharpe_ratio']:>6.2f} PR:{payoff:.2f}")
            
            result = {
                'sl': sl, 'tp': tp, 'ratio': tp/sl,
                'pnl': pnl, 'return_pct': m['total_return'],
                'sharpe': m['sharpe_ratio'], 'win_rate': m['win_rate'],
                'max_dd': m['max_drawdown_pct'], 'trades': num_trades,
                'payoff': payoff
            }
            results.append(result)
            
            if pnl > best_pnl:
                best_pnl = pnl
                best_config = result
    
    # Find best by composite score: PnL weighted + Sharpe bonus + Payoff bonus
    for r in results:
        r['score'] = r['pnl'] + (r['sharpe'] * 0.3) + (r['payoff'] * 0.2)
    
    best_by_score = max(results, key=lambda x: x['score'])
    
    print(f"\n  🏆 MEJOR {sym}:")
    print(f"     PnL max: SL={best_config['sl']}%/TP={best_config['tp']}% → ${best_config['pnl']:+.3f}")
    print(f"     Score:   SL={best_by_score['sl']}%/TP={best_by_score['tp']}% → Score:{best_by_score['score']:.3f}")
    
    # Use the best by composite score
    winner = best_by_score
    optimal_profiles[sym] = {
        'sl_pct': winner['sl'] / 100.0,  # Convert to decimal
        'tp_pct': winner['tp'] / 100.0,
        'ratio': winner['ratio'],
        'sharpe': winner['sharpe'],
        'pnl': winner['pnl'],
        'win_rate': winner['win_rate'],
        'trades': winner['trades'],
        'payoff': winner['payoff']
    }

# ============================================================
# GENERATE ADAPTIVE PROFILES
# ============================================================
print("\n" + "="*80)
print("🏆 PERFILES ADAPTATIVOS ÓPTIMOS POR SÍMBOLO")
print("="*80)
print(f"{'Símbolo':<15} {'SL%':>6} {'TP%':>6} {'Ratio':>6} {'PnL':>10} {'WR':>7} {'Sharpe':>8} {'Payoff':>8}")
print("-"*80)

for sym, prof in optimal_profiles.items():
    flag = "🟢" if prof['pnl'] > 0 else "🔴"
    print(f"{flag} {sym:<13} {prof['sl_pct']*100:>5.1f}% {prof['tp_pct']*100:>5.1f}% {prof['ratio']:>5.1f}x ${prof['pnl']:>+8.3f} {prof['win_rate']:>5.1f}% {prof['sharpe']:>7.2f} {prof['payoff']:>6.2f}")

# ============================================================
# GENERATE PYTHON CODE FOR TECHNICAL.PY
# ============================================================
print("\n" + "="*80)
print("📋 CÓDIGO GENERADO PARA technical.py (COPIAR Y PEGAR)")
print("="*80)

# Map remaining symbols to nearest profile
def classify_profile(sl_dec):
    if sl_dec <= 0.008:
        return 'AGGRESSIVE'
    elif sl_dec <= 0.012:
        return 'BALANCED'
    else:
        return 'CONSERVATIVE'

# Generate PER_SYMBOL_PROFILES dict
print("\n# === V2 OPTIMIZED PER-SYMBOL ADAPTIVE PROFILES ===")
print("PER_SYMBOL_PROFILES = {")
for sym, prof in optimal_profiles.items():
    profile_name = classify_profile(prof['sl_pct'])
    print(f"    '{sym}': {{'sl_pct': {prof['sl_pct']:.4f}, 'tp_pct': {prof['tp_pct']:.4f}, 'profile': '{profile_name}'}},  # PnL:${prof['pnl']:+.2f} S:{prof['sharpe']:.1f}")
print("}")

# Save optimal profiles as JSON
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimal_profiles.json')
with open(output_path, 'w') as f:
    json.dump(optimal_profiles, f, indent=2)
print(f"\n💾 Perfiles guardados en: {output_path}")

# Generate SYMBOL_MAP update
print("\n# === SYMBOL_MAP UPDATE ===")
print("SYMBOL_MAP = {")
all_symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT',
    'ADA/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'AVAX/USDT',
    'NEAR/USDT', 'INJ/USDT', 'PEPE/USDT', 'RENDER/USDT', 'SHIB/USDT',
    'ATOM/USDT', 'LTC/USDT', 'OP/USDT', 'ARB/USDT'
]
for sym in all_symbols:
    if sym in optimal_profiles:
        prof = optimal_profiles[sym]
        profile_name = classify_profile(prof['sl_pct'])
        print(f"    '{sym}': '{profile_name}',  # Optimized: SL={prof['sl_pct']*100:.1f}%/TP={prof['tp_pct']*100:.1f}%")
    else:
        # Default: use closest category (BALANCED for unknowns)
        print(f"    '{sym}': 'BALANCED',  # Default (not optimized yet)")
print("}")

print("\n✅ OPTIMIZACIÓN PER-SYMBOL COMPLETADA")
