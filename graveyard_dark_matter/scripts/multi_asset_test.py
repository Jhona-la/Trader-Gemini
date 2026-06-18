"""Multi-Asset Hyper-Selective Scalping Verification"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("="*80)
print("🌐 MULTI-ASSET HIPER-SELECTIVO — TEST DE RENTABILIDAD GLOBAL")
print("="*80)

from scripts.run_multi_horizon_backtest import (
    fetch_data, run_strategy_backtest, INITIAL_CAPITAL, LEVERAGE
)

# Simulamos 15 días completos para ver si se llega a la meta de los 30-50 USD
TEST_DAYS = 15
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']

total_pnl = 0.0
total_trades = 0
total_wins = 0
symbol_stats = {}

print(f"💰 Capital Inicial: ${INITIAL_CAPITAL} | Apalancamiento: {LEVERAGE}x")
print(f"🎯 META: Pasar de $13 a $30+ en {TEST_DAYS} días")
print(f"🌍 Evaluando {len(SYMBOLS)} monedas con Filtro Franco-Tirador...\n")

for symbol in SYMBOLS:
    print(f"📡 Procesando {symbol}...")
    # Fetch 15 days data
    df = fetch_data(symbol, TEST_DAYS)
    if df is None or len(df) < 500:
        print(f"  ❌ Fallo al descargar {symbol}")
        continue
        
    # Correr Technical que ahora tiene min_votes = 6.5 (Hyper-Selective)
    result = run_strategy_backtest(df, symbol, 'Technical', INITIAL_CAPITAL, LEVERAGE, horizon_days=1)
    
    pnl = result['pnl_usd']
    trades = result['trades']
    wins_ratio = result['win_rate'] / 100.0
    wins = int(trades * wins_ratio)
    
    total_pnl += pnl
    total_trades += trades
    total_wins += wins
    
    status = "✅" if pnl >= 0 else "❌"
    print(f"  {status} {symbol}: PNL ${pnl:+.4f} | Trades: {trades} | WR: {result['win_rate']:.1f}% | DD: {result['max_drawdown']:.2f}%")
    
    if trades > 0:
        symbol_stats[symbol] = result

print("\n" + "="*80)
print("🎯 RESULTADO GLOBAL DEL ENJAMBRE MULTI-MONEDA")
print("="*80)

capital_final = INITIAL_CAPITAL + total_pnl
global_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

print(f"💵 Capital Base:   ${INITIAL_CAPITAL:.2f}")
print(f"📈 PNL Acumulado:  ${total_pnl:+.4f}")
print(f"🏦 CAPITAL FINAL:  ${capital_final:.2f} (ROI: {total_pnl/INITIAL_CAPITAL*100:+.1f}%)")
print(f"🎯 Win Rate Global:{global_wr:.1f}%")
print(f"📝 Total Trades:   {total_trades}")

if capital_final >= 30:
    print("\n✅ ¡OBJETIVO LOGRADO! El capital superó los $30 USD en 15 días asegurando el alimento.")
elif capital_final > INITIAL_CAPITAL:
    print("\n⚠️ SISTEMA RENTABLE, pero no alcanza la meta de los 30 USD. Optuna necesitará buscar más trades o leverage dinámico.")
else:
    print("\n❌ EL RENDIMIENTO AÚN NO ES SUFICIENTE. Se requiere tunear aún más los requisitos o ajustar apalancamiento.")
