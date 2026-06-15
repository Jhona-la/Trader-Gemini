"""
🔬 DIAGNÓSTICO FORENSE: ¿Por qué SIZING_FAILED rechaza el 98% de señales?
Simula exactamente lo que haría el backtest para verificar cada paso del sizing.
"""
import sys
sys.path.insert(0, r'c:\Users\jhona\Documents\Proyectos\Trader Gemini')

from config import Config
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager

print("=" * 70)
print("🔬 DIAGNOSTIC: SIZING_FAILED ROOT CAUSE ANALYSIS")
print("=" * 70)

# Step 1: Create portfolio identical to backtest
capital = 13.0
print(f"\n📦 Step 1: Creating Portfolio with capital=${capital}")
portfolio = Portfolio(initial_capital=capital)
print(f"  ✅ Portfolio created. Cash: ${portfolio.current_cash:.2f}")
print(f"  ✅ Available cash (SCALPING): ${portfolio.get_available_cash(horizon='SCALPING'):.2f}")
print(f"  ✅ Available cash (SWING): ${portfolio.get_available_cash(horizon='SWING'):.2f}")

# Step 2: Create RiskManager identical to backtest
print(f"\n📦 Step 2: Creating RiskManager")
risk_manager = RiskManager(portfolio=portfolio)
print(f"  ✅ RiskManager created.")

# Step 3: Test size_position directly for each symbol
symbols = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"]
horizons = ["SCALPING", "SWING"]
print(f"\n📦 Step 3: Testing size_position for each symbol x horizon")

for symbol in symbols:
    for horizon in horizons:
        print(f"\n  --- {symbol} / {horizon} ---")
        # Simulate having a price
        if symbol == "BTC/USDT":
            price = 100000.0
        elif symbol == "ETH/USDT":
            price = 3500.0
        elif symbol == "SOL/USDT":
            price = 150.0
        elif symbol == "BNB/USDT":
            price = 650.0
        else:
            price = 100.0
        
        # Inject price into portfolio
        portfolio.update_market_price(symbol, price)
        
        signal_metadata = {'strength': 0.8, 'ml_confidence': 0.8}
        try:
            result = risk_manager.size_position(
                symbol=symbol,
                risk_pct=0.02,
                multiplier=1.0,
                horizon=horizon,
                current_price=price,
                signal_metadata=signal_metadata,
                direction="LONG"
            )
            if result:
                print(f"  ✅ SUCCESS: qty={result['quantity']:.6f}, notional=${result['notional']:.2f}, "
                      f"margin=${result['dollar_size']:.2f}, leverage={result['leverage']}x")
            else:
                rejection = signal_metadata.get('rejection_reason', 'UNKNOWN')
                print(f"  ❌ FAILED: result=None, rejection_reason={rejection}")
        except Exception as e:
            import traceback
            print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()

# Step 4: Debug available cash computation
print(f"\n📦 Step 4: Debug available cash breakdown")
print(f"  current_cash: ${getattr(portfolio, 'current_cash', 'N/A')}")
print(f"  used_margin: ${getattr(portfolio, 'used_margin', 'N/A')}")
print(f"  pending_cash: ${getattr(portfolio, 'pending_cash', 'N/A')}")

# Step 5: Check _get_asset_params
print(f"\n📦 Step 5: Testing _get_asset_params()")
for symbol in symbols:
    for horizon in horizons:
        try:
            params = risk_manager._get_asset_params(symbol, horizon)
            print(f"  {symbol}/{horizon}: sl={params.get('stop_loss_pct', 'N/A')}, "
                  f"tp={params.get('take_profit_pct', 'N/A')}, leverage={params.get('leverage', 'N/A')}")
        except AttributeError:
            # Might not exist, try the method name from generate_order
            print(f"  {symbol}/{horizon}: _get_asset_params not found, checking horizon_params...")
            h_params = risk_manager.horizon_params.get(horizon, {})
            print(f"    horizon_params: {h_params}")
        except Exception as e:
            print(f"  {symbol}/{horizon}: ERROR - {type(e).__name__}: {e}")

# Step 6: Check temporal supervisor (line 1678 blocks during startup)
print(f"\n📦 Step 6: Checking TemporalSupervisor state")
ts = getattr(risk_manager, 'temporal_supervisor', None)
if ts:
    print(f"  Phase: {getattr(ts, 'current_phase', 'N/A')}")
    print(f"  State: {getattr(ts, 'state', 'N/A')}")
else:
    print(f"  ✅ No temporal supervisor (no startup block)")

# Step 7: Check kill switch
print(f"\n📦 Step 7: Kill switch state")
ks = getattr(risk_manager, 'kill_switch', None)
if ks:
    print(f"  Active: {ks.active}")
    print(f"  Reason: {getattr(ks, 'activation_reason', 'N/A')}")
else:
    print(f"  ✅ No kill switch")

print(f"\n{'=' * 70}")
print("🔬 DIAGNOSIS COMPLETE")
print(f"{'=' * 70}")
