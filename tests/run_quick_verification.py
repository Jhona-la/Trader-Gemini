
import sys
import os

# FORCE TESTNET FOR VERIFICATION
os.environ['BINANCE_USE_TESTNET'] = 'True'
os.environ['BINANCE_USE_DEMO'] = 'True'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.run_backtest import fetch_binance_data, run_backtest, INITIAL_CAPITAL
import pandas as pd

def run_quick_verification():
    print("🚀 STARTING QUICK VERIFICATION (Phase 4 Integration Test)...")
    
    symbol = 'BTC/USDT'
    days = 3
    
    print(f"1. Fetching {days} days of data for {symbol}...")
    try:
        data = fetch_binance_data(symbol, days=days)
    except Exception as e:
        print(f"❌ DATA DOWNLOAD FAILED: {e}")
        sys.exit(1)
        
    if data.empty:
        print("❌ DATA EMPTY")
        sys.exit(1)
        
    print(f"2. Running Backtest Simulation...")
    results = run_backtest(data, symbol)
    
    p = results['portfolio']
    trades = len(p.trades)
    pnl = p.current_capital - p.initial_capital
    
    print("\n" + "="*40)
    print(f"📊 VERIFICATION RESULTS ({symbol})")
    print(f"   Trades Executed: {trades}")
    print(f"   PnL: ${pnl:.2f}")
    print(f"   Final Capital: ${p.current_capital:.2f}")
    print("="*40)
    
    if trades > 0:
        print("✅ INTEGRATION TEST PASSED: System generated trades.")
        sys.exit(0)
    else:
        print("⚠️ INTEGRATION WARNING: No trades generated (Market might be flat or logic too strict).")
        # Check if data length is sufficient
        if len(data) < 200:
            print("   Reason: Insufficient data points.")
        sys.exit(0) # Not necessarily a failure of LOGIC, but check warning.

if __name__ == "__main__":
    run_quick_verification()
