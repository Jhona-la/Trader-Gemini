import sys
import os
from datetime import datetime, timezone
import queue

# Ensure we can import from the root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from execution.binance_executor import BinanceExecutor
from core.portfolio import Portfolio
from utils.logger import logger

def audit_positions():
    print("=" * 70)
    print("🕵️  TRADER GEMINI - FUTURES DEMO AUDIT")
    print("=" * 70)
    
    # 1. Initialize Portfolio (to see what's in DB/State)
    portfolio = Portfolio(initial_capital=15.0)
    portfolio.restore_state_from_db()
    
    # 2. Initialize Executor
    events = queue.Queue()
    executor = BinanceExecutor(events, portfolio=portfolio)
    
    # 3. Fetch Live Data
    print("\n📡 Fetching Live Binance Demo State...")
    balance = executor.get_all_balances()
    
    try:
        live_positions = executor.exchange.fetch_positions()
        # Filter non-zero
        active_live = [p for p in live_positions if float(p['info']['positionAmt']) != 0]
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        active_live = []

    # 4. Compare
    print("\n" + "=" * 70)
    print("📊 POSITION COMPARISON")
    print("-" * 70)
    
    bot_positions = portfolio.positions
    
    # Symbols involved
    all_symbols = set(list(bot_positions.keys()) + [p['symbol'] for p in active_live])
    
    if not all_symbols:
        print("✅ No positions found (Live or Bot).")
    else:
        print(f"{'Symbol':15} | {'Bot State':15} | {'Binance Live':15} | {'Status'}")
        print("-" * 70)
        
        for sym in all_symbols:
            # Map symbol format (Internal BTC/USDT vs Exchange BTCUSDT)
            exchange_sym = sym.replace('/', '') if '/' in sym else sym
            
            # Bot State
            bot_pos = bot_positions.get(sym, {})
            bot_qty = bot_pos['quantity']
            
            # Live State
            live_entry = next((p for p in active_live if p['symbol'] == exchange_sym), None)
            live_qty = float(live_entry['info']['positionAmt']) if live_entry else 0
            
            # Status check
            if abs(bot_qty - live_qty) < 0.000001:
                status = "✅ MATCH"
            elif bot_qty != 0 and live_qty == 0:
                status = "❌ GHOST (Bot only)"
            elif bot_qty == 0 and live_qty != 0:
                status = "⚠️  MANUAL/LEGACY"
            else:
                status = "❓ MISMATCH"
                
            print(f"{sym:15} | {bot_qty:15.4f} | {live_qty:15.4f} | {status}")

    print("=" * 70)
    print(f"Audit Complete at {datetime.now(timezone.utc)}")

if __name__ == "__main__":
    audit_positions()
