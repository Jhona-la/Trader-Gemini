import asyncio
from core.portfolio import Portfolio
from data.database import DatabaseHandler
import config as Config

async def test_restore():
    print("Testing DB and Portfolio Restore...")
    
    # 1. Initialize Database
    db = DatabaseHandler("test_restore.db")
    db.create_tables()
    
    # 2. Insert a fake position with all fields
    db.update_position(
        symbol="BTC/USDT",
        quantity=0.05,
        entry_price=60000.0,
        current_price=61000.0,
        pnl=50.0,
        sl_pct=0.015,
        tp_pct=0.030,
        horizon="SWING",
        strategy_id="TEST_STRAT"
    )
    
    # 3. Initialize Portfolio
    portfolio = Portfolio(initial_capital=100.0)
    portfolio.db = db
    
    # 4. Trigger restore
    portfolio.restore_state_from_db()
    
    # 5. Check virtual ledger
    print("\n--- Final Virtual Ledger State ---")
    for k, v in portfolio.virtual_ledger.items():
        print(f"{k}: {v}")
        
    print("\nTest passed if virtual_ledger contains BTC/USDT_SWING with sl_pct=0.015!")

if __name__ == "__main__":
    asyncio.run(test_restore())
