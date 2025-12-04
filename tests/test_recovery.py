"""
Trader Gemini - Portfolio + Database Integration Test

Tests the integration between Portfolio class and DatabaseHandler:
1. Trade execution logging to DB
2. Position state synchronization
3. Crash recovery with Portfolio restoration
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.portfolio import Portfolio
from core.events import FillEvent
from config import Config

def test_portfolio_db_integration():
    """Test Portfolio integration with DatabaseHandler"""
    print("\n" + "="*70)
    print("TEST: Portfolio + Database Integration")
    print("="*70)
    
    # Create test portfolio with database
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    portfolio = Portfolio(
        initial_capital=10000.0,
        csv_path=test_csv,
        status_path=test_status
    )
    
    try:
        # Simulate BUY trade
        buy_event = FillEvent(
            timeindex=datetime.now(),
            symbol='BTC/USDT',
            exchange='BINANCE',
            quantity=0.01,
            direction='BUY',
            fill_cost=500.0  # 0.01 BTC @ $50,000
        )
        buy_event.strategy_id = 'TestStrategy'
        
        print("\n  📊 Executing BUY trade...")
        portfolio.update_fill(buy_event)
        print("  ✅ BUY trade executed and logged to DB")
        
        # Verify position in database
        db_positions = portfolio.db.get_open_positions()
        assert 'BTC/USDT' in db_positions, "Position not found in DB"
        assert db_positions['BTC/USDT']['quantity'] == 0.01, "Quantity mismatch"
        print(f"  ✅ Position verified in DB: {db_positions['BTC/USDT']}")
        
        # Simulate price update
        portfolio.update_market_price('BTC/USDT', 51000.0)
        print("  ✅ Price update logged to DB")
        
        # Verify updated position
        db_positions = portfolio.db.get_open_positions()
        assert db_positions['BTC/USDT']['current_price'] == 51000.0, "Price not updated"
        print(f"  ✅ Current price in DB: ${db_positions['BTC/USDT']['current_price']}")
        
        # Simulate SELL trade (close position)
        sell_event = FillEvent(
            timeindex=datetime.now(),
            symbol='BTC/USDT',
            exchange='BINANCE',
            quantity=0.01,
            direction='SELL',
            fill_cost=510.0  # 0.01 BTC @ $51,000 (profit: $10)
        )
        sell_event.strategy_id = 'TestStrategy'
        
        print("\n  📊 Executing SELL trade (close position)...")
        portfolio.update_fill(sell_event)
        print("  ✅ SELL trade executed and logged to DB")
        
        # Verify position is closed in database
        db_positions = portfolio.db.get_open_positions()
        assert 'BTC/USDT' not in db_positions, "Position not closed in DB"
        print("  ✅ Position removed from DB after close")
        
        # Verify trades are logged
        conn = portfolio.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM trades WHERE symbol = 'BTC/USDT'")
        trade_count = cursor.fetchone()['count']
        
        assert trade_count == 2, f"Expected 2 trades, found {trade_count}"
        print(f"  ✅ Found {trade_count} trades in DB (BUY + SELL)")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        portfolio.db.close()
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        
        # Remove test database
        test_db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

def test_crash_recovery_with_portfolio():
    """Test crash recovery using Portfolio restoration"""
    print("\n" + "="*70)
    print("TEST: Crash Recovery with Portfolio")
    print("="*70)
    
    test_csv = os.path.join(Config.DATA_DIR, "test_trades.csv")
    test_status = os.path.join(Config.DATA_DIR, "test_status.csv")
    
    # Create portfolio and open positions
    portfolio1 = Portfolio(
        initial_capital=10000.0,
        csv_path=test_csv,
        status_path=test_status
    )
    
    try:
        # Open two positions
        buy_btc = FillEvent(
            timeindex=datetime.now(),
            symbol='BTC/USDT',
            exchange='BINANCE',
            quantity=0.01,
            direction='BUY',
            fill_cost=500.0
        )
        buy_btc.strategy_id = 'Strategy1'
        
        buy_eth = FillEvent(
            timeindex=datetime.now(),
            symbol='ETH/USDT',
            exchange='BINANCE',
            quantity=0.5,
            direction='BUY',
            fill_cost=1500.0
        )
        buy_eth.strategy_id = 'Strategy2'
        
        portfolio1.update_fill(buy_btc)
        portfolio1.update_fill(buy_eth)
        
        print("  ✅ Created 2 open positions")
        print(f"     - BTC/USDT: 0.01")
        print(f"     - ETH/USDT: 0.5")
        
        # Close portfolio (simulate crash)
        portfolio1.db.close()
        print("\n  💥 Simulated crash (portfolio closed)")
        
        # Create new portfolio instance (simulate restart)
        portfolio2 = Portfolio(
            initial_capital=10000.0,
            csv_path=test_csv,
            status_path=test_status
        )
        
        print("\n  🔄 Attempting to restore state from DB...")
        recovery_success = portfolio2.restore_state_from_db()
        
        assert recovery_success, "State restoration failed"
        print("  ✅ State restoration successful")
        
        # Verify positions were restored
        assert len(portfolio2.positions) >= 2, "Not all positions restored"
        assert 'BTC/USDT' in portfolio2.positions, "BTC position not restored"
        assert 'ETH/USDT' in portfolio2.positions, "ETH position not restored"
        
        print(f"  ✅ Restored {len(portfolio2.positions)} positions")
        print(f"     - BTC/USDT: {portfolio2.positions['BTC/USDT']['quantity']}")
        print(f"     - ETH/USDT: {portfolio2.positions['ETH/USDT']['quantity']}")
        
        portfolio2.db.close()
        return True
        
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        for file in [test_csv, test_status]:
            if os.path.exists(file):
                os.remove(file)
        
        test_db_path = os.path.join(Config.DATA_DIR, "trader_gemini.db")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("🔗 TRADER GEMINI - PORTFOLIO + DB INTEGRATION TESTS")
    print("="*70)
    
    results = {
        'Portfolio DB Integration': test_portfolio_db_integration(),
        'Crash Recovery with Portfolio': test_crash_recovery_with_portfolio()
    }
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n  Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  🎉 ALL INTEGRATION TESTS PASSED!")
        print("  Phase 2 (Persistence & State Isolation) is COMPLETE")
    
    return 0 if total_passed == total_tests else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
