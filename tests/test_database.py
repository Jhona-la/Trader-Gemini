"""
Trader Gemini - Database Test Suite

Tests for:
1. Database initialization and schema creation
2. CRUD operations (trades, signals, positions)
3. Crash recovery simulation
"""

import sys
import os
from pathlib import Path
import sqlite3
from datetime import datetime

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.database import DatabaseHandler
from config import Config

def test_database_initialization():
    """Test that database is created with correct schema"""
    print("\n" + "="*70)
    print("TEST 1: Database Initialization")
    print("="*70)
    
    # Create test database
    test_db = DatabaseHandler("test_trader.db")
    db_path = os.path.join(Config.DATA_DIR, "test_trader.db")
    
    try:
        # Verify database file exists
        assert os.path.exists(db_path), "Database file not created"
        print("  ✅ Database file created")
        
        # Verify tables exist
        conn = test_db.get_connection()
        cursor = conn.cursor()
        
        tables = ['trades', 'signals', 'positions', 'errors']
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            assert table in existing_tables, f"Table {table} not created"
            print(f"  ✅ Table '{table}' exists")
            
    finally:
        test_db.close()
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

def test_trade_logging():
    """Test logging trades to database"""
    print("\n" + "="*70)
    print("TEST 2: Trade Logging")
    print("="*70)
    
    test_db = DatabaseHandler("test_trader.db")
    db_path = os.path.join(Config.DATA_DIR, "test_trader.db")
    
    try:
        # Log sample trade
        trade = {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000.0,
            'timestamp': datetime.now(),
            'strategy_id': 'TestStrategy',
            'pnl': 0.0,
            'commission': 0.5
        }
        
        test_db.log_trade(trade)
        print("  ✅ Trade logged to database")
        
        # Verify trade was saved
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE symbol = ?", ('BTC/USDT',))
        result = cursor.fetchone()
        
        assert result is not None, "Trade not found in database"
        assert result['side'] == 'BUY', "Trade side mismatch"
        assert result['quantity'] == 0.001, "Trade quantity mismatch"
        print("  ✅ Trade data verified in database")
        
    finally:
        test_db.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def test_position_persistence():
    """Test position state persistence"""
    print("\n" + "="*70)
    print("TEST 3: Position Persistence")
    print("="*70)
    
    test_db = DatabaseHandler("test_trader.db")
    db_path = os.path.join(Config.DATA_DIR, "test_trader.db")
    
    try:
        # Update position
        test_db.update_position(
            symbol='ETH/USDT',
            quantity=0.5,
            entry_price=3000.0,
            current_price=3050.0,
            pnl=25.0
        )
        print("  ✅ Position saved to database")
        
        # Retrieve position
        positions = test_db.get_open_positions()
        
        assert 'ETH/USDT' in positions, "Position not found"
        assert positions['ETH/USDT']['quantity'] == 0.5, "Quantity mismatch"
        assert positions['ETH/USDT']['entry_price'] == 3000.0, "Entry price mismatch"
        print("  ✅ Position retrieved correctly")
        
        # Close position (quantity = 0)
        test_db.update_position(
            symbol='ETH/USDT',
            quantity=0,
            entry_price=3000.0,
            current_price=3100.0,
            pnl=50.0
        )
        print("  ✅ Position closed")
        
        # Verify position is removed
        positions = test_db.get_open_positions()
        assert 'ETH/USDT' not in positions, "Closed position still in database"
        print("  ✅ Closed position removed from active positions")
        
    finally:
        test_db.close()
        if os.path.exists(db_path):
            os.remove(db_path)

def test_crash_recovery_simulation():
    """Simulate crash and verify state recovery"""
    print("\n" + "="*70)
    print("TEST 4: Crash Recovery Simulation")
    print("="*70)
    
    test_db = DatabaseHandler("test_trader.db")
    db_path = os.path.join(Config.DATA_DIR, "test_trader.db")
    
    try:
        # Simulate active positions before crash
        positions_before_crash = {
            'BTC/USDT': {'quantity': 0.01, 'entry': 50000.0},
            'ETH/USDT': {'quantity': 0.5, 'entry': 3000.0},
            'SOL/USDT': {'quantity': 10.0, 'entry': 100.0}
        }
        
        for symbol, data in positions_before_crash.items():
            test_db.update_position(
                symbol=symbol,
                quantity=data['quantity'],
                entry_price=data['entry'],
                current_price=data['entry'],
                pnl=0.0
            )
        
        print(f"  ✅ Saved {len(positions_before_crash)} positions before 'crash'")
        
        # Simulate crash: Close and reopen database
        test_db.close()
        print("  💥 Simulated crash (database connection closed)")
        
        # Simulate restart: New database instance
        test_db_recovery = DatabaseHandler("test_trader.db")
        
        # Recover positions
        recovered_positions = test_db_recovery.get_open_positions()
        
        assert len(recovered_positions) == len(positions_before_crash), \
            f"Position count mismatch: {len(recovered_positions)} vs {len(positions_before_crash)}"
        print(f"  ✅ Recovered {len(recovered_positions)} positions after restart")
        
        for symbol in positions_before_crash:
            assert symbol in recovered_positions, f"{symbol} not recovered"
            assert recovered_positions[symbol]['quantity'] == positions_before_crash[symbol]['quantity'], \
                f"{symbol} quantity mismatch"
            print(f"  ✅ {symbol}: Verified (Qty: {recovered_positions[symbol]['quantity']})")
        
        test_db_recovery.close()
        
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def run_all_tests():
    """Run all database tests"""
    print("\n" + "="*70)
    print("🗄️ TRADER GEMINI - DATABASE TEST SUITE")
    print("="*70)
    
    tests = {
        'Database Initialization': test_database_initialization,
        'Trade Logging': test_trade_logging,
        'Position Persistence': test_position_persistence,
        'Crash Recovery': test_crash_recovery_simulation
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            test_func()
            results[name] = True
        except Exception as e:
            print(f"  ❌ FAILED: {test_name if 'test_name' in locals() else name}: {e}")
            results[name] = False
    
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
        print("\n  🎉 ALL TESTS PASSED - Database ready for production!")
    
    return 0 if total_passed == total_tests else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
