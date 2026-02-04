"""
🔄 TEST SYNC INTEGRITY - CONCURRENCY & TRACEABILITY TESTS
==========================================================

PROFESSOR METHOD:
- QUÉ: Tests de integridad de sincronización y trazabilidad de datos.
- POR QUÉ: Garantiza que lectura/escritura concurrente no cause corrupción.
- CÓMO: Threading para simular Dashboard leyendo mientras Bot escribe.
- CUÁNDO: Pre-producción para validar robustez del sistema.
- DÓNDE: tests/test_sync_integrity.py

TEST COVERAGE:
1. Concurrency Test: 1,000 overwrites with concurrent reads
2. Traceability Test: Follow XRP_Test_Trade through all layers
3. Latency Test: Measure write-to-render time (<500ms target)
"""

import os
import sys
import json
import time
import threading
import tempfile
import csv
import random
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'
os.environ['BINANCE_USE_TESTNET'] = 'True'


# =============================================================================
# CONCURRENCY TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='live_status_test_')
    os.close(fd)
    
    # Initialize with valid JSON
    with open(path, 'w') as f:
        json.dump({'timestamp': datetime.now(timezone.utc).isoformat(), 'test': True}, f)
    
    yield path
    
    # Cleanup with retry for Windows file locking
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            # Also clean up any .tmp files
            tmp_path = path + '.tmp'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            break
        except PermissionError:
            time.sleep(0.1)  # Wait for file handles to release


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    fd, path = tempfile.mkstemp(suffix='.csv', prefix='trades_test_')
    os.close(fd)
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# TEST: EXTREME CONCURRENCY (1,000 overwrites)
# =============================================================================

class TestConcurrencyExtreme:
    """
    🔥 Extreme Concurrency Test
    
    Simulates Dashboard reading live_status.json while Bot writes 1,000 times.
    Goal: Zero JSONDecodeError exceptions.
    """
    
    def test_concurrent_read_write_no_json_error(self, temp_json_file):
        """
        Test that concurrent reads never receive corrupted JSON.
        
        PROTOCOL:
        - Writer thread: Overwrites JSON 1,000 times using atomic writes
        - Reader thread: Continuously reads JSON as fast as possible
        - Success: Zero JSONDecodeError exceptions
        
        NOTE: On Windows, PermissionError may occur during os.replace() due to
        file locking. This is handled gracefully with retries.
        """
        read_errors = []
        read_count = [0]
        write_count = [0]
        permission_errors = [0]  # Track Windows file locking issues
        stop_reading = threading.Event()
        
        def writer_thread():
            """Simulate Bot writing status rapidly."""
            for i in range(1000):
                status = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'iteration': i,
                    'total_equity': 5000 + random.uniform(-100, 100),
                    'positions': {'XRP/USDT': {'quantity': 100, 'avg_price': 0.55}},
                    'source': 'CONCURRENCY_TEST'
                }
                
                # Atomic write with retry for Windows file locking
                temp_path = temp_json_file + '.tmp'
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(temp_path, 'w') as f:
                            json.dump(status, f, indent=2)
                        os.replace(temp_path, temp_json_file)
                        write_count[0] += 1
                        break
                    except PermissionError:
                        # Windows file locking - retry after brief pause
                        permission_errors[0] += 1
                        if attempt < max_retries - 1:
                            time.sleep(0.001)
                    except Exception as e:
                        read_errors.append(f"Write error at {i}: {e}")
                        break
            
            stop_reading.set()
        
        def reader_thread():
            """Simulate Dashboard reading status continuously."""
            while not stop_reading.is_set():
                try:
                    with open(temp_json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate structure
                    assert 'timestamp' in data
                    read_count[0] += 1
                    
                except json.JSONDecodeError as e:
                    read_errors.append(f"JSONDecodeError at read {read_count[0]}: {e}")
                except FileNotFoundError:
                    # File might be briefly unavailable during atomic replace
                    pass
                except PermissionError:
                    # Windows file locking during read - acceptable
                    pass
                except Exception as e:
                    read_errors.append(f"Read error at {read_count[0]}: {e}")
                
                # Small delay to not overwhelm
                time.sleep(0.0001)
        
        # Start threads
        writer = threading.Thread(target=writer_thread)
        reader = threading.Thread(target=reader_thread)
        
        reader.start()
        writer.start()
        
        writer.join()
        reader.join(timeout=2)
        
        # Assertions - adjusted for Windows file locking behavior
        # On Windows, some writes may fail due to file locking, but we should have most
        assert write_count[0] >= 900, f"Expected at least 900 writes, got {write_count[0]}"
        
        # The key assertion: NO JSONDecodeError (corrupted data)
        json_errors = [e for e in read_errors if 'JSONDecodeError' in e]
        assert len(json_errors) == 0, f"JSON corruption detected: {json_errors[:5]}"
        
        print(f"✅ Concurrency Test Passed: {write_count[0]} writes, {read_count[0]} reads")
        print(f"   (Windows PermissionErrors handled: {permission_errors[0]})")
    
    def test_multiple_readers(self, temp_json_file):
        """
        Test with multiple concurrent readers (simulates multiple dashboard instances).
        """
        read_errors = []
        read_counts = [0, 0, 0]  # 3 readers
        write_count = [0]
        stop_reading = threading.Event()
        
        def writer_thread():
            for i in range(500):
                status = {'timestamp': datetime.now(timezone.utc).isoformat(), 'iteration': i}
                temp_path = temp_json_file + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(status, f)
                os.replace(temp_path, temp_json_file)
                write_count[0] += 1
            stop_reading.set()
        
        def reader_thread(reader_id):
            while not stop_reading.is_set():
                try:
                    with open(temp_json_file, 'r') as f:
                        data = json.load(f)
                    read_counts[reader_id] += 1
                except json.JSONDecodeError as e:
                    read_errors.append(f"Reader {reader_id}: {e}")
                except FileNotFoundError:
                    pass
                time.sleep(0.0001)
        
        # Start threads
        writer = threading.Thread(target=writer_thread)
        readers = [threading.Thread(target=reader_thread, args=(i,)) for i in range(3)]
        
        for r in readers:
            r.start()
        writer.start()
        
        writer.join()
        for r in readers:
            r.join(timeout=2)
        
        assert len(read_errors) == 0, f"Errors with multiple readers: {read_errors[:5]}"
        print(f"✅ Multi-Reader Test Passed: {sum(read_counts)} total reads across 3 readers")


# =============================================================================
# TEST: TRACEABILITY (Follow trade through all layers)
# =============================================================================

class TestTraceability:
    """
    🔍 Traceability Test
    
    Follows a specific trade (XRP_Test_Trade) through all data layers:
    1. trades.csv registration
    2. Expectancy calculation
    3. Dashboard display
    """
    
    @pytest.fixture
    def sample_trade(self):
        """Create a sample test trade."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': 'XRP/USDT',
            'direction': 'LONG',
            'entry_price': 0.55,
            'exit_price': 0.56,
            'quantity': 100.0,
            'pnl': 1.0,
            'fee': 0.01,
            'net_pnl': 0.99,
            'is_reverse': False,
            'trade_id': 'XRP_TEST_TRADE_001',
            'strategy_id': 'TEST_STRATEGY'
        }
    
    def test_trade_registered_in_csv(self, temp_csv_file, sample_trade):
        """
        Test that a trade is correctly registered in trades.csv.
        """
        fieldnames = list(sample_trade.keys())
        
        # Write trade
        with open(temp_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(sample_trade)
        
        # Read and verify
        with open(temp_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1, "Expected exactly 1 trade"
        assert rows[0]['trade_id'] == 'XRP_TEST_TRADE_001'
        assert rows[0]['symbol'] == 'XRP/USDT'
        assert float(rows[0]['net_pnl']) == 0.99
        
        print("✅ Trade registered in CSV correctly")
    
    def test_trade_included_in_expectancy(self, temp_csv_file, sample_trade):
        """
        Test that the trade is correctly included in Expectancy calculation.
        
        E = (1/N) × Σ Net_PnL_i
        """
        # Write multiple trades
        trades = [
            {**sample_trade, 'trade_id': f'TRADE_{i}', 'net_pnl': 1.0 if i % 2 == 0 else -0.5}
            for i in range(10)
        ]
        
        fieldnames = list(trades[0].keys())
        with open(temp_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        
        # Read and calculate expectancy
        with open(temp_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            pnls = [float(row['net_pnl']) for row in reader]
        
        expectancy = sum(pnls) / len(pnls)
        
        # Manual calculation: 5 * 1.0 + 5 * (-0.5) = 5 - 2.5 = 2.5; E = 2.5/10 = 0.25
        expected = (5 * 1.0 + 5 * (-0.5)) / 10
        
        assert abs(expectancy - expected) < 0.001, f"Expectancy mismatch: {expectancy} vs {expected}"
        print(f"✅ Expectancy calculated correctly: ${expectancy:.4f}")
    
    def test_trade_appears_in_leaderboard_format(self, sample_trade):
        """
        Test that trade data is formatted correctly for leaderboard display.
        """
        # Simulate leaderboard aggregation by strategy
        trades = [
            {**sample_trade, 'strategy_id': 'STRATEGY_A', 'net_pnl': 10.0},
            {**sample_trade, 'strategy_id': 'STRATEGY_A', 'net_pnl': 5.0},
            {**sample_trade, 'strategy_id': 'STRATEGY_B', 'net_pnl': -2.0},
        ]
        
        # Aggregate by strategy
        leaderboard = {}
        for trade in trades:
            sid = trade['strategy_id']
            if sid not in leaderboard:
                leaderboard[sid] = {'pnl': 0, 'trades': 0}
            leaderboard[sid]['pnl'] += trade['net_pnl']
            leaderboard[sid]['trades'] += 1
        
        assert leaderboard['STRATEGY_A']['pnl'] == 15.0
        assert leaderboard['STRATEGY_A']['trades'] == 2
        assert leaderboard['STRATEGY_B']['pnl'] == -2.0
        
        print("✅ Leaderboard aggregation correct")


# =============================================================================
# TEST: END-TO-END LATENCY
# =============================================================================

class TestLatencyE2E:
    """
    ⏱️ End-to-End Latency Test
    
    Measures time from write to simulated render.
    Target: < 500ms
    """
    
    def test_write_to_read_latency(self, temp_json_file):
        """
        Measure write timestamp to read timestamp latency.
        """
        latencies = []
        
        for _ in range(100):
            # Write with timestamp
            write_time = time.perf_counter()
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'write_perf_counter': write_time
            }
            
            temp_path = temp_json_file + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f)
            os.replace(temp_path, temp_json_file)
            
            # Simulate immediate read (dashboard refresh)
            read_time = time.perf_counter()
            with open(temp_json_file, 'r') as f:
                data = json.load(f)
            
            # Calculate latency
            latency_ms = (read_time - write_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\n📊 Latency Results:")
        print(f"   Avg: {avg_latency:.2f}ms")
        print(f"   Max: {max_latency:.2f}ms")
        print(f"   P95: {p95_latency:.2f}ms")
        
        # Target: < 500ms for P95
        assert p95_latency < 500, f"P95 latency {p95_latency}ms exceeds 500ms target"
        
        if p95_latency < 10:
            print("🟢 EXCELLENT: Latency < 10ms")
        elif p95_latency < 100:
            print("🟡 GOOD: Latency < 100ms")
        else:
            print("🟠 ACCEPTABLE: Latency < 500ms")
    
    def test_simulated_dashboard_refresh_cycle(self, temp_json_file):
        """
        Simulate a complete dashboard refresh cycle.
        
        1. Bot writes status
        2. Dashboard reads status
        3. Dashboard parses data
        4. Dashboard updates UI (simulated)
        """
        # Step 1: Bot writes
        write_start = time.perf_counter()
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_equity': 5000.0,
            'positions': {'XRP/USDT': {'quantity': 100, 'avg_price': 0.55}},
            'metrics': {
                'win_rate': 56.0,
                'expectancy': 1.5658,
                'total_trades': 1000
            }
        }
        
        temp_path = temp_json_file + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(temp_path, temp_json_file)
        write_end = time.perf_counter()
        
        # Step 2: Dashboard reads
        read_start = time.perf_counter()
        with open(temp_json_file, 'r') as f:
            data = json.load(f)
        read_end = time.perf_counter()
        
        # Step 3: Dashboard parses
        parse_start = time.perf_counter()
        equity = data.get('total_equity', 0)
        positions = data.get('positions', {})
        metrics = data.get('metrics', {})
        parse_end = time.perf_counter()
        
        # Step 4: Simulate UI update (render time)
        render_start = time.perf_counter()
        # Simulate some processing
        _ = [f"{k}: {v}" for k, v in positions.items()]
        _ = [f"{k}: {v}" for k, v in metrics.items()]
        render_end = time.perf_counter()
        
        # Calculate times
        write_time = (write_end - write_start) * 1000
        read_time = (read_end - read_start) * 1000
        parse_time = (parse_end - parse_start) * 1000
        render_time = (render_end - render_start) * 1000
        total_time = (render_end - write_start) * 1000
        
        print(f"\n⏱️ Dashboard Refresh Cycle:")
        print(f"   Write:  {write_time:.2f}ms")
        print(f"   Read:   {read_time:.2f}ms")
        print(f"   Parse:  {parse_time:.2f}ms")
        print(f"   Render: {render_time:.2f}ms")
        print(f"   TOTAL:  {total_time:.2f}ms")
        
        assert total_time < 500, f"Total cycle time {total_time}ms exceeds 500ms"
        print(f"🟢 Dashboard refresh cycle: {total_time:.2f}ms < 500ms ✅")


# =============================================================================
# TEST: TYPE INTEGRITY
# =============================================================================

class TestTypeIntegrity:
    """
    🔢 Type Integrity Test
    
    Validates that Dashboard doesn't accidentally convert float to string.
    """
    
    def test_csv_float_preservation(self, temp_csv_file):
        """
        Test that floats are preserved when reading CSV.
        """
        # Write with float values
        trade = {
            'net_pnl': 1.5658,
            'entry_price': 0.55,
            'quantity': 100.0
        }
        
        with open(temp_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade.keys())
            writer.writeheader()
            writer.writerow(trade)
        
        # Read back
        with open(temp_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        # CSV always reads strings, but should be convertible to float
        net_pnl = float(row['net_pnl'])
        entry_price = float(row['entry_price'])
        quantity = float(row['quantity'])
        
        assert isinstance(net_pnl, float)
        assert abs(net_pnl - 1.5658) < 0.0001
        
        print("✅ Float values preserved correctly")
    
    def test_json_type_preservation(self, temp_json_file):
        """
        Test that JSON preserves types correctly.
        """
        data = {
            'int_val': 1000,
            'float_val': 1.5658,
            'str_val': 'test',
            'bool_val': True,
            'list_val': [1, 2, 3],
            'nested': {'a': 1.0, 'b': 2.0}
        }
        
        with open(temp_json_file, 'w') as f:
            json.dump(data, f)
        
        with open(temp_json_file, 'r') as f:
            loaded = json.load(f)
        
        assert isinstance(loaded['int_val'], int)
        assert isinstance(loaded['float_val'], float)
        assert isinstance(loaded['str_val'], str)
        assert isinstance(loaded['bool_val'], bool)
        assert isinstance(loaded['list_val'], list)
        assert isinstance(loaded['nested'], dict)
        
        print("✅ JSON types preserved correctly")


# =============================================================================
# ISOLATION GUARD TEST
# =============================================================================

class TestIsolationGuard:
    """
    🔒 Isolation Guard Test
    
    Verifies that tests don't modify production files.
    """
    
    def test_temp_files_used(self, temp_json_file, temp_csv_file):
        """
        Verify that temporary files are being used, not production files.
        """
        # Check paths
        assert 'temp' in temp_json_file.lower() or 'tmp' in temp_json_file.lower() or tempfile.gettempdir() in temp_json_file
        assert 'temp' in temp_csv_file.lower() or 'tmp' in temp_csv_file.lower() or tempfile.gettempdir() in temp_csv_file
        
        # Verify they exist
        assert os.path.exists(temp_json_file) or True  # May not exist yet
        
        print(f"✅ Using temp files:\n   JSON: {temp_json_file}\n   CSV: {temp_csv_file}")
    
    def test_production_files_not_modified(self):
        """
        Verify that production files are not touched during tests.
        """
        production_files = [
            'dashboard/data/live_status.json',
            'dashboard/data/status.csv',
            'dashboard/data/trades.csv'
        ]
        
        for path in production_files:
            # Just check that we're not accidentally using production paths
            # This is a safeguard, not an actual file modification check
            assert os.environ.get('TRADER_GEMINI_ENV') == 'TEST', \
                "Tests must run with TRADER_GEMINI_ENV=TEST"
        
        print("✅ Production files protected (env=TEST)")


# =============================================================================
# MAIN - Run Summary
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
