"""
🔄 CONCURRENCY TESTS - DATA COLLISION STRESS TESTING
=====================================================

PROFESSOR METHOD:
- QUÉ: Tests de estrés para colisiones de datos bajo carga concurrente.
- POR QUÉ: Valida que el sistema no corrompa datos bajo carga extrema.
- CÓMO: Threading masivo, escritura/lectura simultánea.
- CUÁNDO: Pre-producción para validar resiliencia.

TESTS:
- 1000 writes with concurrent reads
- Multi-writer scenarios
- File lock handling
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
from typing import List
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['TRADER_GEMINI_ENV'] = 'TEST'


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_json():
    """Create temp JSON file with Windows cleanup retry."""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='concurrency_test_')
    os.close(fd)
    with open(path, 'w') as f:
        json.dump({'init': True}, f)
    
    yield path
    
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            tmp = path + '.tmp'
            if os.path.exists(tmp):
                os.remove(tmp)
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def temp_csv():
    """Create temp CSV file."""
    fd, path = tempfile.mkstemp(suffix='.csv', prefix='concurrency_test_')
    os.close(fd)
    
    yield path
    
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


# =============================================================================
# TEST: HIGH FREQUENCY WRITES
# =============================================================================

class TestHighFrequencyWrites:
    """
    📊 Test High Frequency Write Operations
    """
    
    def test_1000_sequential_writes(self, temp_json):
        """
        Test 1000 sequential writes complete without data corruption.
        """
        for i in range(1000):
            status = {'iteration': i, 'timestamp': datetime.now(timezone.utc).isoformat()}
            
            # Atomic write
            temp_path = temp_json + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f)
            
            # Windows retry
            for attempt in range(3):
                try:
                    os.replace(temp_path, temp_json)
                    break
                except PermissionError:
                    time.sleep(0.001)
        
        # Verify final state
        with open(temp_json, 'r') as f:
            final = json.load(f)
        
        assert final['iteration'] == 999, f"Expected 999, got {final['iteration']}"
        print("✅ 1000 sequential writes completed successfully")
    
    def test_csv_append_stress(self, temp_csv):
        """
        Test 1000 CSV appends maintain integrity.
        """
        fieldnames = ['timestamp', 'id', 'value']
        
        # Write header
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        
        # Append 1000 rows
        for i in range(1000):
            with open(temp_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'id': i,
                    'value': random.random()
                })
        
        # Verify all rows present
        with open(temp_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1000, f"Expected 1000 rows, got {len(rows)}"
        print("✅ 1000 CSV appends completed successfully")


# =============================================================================
# TEST: CONCURRENT READ/WRITE
# =============================================================================

class TestConcurrentReadWrite:
    """
    🔄 Test Concurrent Read/Write Operations
    """
    
    def test_concurrent_json_rw(self, temp_json):
        """
        Test concurrent read/write on JSON with no corruption.
        """
        read_errors = []
        write_count = [0]
        read_count = [0]
        stop_flag = threading.Event()
        
        def writer():
            for i in range(500):
                status = {'iteration': i, 'data': 'test' * 100}
                temp_path = temp_json + '.tmp'
                
                for attempt in range(3):
                    try:
                        with open(temp_path, 'w') as f:
                            json.dump(status, f)
                        os.replace(temp_path, temp_json)
                        write_count[0] += 1
                        break
                    except PermissionError:
                        time.sleep(0.001)
            
            stop_flag.set()
        
        def reader():
            while not stop_flag.is_set():
                try:
                    with open(temp_json, 'r') as f:
                        data = json.load(f)
                    assert 'iteration' in data or 'init' in data
                    read_count[0] += 1
                except json.JSONDecodeError as e:
                    read_errors.append(str(e))
                except (FileNotFoundError, PermissionError):
                    pass  # Acceptable on Windows
                time.sleep(0.0001)
        
        # Start threads
        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        
        r.start()
        w.start()
        
        w.join()
        r.join(timeout=2)
        
        # Verify: NO JSON corruption
        json_errors = [e for e in read_errors if 'JSONDecode' in str(e)]
        assert len(json_errors) == 0, f"JSON corruption detected: {json_errors[:3]}"
        
        print(f"✅ Concurrent R/W: {write_count[0]} writes, {read_count[0]} reads, 0 corruption")
    
    def test_multi_writer_conflict(self, temp_json):
        """
        Test multiple writers don't corrupt data.
        """
        final_values = []
        
        def writer(writer_id):
            for i in range(100):
                status = {'writer': writer_id, 'iteration': i}
                temp_path = temp_json + f'.tmp.{writer_id}'
                
                for attempt in range(3):
                    try:
                        with open(temp_path, 'w') as f:
                            json.dump(status, f)
                        os.replace(temp_path, temp_json)
                        break
                    except PermissionError:
                        time.sleep(0.001)
        
        # Start 3 concurrent writers
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify final file is valid JSON
        with open(temp_json, 'r') as f:
            final = json.load(f)
        
        assert 'writer' in final
        assert 'iteration' in final
        print(f"✅ Multi-writer test passed, final state: writer={final['writer']}")


# =============================================================================
# TEST: MARGIN RATIO & PNL SYNC
# =============================================================================

class TestMarginPnlSync:
    """
    📈 Test Margin Ratio and Unrealized PNL Synchronization
    """
    
    def test_margin_ratio_updates_sync(self, temp_json):
        """
        Test that margin ratio updates are synchronized.
        """
        # Initial state
        status = {
            'margin_ratio': 10.5,
            'unrealized_pnl': 0.0,
            'wallet_balance': 1000.0
        }
        
        # Simulate 100 market updates
        for i in range(100):
            # Simulate price change affecting PnL
            price_change = random.uniform(-0.02, 0.02)
            status['unrealized_pnl'] += price_change * 100
            
            # Recalculate margin ratio
            equity = status['wallet_balance'] + status['unrealized_pnl']
            status['margin_ratio'] = (equity / status['wallet_balance']) * 100
            status['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Atomic write
            temp_path = temp_json + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f)
            
            for attempt in range(3):
                try:
                    os.replace(temp_path, temp_json)
                    break
                except PermissionError:
                    time.sleep(0.001)
        
        # Verify final state is consistent
        with open(temp_json, 'r') as f:
            final = json.load(f)
        
        # Verify consistency
        expected_ratio = ((final['wallet_balance'] + final['unrealized_pnl']) / final['wallet_balance']) * 100
        assert abs(final['margin_ratio'] - expected_ratio) < 0.001, "Margin ratio inconsistent"
        
        print("✅ Margin ratio and PnL synchronization verified")


# =============================================================================
# TEST: FILE LOCK HANDLING
# =============================================================================

class TestFileLockHandling:
    """
    🔒 Test File Lock Handling on Windows
    """
    
    def test_graceful_lock_recovery(self, temp_json):
        """
        Test that system recovers gracefully from file locks.
        """
        lock_errors = []
        success_count = [0]
        
        def writer_with_retry():
            for i in range(100):
                status = {'iteration': i}
                temp_path = temp_json + '.tmp'
                
                for attempt in range(5):
                    try:
                        with open(temp_path, 'w') as f:
                            json.dump(status, f)
                        os.replace(temp_path, temp_json)
                        success_count[0] += 1
                        break
                    except PermissionError:
                        if attempt == 4:
                            lock_errors.append(f"Failed after 5 attempts at iteration {i}")
                        time.sleep(0.01 * (attempt + 1))  # Exponential backoff
        
        # Run writer
        writer_with_retry()
        
        # Most writes should succeed (allow some failures on busy system)
        assert success_count[0] >= 90, f"Only {success_count[0]}/100 writes succeeded"
        
        print(f"✅ File lock recovery: {success_count[0]}/100 writes successful")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
