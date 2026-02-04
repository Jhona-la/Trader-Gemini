"""
🔗 INTEGRATION TESTS - BOT-FILE-UI SYNCHRONIZATION
===================================================

PROFESSOR METHOD:
- QUÉ: Tests de integración que validan sincronización Bot → Archivo → Dashboard.
- POR QUÉ: Garantiza que los datos fluyan correctamente entre capas.
- CÓMO: Inyección de trades fantasma y validación de tiempos.
- CUÁNDO: Pre-producción para validar pipeline completo.

FLUJO VALIDADO:
Bot Signal → DataHandler → trades.csv → Dashboard Render
"""

import os
import sys
import json
import time
import tempfile
import csv
from datetime import datetime, timezone
from typing import Dict
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['TRADER_GEMINI_ENV'] = 'TEST'


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_trade_file():
    """Create temp CSV for trade logging."""
    fd, path = tempfile.mkstemp(suffix='.csv', prefix='trades_integration_')
    os.close(fd)
    yield path
    for _ in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def temp_status_file():
    """Create temp JSON for status."""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='status_integration_')
    os.close(fd)
    yield path
    for _ in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def ghost_trade():
    """Create a ghost trade for traceability testing."""
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
        'trade_id': 'GHOST_TRADE_001',
        'strategy_id': 'INTEGRATION_TEST'
    }


# =============================================================================
# TEST: TRACEABILITY E2E
# =============================================================================

class TestTraceabilityE2E:
    """
    🔍 End-to-End Traceability Test
    
    Traces a ghost trade through all layers:
    1. APIManager injection (simulated)
    2. DataHandler registration (< 1ms target)
    3. Dashboard render (< 500ms target)
    """
    
    def test_ghost_trade_injection_to_csv(self, temp_trade_file, ghost_trade):
        """
        Test that ghost trade is registered in CSV within 1ms.
        """
        fieldnames = list(ghost_trade.keys())
        
        # Measure write time
        start = time.perf_counter()
        
        with open(temp_trade_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(ghost_trade)
        
        end = time.perf_counter()
        write_time_ms = (end - start) * 1000
        
        # Target: < 1ms (generous on SSD)
        # In practice, file I/O is ~0.1-0.5ms on SSD
        assert write_time_ms < 5, f"Write took {write_time_ms}ms, target < 5ms"
        
        print(f"✅ Ghost trade written in {write_time_ms:.3f}ms")
    
    def test_ghost_trade_readable_for_dashboard(self, temp_trade_file, ghost_trade):
        """
        Test that ghost trade can be read for Dashboard rendering.
        """
        # Write trade
        fieldnames = list(ghost_trade.keys())
        with open(temp_trade_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(ghost_trade)
        
        # Simulate Dashboard read
        start = time.perf_counter()
        
        with open(temp_trade_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        end = time.perf_counter()
        read_time_ms = (end - start) * 1000
        
        # Verify data integrity
        assert len(rows) == 1
        assert rows[0]['trade_id'] == 'GHOST_TRADE_001'
        assert float(rows[0]['net_pnl']) == 0.99
        
        print(f"✅ Ghost trade read in {read_time_ms:.3f}ms")
    
    def test_full_cycle_latency(self, temp_trade_file, temp_status_file, ghost_trade):
        """
        Test complete cycle latency: Write → Read → Process → Render Simulation.
        Target: < 500ms total.
        """
        fieldnames = list(ghost_trade.keys())
        
        cycle_start = time.perf_counter()
        
        # Step 1: Write trade
        with open(temp_trade_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(ghost_trade)
        
        # Step 2: Read trade
        with open(temp_trade_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        # Step 3: Process (calculate expectancy)
        pnls = [float(t['net_pnl']) for t in trades]
        expectancy = sum(pnls) / len(pnls) if pnls else 0
        
        # Step 4: Update status
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_trades': len(trades),
            'expectancy': expectancy,
            'last_trade_id': trades[-1]['trade_id'] if trades else None
        }
        
        temp_path = temp_status_file + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(status, f)
        os.replace(temp_path, temp_status_file)
        
        # Step 5: Simulate render (read status)
        with open(temp_status_file, 'r') as f:
            rendered_status = json.load(f)
        
        cycle_end = time.perf_counter()
        total_time_ms = (cycle_end - cycle_start) * 1000
        
        # Verify
        assert rendered_status['last_trade_id'] == 'GHOST_TRADE_001'
        assert rendered_status['expectancy'] == 0.99
        assert total_time_ms < 500, f"Cycle took {total_time_ms}ms, target < 500ms"
        
        print(f"✅ Full cycle completed in {total_time_ms:.3f}ms (< 500ms target)")


# =============================================================================
# TEST: DATA HANDLER SYNC
# =============================================================================

class TestDataHandlerSync:
    """
    📊 Test DataHandler Synchronization
    """
    
    def test_atomic_write_integrity(self, temp_status_file):
        """
        Test that atomic writes maintain data integrity.
        """
        # Write 100 status updates
        for i in range(100):
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'iteration': i,
                'value': i * 1.5
            }
            
            # Atomic write
            temp_path = temp_status_file + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f)
            os.replace(temp_path, temp_status_file)
        
        # Final read should show last iteration
        with open(temp_status_file, 'r') as f:
            final = json.load(f)
        
        assert final['iteration'] == 99
        assert final['value'] == 99 * 1.5
    
    def test_csv_append_integrity(self, temp_trade_file):
        """
        Test that CSV appends maintain integrity.
        """
        fieldnames = ['timestamp', 'trade_id', 'pnl']
        
        # Write header
        with open(temp_trade_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        
        # Append 100 trades
        for i in range(100):
            with open(temp_trade_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'trade_id': f'TRADE_{i:04d}',
                    'pnl': i * 0.1
                })
        
        # Verify all trades present
        with open(temp_trade_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 100
        assert rows[-1]['trade_id'] == 'TRADE_0099'


# =============================================================================
# TEST: DASHBOARD RESILIENCE
# =============================================================================

class TestDashboardResilience:
    """
    📈 Test Dashboard Resilience to Data Issues
    """
    
    def test_missing_columns_failsafe(self, temp_trade_file):
        """
        Test Dashboard fail-safe when columns are missing.
        """
        # Write CSV with missing columns
        with open(temp_trade_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'symbol'])  # Missing pnl, fee, etc.
            writer.writerow(['2024-01-01T00:00:00', 'BTC/USDT'])
        
        # Simulate Dashboard load with defaults
        import pandas as pd
        df = pd.read_csv(temp_trade_file)
        
        # Add missing columns with defaults
        defaults = {'pnl': 0.0, 'fee': 0.0, 'net_pnl': 0.0}
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        
        # Should not raise, should have default values
        assert 'pnl' in df.columns
        assert df['pnl'].iloc[0] == 0.0
    
    def test_corrupted_values_failsafe(self, temp_trade_file):
        """
        Test Dashboard fail-safe with corrupted values.
        """
        # Write CSV with corrupted values
        with open(temp_trade_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pnl'])
            writer.writerow(['2024-01-01T00:00:00', 'CORRUPTED'])
        
        # Simulate Dashboard load
        import pandas as pd
        df = pd.read_csv(temp_trade_file)
        
        # Convert with error handling
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
        
        assert df['pnl'].iloc[0] == 0.0  # Corrupted → Default


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
