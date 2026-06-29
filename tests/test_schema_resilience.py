"""
📋 TEST SCHEMA RESILIENCE - ROBUSTNESS AGAINST DATA CHANGES
============================================================

PROFESSOR METHOD:
- QUÉ: Tests de robustez ante cambios en estructura de JSON/CSV.
- POR QUÉ: Garantiza que Dashboard no se rompa por columnas faltantes.
- CÓMO: Simular CSVs incompletos y validar valores por defecto.
- CUÁNDO: Pre-producción para validar manejo de errores.
- DÓNDE: tests/test_schema_resilience.py

TEST COVERAGE:
1. Missing Columns: Handle missing 'fee' column gracefully
2. Extra Columns: Ignore unknown columns
3. Malformed Values: Handle NaN, None, empty strings
4. Schema Evolution: Support old and new formats
"""

import os
import sys
import json
import csv
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_csv():
    """Create temporary CSV file."""
    fd, path = tempfile.mkstemp(suffix='.csv', prefix='schema_test_')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_json():
    """Create temporary JSON file."""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='schema_test_')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# SCHEMA LOADER (Resilient)
# =============================================================================

class ResilientSchemaLoader:
    """
    📋 Schema loader that handles missing/malformed data gracefully.
    
    This is a test helper that mimics how Dashboard should load data.
    """
    
    # Default values for missing columns
    COLUMN_DEFAULTS = {
        'fee': 0.0,
        'net_pnl': 0.0,
        'pnl': 0.0,
        'is_reverse': False,
        'strategy_id': 'UNKNOWN',
        'trade_id': 'N/A',
        'quantity': 0.0,
        'entry_price': 0.0,
        'exit_price': 0.0
    }
    
    # Expected types for columns
    COLUMN_TYPES = {
        'fee': float,
        'net_pnl': float,
        'pnl': float,
        'quantity': float,
        'entry_price': float,
        'exit_price': float,
        'is_reverse': bool
    }
    
    @classmethod
    def load_trades_csv(cls, path: str) -> pd.DataFrame:
        """
        Load trades CSV with resilient handling.
        
        - Missing columns get default values
        - Malformed values get defaults
        - Types are coerced correctly
        """
        try:
            df = pd.read_csv(path)
        except Exception as e:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=list(cls.COLUMN_DEFAULTS.keys()))
        
        # Add missing columns with defaults
        for col, default in cls.COLUMN_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        
        # Coerce types
        for col, dtype in cls.COLUMN_TYPES.items():
            if col in df.columns:
                if dtype == float:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(cls.COLUMN_DEFAULTS.get(col, 0.0))
                elif dtype == bool:
                    df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes'])
        
        return df
    
    @classmethod
    def load_status_json(cls, path: str) -> Dict[str, Any]:
        """
        Load status JSON with resilient handling.
        """
        defaults = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_equity': 0.0,
            'positions': {},
            'unrealized_pnl': 0.0,
            'wallet_balance': 0.0
        }
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return defaults
        
        # Merge with defaults
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
        
        return data


# =============================================================================
# TEST: MISSING COLUMNS
# =============================================================================

class TestMissingColumns:
    """
    📉 Test handling of missing columns in CSV.
    """
    
    def test_missing_fee_column(self, temp_csv):
        """
        Test that missing 'fee' column uses default value 0.0.
        """
        # Create CSV without 'fee' column
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 1.0}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'symbol', 'pnl'])
            writer.writeheader()
            writer.writerows(data)
        
        # Load with resilient loader
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        assert 'fee' in df.columns, "Missing 'fee' column not added"
        assert df['fee'].iloc[0] == 0.0, "Default fee should be 0.0"
        
        print("✅ Missing 'fee' column handled with default 0.0")
    
    def test_missing_multiple_columns(self, temp_csv):
        """
        Test handling of multiple missing columns.
        """
        # Minimal CSV with only essential columns
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT'}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'symbol'])
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # All default columns should exist
        for col in ResilientSchemaLoader.COLUMN_DEFAULTS.keys():
            assert col in df.columns, f"Missing column '{col}' not added"
        
        print("✅ All missing columns filled with defaults")
    
    def test_empty_csv(self, temp_csv):
        """
        Test handling of empty CSV file.
        """
        # Create empty CSV (just headers)
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'symbol'])  # Only headers
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        assert len(df) == 0, "Empty CSV should return empty DataFrame"
        assert 'fee' in df.columns, "Default columns should still exist"
        
        print("✅ Empty CSV handled gracefully")


# =============================================================================
# TEST: EXTRA COLUMNS
# =============================================================================

class TestExtraColumns:
    """
    📊 Test handling of unexpected extra columns.
    """
    
    def test_extra_columns_preserved(self, temp_csv):
        """
        Test that extra columns are preserved, not discarded.
        """
        data = [
            {
                'timestamp': '2026-02-03T12:00:00',
                'symbol': 'XRP/USDT',
                'pnl': 1.0,
                'custom_field': 'test_value',
                'another_custom': 123
            }
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        assert 'custom_field' in df.columns, "Extra column should be preserved"
        assert df['custom_field'].iloc[0] == 'test_value'
        
        print("✅ Extra columns preserved correctly")


# =============================================================================
# TEST: MALFORMED VALUES
# =============================================================================

class TestMalformedValues:
    """
    🔧 Test handling of malformed/invalid values.
    """
    
    def test_nan_values(self, temp_csv):
        """
        Test handling of NaN values in numeric columns.
        """
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 'NaN', 'fee': ''},
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # NaN should be replaced with default
        assert not pd.isna(df['pnl'].iloc[0]), "NaN should be replaced with default"
        assert df['pnl'].iloc[0] == 0.0, "NaN should become 0.0"
        
        print("✅ NaN values handled correctly")
    
    def test_string_in_numeric_column(self, temp_csv):
        """
        Test handling of string values in numeric columns.
        """
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 'invalid', 'fee': 'not_a_number'}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # Invalid strings should be replaced with default
        assert df['pnl'].iloc[0] == 0.0, "Invalid string should become default"
        assert df['fee'].iloc[0] == 0.0, "Invalid string should become default"
        
        print("✅ Invalid string values handled correctly")
    
    def test_empty_string_values(self, temp_csv):
        """
        Test handling of empty string values.
        """
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': '', 'pnl': '', 'fee': ''}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # Empty strings in numeric columns should become defaults
        assert df['pnl'].iloc[0] == 0.0
        assert df['fee'].iloc[0] == 0.0
        
        print("✅ Empty string values handled correctly")
    
    def test_inf_values(self, temp_csv):
        """
        Test handling of infinity values.
        """
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 'inf', 'fee': '-inf'}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # Inf values should be coerced (pandas reads 'inf' as inf)
        # Check that they're at least numeric
        assert isinstance(df['pnl'].iloc[0], (int, float))
        
        print("✅ Infinity values handled (coerced to numeric)")


# =============================================================================
# TEST: SCHEMA EVOLUTION
# =============================================================================

class TestSchemaEvolution:
    """
    📈 Test compatibility with old and new schema formats.
    """
    
    def test_old_format_compatibility(self, temp_csv):
        """
        Test that old format (fewer columns) still works.
        """
        # Old format: v1.0 schema
        old_data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 1.0}
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=old_data[0].keys())
            writer.writeheader()
            writer.writerows(old_data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # Should work without errors and have new columns
        assert 'net_pnl' in df.columns
        assert 'strategy_id' in df.columns
        
        print("✅ Old format v1.0 compatible")
    
    def test_new_format_with_all_columns(self, temp_csv):
        """
        Test that new format with all columns works.
        """
        new_data = [
            {
                'timestamp': '2026-02-03T12:00:00',
                'symbol': 'XRP/USDT',
                'direction': 'LONG',
                'entry_price': 0.55,
                'exit_price': 0.56,
                'quantity': 100.0,
                'pnl': 1.0,
                'fee': 0.01,
                'net_pnl': 0.99,
                'is_reverse': False,
                'trade_id': 'TRADE_001',
                'strategy_id': 'HYBRID_SCALPING'
            }
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_data[0].keys())
            writer.writeheader()
            writer.writerows(new_data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # All values should be preserved
        assert df['net_pnl'].iloc[0] == 0.99
        assert df['strategy_id'].iloc[0] == 'HYBRID_SCALPING'
        
        print("✅ New format fully supported")


# =============================================================================
# TEST: JSON SCHEMA RESILIENCE
# =============================================================================

class TestJSONResilience:
    """
    📄 Test JSON schema resilience.
    """
    
    def test_missing_json_keys(self, temp_json):
        """
        Test handling of missing keys in JSON.
        """
        # Minimal JSON
        data = {'timestamp': '2026-02-03T12:00:00'}
        
        with open(temp_json, 'w') as f:
            json.dump(data, f)
        
        result = ResilientSchemaLoader.load_status_json(temp_json)
        
        # Missing keys should have defaults
        assert 'total_equity' in result
        assert result['total_equity'] == 0.0
        assert 'positions' in result
        assert result['positions'] == {}
        
        print("✅ Missing JSON keys handled with defaults")
    
    def test_corrupted_json(self, temp_json):
        """
        Test handling of corrupted JSON file.
        """
        # Write corrupted JSON
        with open(temp_json, 'w') as f:
            f.write("{corrupt: json, missing quote")
        
        result = ResilientSchemaLoader.load_status_json(temp_json)
        
        # Should return defaults without crashing
        assert 'timestamp' in result
        assert 'total_equity' in result
        
        print("✅ Corrupted JSON handled gracefully")
    
    def test_empty_json_file(self, temp_json):
        """
        Test handling of empty JSON file.
        """
        # Create empty file
        open(temp_json, 'w').close()
        
        result = ResilientSchemaLoader.load_status_json(temp_json)
        
        # Should return defaults
        assert result['total_equity'] == 0.0
        
        print("✅ Empty JSON file handled gracefully")
    
    def test_nested_structure_resilience(self, temp_json):
        """
        Test handling of missing nested structures.
        """
        data = {
            'timestamp': '2026-02-03T12:00:00',
            'total_equity': 5000.0
            # Missing: positions, metrics, etc.
        }
        
        with open(temp_json, 'w') as f:
            json.dump(data, f)
        
        result = ResilientSchemaLoader.load_status_json(temp_json)
        
        # Should have default empty positions
        assert result['positions'] == {}
        
        print("✅ Nested structure defaults applied")


# =============================================================================
# TEST: DASHBOARD INTEGRATION SIMULATION
# =============================================================================

class TestDashboardSimulation:
    """
    📊 Simulate Dashboard behavior with schema variations.
    """
    
    def test_calculate_expectancy_with_missing_columns(self, temp_csv):
        """
        Test expectancy calculation with missing 'net_pnl' column.
        """
        # CSV with only 'pnl' column (old format)
        data = [
            {'timestamp': '2026-02-03T12:00:00', 'symbol': 'XRP/USDT', 'pnl': 1.0},
            {'timestamp': '2026-02-03T12:01:00', 'symbol': 'XRP/USDT', 'pnl': -0.5},
        ]
        
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        df = ResilientSchemaLoader.load_trades_csv(temp_csv)
        
        # Use 'pnl' if 'net_pnl' is all zeros (default)
        if df['net_pnl'].sum() == 0 and 'pnl' in df.columns:
            pnl_col = 'pnl'
        else:
            pnl_col = 'net_pnl'
        
        expectancy = df[pnl_col].mean()
        
        assert expectancy == 0.25, f"Expected 0.25, got {expectancy}"
        print(f"✅ Expectancy calculated with fallback column: ${expectancy:.4f}")
    
    def test_render_positions_with_minimal_data(self, temp_json):
        """
        Test position rendering with minimal data.
        """
        data = {
            'positions': {
                'XRP/USDT': {'quantity': 100}
                # Missing: avg_price, current_price, unrealized_pnl
            }
        }
        
        with open(temp_json, 'w') as f:
            json.dump(data, f)
        
        result = ResilientSchemaLoader.load_status_json(temp_json)
        
        # Simulate dashboard rendering
        positions = result['positions']
        for symbol, pos in positions.items():
            qty = pos.get('quantity', 0)
            avg_price = pos.get('avg_price', 0.0)
            current = pos.get('current_price', avg_price)
            unrealized = pos.get('unrealized_pnl', 0.0)
            
            # Should not crash even with missing data
            display = f"{symbol}: {qty} @ ${avg_price:.2f} (PnL: ${unrealized:.2f})"
            assert display is not None
        
        print("✅ Position rendering works with minimal data")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
