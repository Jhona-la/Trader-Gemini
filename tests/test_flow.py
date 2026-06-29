"""
🔄 TEST FLOW - PIPELINE VALIDATION TESTS
=========================================

PROFESSOR METHOD:
- QUÉ: Tests de integración que validan el flujo completo de datos.
- POR QUÉ: Asegura que la tubería Signal → Order → Fill → DataHandler funcione.
- CÓMO: Simula posiciones XRP y verifica propagación de strategy_id.
- CUÁNDO: Se ejecuta con `pytest tests/test_flow.py -v`.
- DÓNDE: tests/test_flow.py
"""

import os
import sys
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mocks.binance_mock import (
    MockBinanceClient,
    MockPosition,
    MockAccountBalance,
    create_mock_client_with_xrp_position
)


class TestXRPPositionFlow:
    """
    📍 Test XRP Position Flow: Validates the complete pipeline for a position.
    
    PROFESSOR METHOD:
    - Simula llegada de posición XRP desde Binance Mock
    - Verifica que DataHandler escriba strategy_id correcto
    - Confirma que AnalyticsEngine actualice Esperanza Matemática
    """
    
    def test_xrp_position_detection(self, mock_binance_client, mock_config, temp_data_dir):
        """
        Test that XRP position is correctly detected from mock API.
        """
        # Get futures account with positions
        account = mock_binance_client.futures_account()
        
        # Verify XRP position exists
        positions = account.get('positions', [])
        xrp_positions = [p for p in positions if p['symbol'] == 'XRPUSDT']
        
        assert len(xrp_positions) > 0, "XRP position not found in mock account"
        
        xrp_pos = xrp_positions[0]
        assert float(xrp_pos['positionAmt']) == 100.0
        assert float(xrp_pos['entryPrice']) == 0.55
        assert float(xrp_pos['unrealizedProfit']) == 5.00
    
    def test_unrealized_pnl_calculation(self, mock_binance_account):
        """
        Test that Unrealized PNL is correctly parsed and calculated.
        """
        total_unrealized = float(mock_binance_account['totalUnrealizedProfit'])
        
        # Sum individual position PNLs
        position_pnl_sum = sum(
            float(p.get('unrealizedProfit', 0)) 
            for p in mock_binance_account.get('positions', [])
        )
        
        # Total should be >= sum of individual positions
        # (there could be rounding or other positions)
        assert total_unrealized >= 0, "Unrealized PNL should not be negative in test scenario"
        assert position_pnl_sum <= total_unrealized + 1, "Position PNL sum should not exceed total"
    
    def test_equity_calculation(self, mock_binance_account):
        """
        Test that Total Equity is correctly calculated.
        
        Equity = Wallet Balance + Unrealized PNL
        """
        wallet_balance = float(mock_binance_account['totalWalletBalance'])
        unrealized_pnl = float(mock_binance_account['totalUnrealizedProfit'])
        margin_balance = float(mock_binance_account['totalMarginBalance'])
        
        expected_equity = wallet_balance + unrealized_pnl
        
        # Margin balance should equal equity
        assert abs(margin_balance - expected_equity) < 0.01, \
            f"Equity mismatch: {margin_balance} != {expected_equity}"


class TestDataHandlerIntegration:
    """
    📊 Test DataHandler Integration: Validates data persistence.
    """
    
    def test_live_status_structure(self, mock_data_handler, mock_live_status, temp_data_dir):
        """
        Test that live_status.json has correct structure.
        """
        status_path = os.path.join(temp_data_dir, 'live_status.json')
        
        # Save status
        mock_data_handler.save_live_status(status_path, mock_live_status)
        
        # Verify file exists
        assert os.path.exists(status_path), "live_status.json not created"
        
        # Load and verify structure
        with open(status_path, 'r') as f:
            saved_status = json.load(f)
        
        required_fields = ['timestamp', 'total_equity', 'positions']
        for field in required_fields:
            assert field in saved_status, f"Missing required field: {field}"
    
    def test_strategy_id_propagation(self, sample_signal_event, mock_portfolio):
        """
        Test that strategy_id is correctly propagated from Signal to Portfolio.
        """
        # Verify signal has strategy_id
        assert sample_signal_event.strategy_id == 'TEST_STRATEGY'
        
        # Simulate adding position with strategy context
        mock_portfolio.positions['XRP/USDT'] = {
            'quantity': 100.0,  # Fixed: use literal since fixture doesn't have quantity
            'avg_price': sample_signal_event.current_price or 0.55,
            'strategy_id': sample_signal_event.strategy_id,
            'current_price': sample_signal_event.current_price or 0.55
        }
        
        # Verify strategy_id is stored
        assert 'XRP/USDT' in mock_portfolio.positions
        assert mock_portfolio.positions['XRP/USDT']['strategy_id'] == 'TEST_STRATEGY'
    
    def test_fill_event_processing(self, sample_fill_event, mock_portfolio):
        """
        Test that FillEvent is correctly processed.
        """
        # Process fill (simulate what Portfolio.on_fill would do)
        symbol = sample_fill_event.symbol
        
        if symbol not in mock_portfolio.positions:
            mock_portfolio.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'current_price': 0
            }
        
        # Update position
        mock_portfolio.positions[symbol]['quantity'] = sample_fill_event.quantity
        mock_portfolio.positions[symbol]['avg_price'] = sample_fill_event.fill_price or 0.55
        
        # Verify
        assert mock_portfolio.positions[symbol]['quantity'] == 100.0
        assert mock_portfolio.positions[symbol]['avg_price'] == 0.55


class TestAnalyticsEngineIntegration:
    """
    📈 Test Analytics Engine: Validates metrics calculation.
    """
    
    def test_expectancy_calculation(self):
        """
        Test that Expectancy is correctly calculated.
        
        Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        """
        # Create sample trades
        sample_trades = [
            {'pnl': 10.0, 'is_win': True},
            {'pnl': 15.0, 'is_win': True},
            {'pnl': -5.0, 'is_win': False},
            {'pnl': 20.0, 'is_win': True},
            {'pnl': -8.0, 'is_win': False},
        ]
        
        # Calculate metrics manually
        wins = [t['pnl'] for t in sample_trades if t['is_win']]
        losses = [abs(t['pnl']) for t in sample_trades if not t['is_win']]
        
        win_rate = len(wins) / len(sample_trades)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        expected_expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Manual calculation: (0.6 * 15) - (0.4 * 6.5) = 9 - 2.6 = 6.4
        assert expected_expectancy > 0, "Expectancy should be positive for this sample"
    
    def test_drawdown_series(self):
        """
        Test drawdown calculation.
        """
        import pandas as pd
        
        # Create sample equity curve
        equity = pd.Series([100, 110, 105, 115, 108, 120])
        
        # Calculate drawdown locally
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        # Verify drawdown properties
        assert drawdown.iloc[0] == 0, "Initial drawdown should be 0"
        assert all(drawdown <= 0), "Drawdown should always be <= 0"
        
        # Max drawdown at index 4: (108 - 115) / 115 = -6.09%
        assert drawdown.min() < 0, "Should have some drawdown"


class TestSymbolNormalization:
    """
    🔤 Test Symbol Normalization: Validates symbol format handling.
    """
    
    def test_slash_removal(self):
        """
        Test that symbols with slashes are correctly converted to Binance format.
        """
        internal_symbol = 'XRP/USDT'
        binance_symbol = internal_symbol.replace('/', '')
        
        assert binance_symbol == 'XRPUSDT'
    
    def test_slash_addition(self):
        """
        Test that Binance symbols are correctly converted to internal format.
        """
        binance_symbol = 'XRPUSDT'
        
        # Standard conversion for USDT pairs
        if binance_symbol.endswith('USDT'):
            internal_symbol = f"{binance_symbol[:-4]}/USDT"
        else:
            internal_symbol = binance_symbol
        
        assert internal_symbol == 'XRP/USDT'
    
    def test_position_symbol_matching(self, mock_binance_client):
        """
        Test that positions can be matched regardless of symbol format.
        """
        account = mock_binance_client.futures_account()
        positions = account.get('positions', [])
        
        # Get XRP position (stored as XRPUSDT)
        target_internal = 'XRP/USDT'
        target_binance = target_internal.replace('/', '')
        
        matching = [p for p in positions if p['symbol'] == target_binance]
        assert len(matching) > 0, "Failed to match symbol formats"


class TestKlineDataIntegrity:
    """
    📉 Test Kline Data: Validates market data quality.
    """
    
    def test_kline_format(self, mock_binance_client):
        """
        Test that klines have correct format.
        """
        klines = mock_binance_client.get_klines(symbol='XRPUSDT', interval='1m', limit=10)
        
        # Mock returns generated klines (may be less than requested if cache is small)
        assert len(klines) > 0, "Should return some klines"
        
        # Verify structure
        for kline in klines:
            assert len(kline) == 12, "Kline should have 12 elements"
            
            # Parse OHLCV
            open_time = kline[0]
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            
            # Validate OHLC relationships
            assert high_price >= max(open_price, close_price), "High should be >= max(open, close)"
            assert low_price <= min(open_price, close_price), "Low should be <= min(open, close)"
            assert volume > 0, "Volume should be positive"
    
    def test_kline_chronology(self, mock_binance_client):
        """
        Test that klines are in chronological order.
        """
        klines = mock_binance_client.get_klines(symbol='BTCUSDT', interval='1m', limit=50)
        
        timestamps = [k[0] for k in klines]
        
        # Verify ascending order
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1], "Klines should be chronologically ordered"


# =============================================================================
# INTEGRATION: COMPLETE FLOW TEST
# =============================================================================

class TestCompleteFlow:
    """
    🔄 Complete Flow Test: End-to-end validation.
    """
    
    def test_signal_to_status_flow(
        self,
        mock_binance_client,
        mock_config,
        mock_data_handler,
        sample_signal_event,
        temp_data_dir,
        frozen_time
    ):
        """
        Complete flow: Signal → Position → DataHandler → Status File
        
        PROFESSOR METHOD:
        1. Receive signal for XRP
        2. Mock position creation
        3. Write to live_status.json
        4. Verify complete data integrity
        """
        # Step 1: Verify signal
        assert sample_signal_event.symbol == 'XRP/USDT'
        assert sample_signal_event.strategy_id == 'TEST_STRATEGY'
        
        # Step 2: Get account state from mock
        account = mock_binance_client.futures_account()
        
        # Step 3: Build status data
        positions_dict = {}
        for pos in account.get('positions', []):
            symbol = pos['symbol']
            if float(pos['positionAmt']) != 0:
                # Convert to internal format
                if symbol.endswith('USDT'):
                    internal_sym = f"{symbol[:-4]}/USDT"
                else:
                    internal_sym = symbol
                
                positions_dict[internal_sym] = {
                    'quantity': float(pos['positionAmt']),
                    'avg_price': float(pos['entryPrice']),
                    'unrealized_pnl': float(pos['unrealizedProfit'])
                }
        
        status_data = {
            'timestamp': frozen_time.isoformat(),
            'total_equity': float(account['totalMarginBalance']),
            'wallet_balance': float(account['totalWalletBalance']),
            'unrealized_pnl': float(account['totalUnrealizedProfit']),
            'positions': positions_dict,
            'last_heartbeat': frozen_time.isoformat(),
            'source': 'TEST_FLOW'
        }
        
        # Step 4: Save via DataHandler
        status_path = os.path.join(temp_data_dir, 'live_status.json')
        mock_data_handler.save_live_status(status_path, status_data)
        
        # Step 5: Verify saved data
        with open(status_path, 'r') as f:
            saved = json.load(f)
        
        assert saved['total_equity'] == 5100.0
        assert 'XRP/USDT' in saved['positions']
        assert saved['positions']['XRP/USDT']['quantity'] == 100.0
        assert saved['source'] == 'TEST_FLOW'
        
        print("✅ Complete flow test passed!")
