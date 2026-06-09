import pytest
import numpy as np
from datetime import datetime, timezone
from config import Config
from strategies.technical import HybridScalpingStrategy
from core.events import SignalEvent
from core.enums import SignalType, TimeFrame
from core.genotype import Genotype

class MockDataProvider:
    def __init__(self):
        self.symbol_list = ['BTC/USDT']
        
    def get_latest_bars(self, symbol, n=100, timeframe='5m'):
        # Generate some mock data that triggers a setup
        close_prices = np.linspace(50000, 51000, n)
        high_prices = close_prices + 100
        low_prices = close_prices - 100
        volume = np.random.uniform(5, 15, n)
        
        # Add some volatility at the end to ensure ATR > 0
        close_prices[-5:] = [50800, 50900, 50850, 51000, 50950]
        
        return {
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'open': close_prices - 50,
            'volume': volume
        }
        
    def get_active_positions(self):
        return {}
        
    def get_order_flow_metrics(self, symbol):
        return {'is_toxic': False}

class MockQueue:
    def __init__(self):
        self.items = []
        
    def put(self, item):
        self.items.append(item)

def test_horizon_config_loading():
    """Test that SCALPING and SWING load different parameters from Config"""
    dp = MockDataProvider()
    q_scalp = MockQueue()
    q_swing = MockQueue()
    
    # Instantiate both strategies with different horizons
    strat_scalp = HybridScalpingStrategy(dp, q_scalp, Genotype('BTC/USDT'), horizon='SCALPING')
    strat_swing = HybridScalpingStrategy(dp, q_swing, Genotype('BTC/USDT'), horizon='SWING')
    
    # 1. Verify TP/SL limits are different
    assert strat_scalp.TP_PCT < strat_swing.TP_PCT, "Scalping TP should be smaller than Swing TP"
    assert strat_scalp.SL_PCT < strat_swing.SL_PCT, "Scalping SL should be smaller than Swing SL"
    
    # 2. Verify Primary Timeframe
    assert strat_scalp.PRIMARY_TF == '1m'  # Config.Horizons.Scalping['primary_tf'] = '1m' (HFT)
    assert strat_swing.PRIMARY_TF == '1h'
    
    print(f"Scalp: TP={strat_scalp.TP_PCT*100}%, SL={strat_scalp.SL_PCT*100}%")
    print(f"Swing: TP={strat_swing.TP_PCT*100}%, SL={strat_swing.SL_PCT*100}%")

def test_dynamic_risk_calculation():
    """Test that the dynamic risk calculator respects the horizon bounds"""
    dp = MockDataProvider()
    q = MockQueue()
    
    strat_scalp = HybridScalpingStrategy(dp, q, Genotype('BTC/USDT'), horizon='SCALPING')
    strat_swing = HybridScalpingStrategy(dp, q, Genotype('BTC/USDT'), horizon='SWING')
    
    # Mock indicators dict
    inds = {
        'atr': np.array([0.0] * 50 + [100.0]), # ATR of 100 on price 50000 = 0.2%
        'rsi': np.array([50] * 50)
    }
    
    # Get dynamic bounds
    sl_mult, tp_mult, scalp_sl, scalp_tp = strat_scalp._calculate_dynamic_risk_params(
        inds, 50000, setup_type="MEAN_REV", regime="RANGING"
    )
    
    _, _, swing_sl, swing_tp = strat_swing._calculate_dynamic_risk_params(
        inds, 50000, setup_type="MEAN_REV", regime="RANGING"
    )
    
    assert scalp_tp < swing_tp, "Dynamic Scalping TP must be tighter than Swing TP"
    assert scalp_sl < swing_sl, "Dynamic Scalping SL must be tighter than Swing SL"
    
    print(f"Dynamic Scalp: TP={scalp_tp*100:.3f}%, SL={scalp_sl*100:.3f}%")
    print(f"Dynamic Swing: TP={swing_tp*100:.3f}%, SL={swing_sl*100:.3f}%")
