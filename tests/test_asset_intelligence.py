import pytest
import time
from datetime import datetime, timezone

from core.events import SignalEvent, SignalType, SignalState
from core.asset_intelligence import get_asset_intelligence, AssetTier
from core.global_state import global_state

def test_tier_thresholds():
    """Verify that assets enforce their specific Tier minimum signal thresholds (A3)."""
    ai = get_asset_intelligence()
    
    # BTC minimum threshold is 0.58
    btc_low = SignalEvent(
        strategy_id="TFTF",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        strength=0.55,
        ml_confidence=0.55, # below 0.58
        metadata={"regime": "TRENDING"}
    )
    passed, reason = ai.verify_opening(btc_low, None)
    assert not passed
    assert "FAIL_A3" in reason
    
    btc_high = SignalEvent(
        strategy_id="TFTF",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        strength=0.62,
        ml_confidence=0.62, # above 0.58
        metadata={"regime": "TRENDING"}
    )
    passed, reason = ai.verify_opening(btc_high, None)
    assert passed
    
    # DOGE minimum threshold is 0.60
    doge_low = SignalEvent(
        strategy_id="TFTF",
        symbol="DOGE/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        strength=0.55,
        ml_confidence=0.55, # below 0.60
        metadata={"regime": "TRENDING"}
    )
    passed, reason = ai.verify_opening(doge_low, None)
    assert not passed
    assert "FAIL_A3" in reason

def test_strategy_compatibility():
    """Verify that forbidden/allowed strategies match the active asset profile (A4)."""
    ai = get_asset_intelligence()
    
    # Mean Reversion is allowed for BTC but forbidden for DOGE
    btc_mr = SignalEvent(
        strategy_id="MEAN_REVERSION",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        ml_confidence=0.75,
        metadata={"regime": "RANGING"}
    )
    passed, reason = ai.verify_opening(btc_mr, None)
    assert passed
    
    doge_mr = SignalEvent(
        strategy_id="MEAN_REVERSION",
        symbol="DOGE/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        ml_confidence=0.85,
        metadata={"regime": "RANGING"}
    )
    passed, reason = ai.verify_opening(doge_mr, None)
    assert not passed
    assert "FAIL_A1" in reason or "FAIL_A4" in reason

def test_outage_and_regulatory_blocks():
    """Verify that network outages and regulatory triggers block openings (A6)."""
    ai = get_asset_intelligence()
    
    # SOL network outage test
    sol_event = SignalEvent(
        strategy_id="TFTF",
        symbol="SOL/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        ml_confidence=0.75,
        metadata={"regime": "TRENDING"}
    )
    
    # When outage is inactive
    setattr(global_state, 'solana_network_outage', False)
    passed, reason = ai.verify_opening(sol_event, None)
    assert passed
    
    # When outage is active
    setattr(global_state, 'solana_network_outage', True)
    passed, reason = ai.verify_opening(sol_event, None)
    assert not passed
    assert "FAIL_A6" in reason
    assert "SOLANA" in reason
    setattr(global_state, 'solana_network_outage', False) # reset
    
    # DOGE sentiment catalyst test
    doge_event = SignalEvent(
        strategy_id="TFTF",
        symbol="DOGE/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        horizon="SCALPING",
        ml_confidence=0.82,
        metadata={"regime": "TRENDING"}
    )
    
    # When sentiment catalyst is active
    setattr(global_state, 'doge_sentiment_catalyst', True)
    passed, reason = ai.verify_opening(doge_event, None)
    assert passed
    
    # When sentiment catalyst is inactive
    setattr(global_state, 'doge_sentiment_catalyst', False)
    passed, reason = ai.verify_opening(doge_event, None)
    assert not passed
    assert "FAIL_A6" in reason
    assert "DOGE" in reason

def test_emergency_exits():
    """Verify that network outages and regulatory indicators trigger emergency closures (C7)."""
    ai = get_asset_intelligence()
    
    sol_position = {
        "symbol": "SOL/USDT",
        "quantity": 10.0,
        "avg_price": 100.0,
        "current_price": 99.0,
        "horizon": "SCALPING",
        "opener_strategy_id": "TFTF"
    }
    
    # Normal execution
    setattr(global_state, 'solana_network_outage', False)
    should_close, reason = ai.verify_closing(sol_position, 99.0, None, datetime.now(timezone.utc))
    assert not should_close
    
    # Under network outage
    setattr(global_state, 'solana_network_outage', True)
    should_close, reason = ai.verify_closing(sol_position, 99.0, None, datetime.now(timezone.utc))
    assert should_close
    assert "EMERGENCY_SOLANA" in reason
    setattr(global_state, 'solana_network_outage', False) # reset

def test_context_invalidation():
    """Verify that context changes trigger exits (C2)."""
    ai = get_asset_intelligence()
    
    btc_position = {
        "symbol": "BTC/USDT",
        "quantity": 0.5,
        "avg_price": 60000.0,
        "current_price": 60100.0,
        "horizon": "SCALPING",
        "opener_strategy_id": "TFTF",
        "last_adx_value": 25,
        "cvd_divergence_streak": 0
    }
    
    # ADX is healthy (25)
    should_close, reason = ai.verify_closing(btc_position, 60100.0, None, datetime.now(timezone.utc))
    assert not should_close
    
    # ADX drops below 20
    btc_position["last_adx_value"] = 18
    should_close, reason = ai.verify_closing(btc_position, 60100.0, None, datetime.now(timezone.utc))
    assert should_close
    assert "ADX" in reason
    
    # Reset ADX, check CVD streak
    btc_position["last_adx_value"] = 25
    btc_position["cvd_divergence_streak"] = 3
    should_close, reason = ai.verify_closing(btc_position, 60100.0, None, datetime.now(timezone.utc))
    assert should_close
    assert "CVD" in reason
