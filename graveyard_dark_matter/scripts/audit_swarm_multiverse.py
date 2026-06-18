import os
import sys
import asyncio

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.portfolio import Portfolio
from core.events import SignalEvent
from core.enums import SignalType
from risk.risk_manager import RiskManager

async def audit_new_features():
    print("\n" + "="*50)
    print("🚀 AUDITORÍA FORENSE: SWARM VOLATILITY & MULTI-VERSE")
    print("="*50)
    
    portfolio = Portfolio()
    risk_mgr = RiskManager(portfolio=portfolio)
    
    print("\n1. Test Swarm Volatility (Allocation Weighting)")
    # Feed fake ATR data
    portfolio.atr_cache = {
        'BTC/USDT': 1500.0, # High Volatility
        'ETH/USDT': 50.0,   # Med Volatility
        'XRP/USDT': 0.05    # Low Volatility
    }
    portfolio.price_cache = {
        'BTC/USDT': 90000.0,
        'ETH/USDT': 3000.0,
        'XRP/USDT': 0.5
    }
    
    # Trigger update
    portfolio.update_relative_strength()
    print("\n   [Volatility Rankings]")
    print(f"   BTC/USDT Multiplier: {portfolio.get_allocation_multiplier('BTC/USDT', True):.4f}x (Expect high > 2.0x)")
    print(f"   ETH/USDT Multiplier: {portfolio.get_allocation_multiplier('ETH/USDT', True):.4f}x (Expect low < 1.0x)")
    print(f"   XRP/USDT Multiplier: {portfolio.get_allocation_multiplier('XRP/USDT', True):.4f}x (Expect very low)")
    
    print("\n2. Test Multi-Verse State Isolation (Hedge Mode)")
    # Simulate SWING Short
    portfolio.open_positions['BTC/USDT'] = {
        'SCALPING': {'quantity': 0, 'entry_price': 0, 'side': None},
        'SWING': {'quantity': -0.1, 'entry_price': 95000, 'side': 'SHORT'}
    }
    print("   Current positions: ", portfolio.open_positions['BTC/USDT'])
    
    # Try SCALPING Long Signal
    sig = SignalEvent('BTC/USDT', SignalType.LONG, 1.0, 90000, horizon='SCALPING', strategy_id='TEST')
    
    is_safe = risk_mgr.validate_entry(sig)
    print(f"\n   Is SCALPING LONG safe while SWING SHORT exists? {is_safe} (Expect True)")
    
    print("\n" + "="*50)
    print("✅ Auditoría Rápida Completada.")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(audit_new_features())
