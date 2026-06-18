import sys
import os
import time
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.events import SignalEvent, SignalType

def run_black_swan_mock():
    print("🚀 [MOCK] Iniciando Black Swan Kelly Resonance Test...")
    
    # 1. Setup minimal dependencies
    class DummyPortfolio:
        def __init__(self):
            # Posición abierta actual
            self.virtual_ledger = {
                "BTC/USDT_SCALPING_LONG": {"quantity": 0.05, "avg_price": 60000, "current_price": 60000}
            }
            self.cash = 13.0
            
        def get_total_equity(self):
            return 13.0
            
        def get_allocation_multiplier(self, sym, is_long):
            return 1.0

    class DummyDataProvider:
        def get_latest_bars(self, symbol, n=60):
            # Create dummy highly correlated data for BTC and ETH
            # Both crashing together (Black Swan)
            np.random.seed(42)
            rets_btc = np.random.uniform(-0.02, -0.01, size=n)
            rets_eth = rets_btc + np.random.uniform(-0.002, 0.002, size=n)
            
            if symbol == "BTC/USDT":
                rets = rets_btc
                base_price = 100.0
            else:
                rets = rets_eth
                base_price = 10.0
                
            prices = [base_price]
            for r in rets[1:]:
                prices.append(prices[-1] * (1 + r))
                
            return np.array(
                [(p,) for p in prices],
                dtype=[('close', 'f8')]
            )

    # Patch the get_data_provider method globally AND in risk_manager
    import data.data_provider
    data.data_provider.get_data_provider = lambda: DummyDataProvider()
    import risk.risk_manager
    risk.risk_manager.get_data_provider = lambda: DummyDataProvider()
    
    # Let's test the mock data right now to be sure!
    dp = data.data_provider.get_data_provider()
    tb = dp.get_latest_bars("ETH/USDT", 60)['close']
    ab = dp.get_latest_bars("BTC/USDT", 60)['close']
    tr = np.diff(tb)/tb[:-1]
    ar = np.diff(ab)/ab[:-1]
    print(f"[DEBUG] Correlation manual check: {np.corrcoef(tr, ar)[0,1]:.4f}")
    
    portfolio = DummyPortfolio()
    rm = RiskManager(portfolio=portfolio)
    rm._cache_initialized = True # Skip cache init
    
    print("\n[MOCK] Escenario: BTC está cayendo fuertemente (Crash sistémico). Tenemos posición abierta en BTC.")
    print("[MOCK] Se recibe una señal para comprar ETH/USDT (que también está cayendo).")
    
    # 2. Trigger Signal for ETH/USDT
    import datetime
    event = SignalEvent(
        strategy_id="TEST",
        symbol="ETH/USDT",
        datetime=datetime.datetime.now(datetime.timezone.utc),
        signal_type=SignalType.LONG,
        strength=1.0,
        ml_confidence=0.9,
        horizon="SCALPING"
    )
    import unittest.mock
    
    # 3. Validar con RiskManager
    print("\n[MOCK] Validando señal a través del RiskManager...")
    
    with unittest.mock.patch('risk.risk_manager.get_data_provider', return_value=DummyDataProvider()):
        with unittest.mock.patch('data.data_provider.get_data_provider', return_value=DummyDataProvider()):
            result = rm._validate_regime_veto(event.symbol, event.signal_type)
    
    print(f"\n[RESULTADO] ¿Señal aprobada?: {result}")
    
    if not result:
        print("✅ ÉXITO: La Matriz de Resonancia Kelly DETUVO la operación por alta correlación. Se evitó el margin call.")
    else:
        print("❌ FALLO: La señal pasó la validación de correlación.")

if __name__ == "__main__":
    run_black_swan_mock()
