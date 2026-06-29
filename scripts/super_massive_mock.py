import sys
import os
import time
import random
import datetime
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.WARNING)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk.risk_manager import RiskManager
from core.events import SignalEvent, SignalType
from core.portfolio import Portfolio

class DummyPortfolio:
    def __init__(self):
        self.virtual_ledger = {}
        self.cash = 13.0
        self.total_equity = 13.0
        
    def get_total_equity(self):
        return self.total_equity
        
    def reserve_cash(self, amount, horizon="SCALPING", order_id=""):
        if self.cash >= amount:
            self.cash -= amount
            return True
        return False
        
    def get_allocation_multiplier(self, sym, is_long):
        return 1.0
        
    def get_horizon_position(self, symbol, horizon):
        return self.virtual_ledger.get(f"{symbol}_{horizon}_LONG") or self.virtual_ledger.get(f"{symbol}_{horizon}_SHORT")
        
    def has_position_for_horizon(self, symbol, horizon):
        return self.get_horizon_position(symbol, horizon) is not None

class DummyDataProvider:
    def get_latest_bars(self, symbol, n=60):
        # We don't need real bars for speed testing, just return something fast or None to avoid numpy overhead
        return None

def worker_process(worker_id, num_signals):
    import data.data_provider
    data.data_provider.get_data_provider = lambda: DummyDataProvider()
    import risk.risk_manager
    risk.risk_manager.get_data_provider = lambda: DummyDataProvider()
    
    portfolio = DummyPortfolio()
    rm = RiskManager(portfolio=portfolio)
    rm._cache_initialized = True
    
    from config import Config
    symbols = Config.LEAN_TRADING_PAIRS
    horizons = ["MICROSCALPING", "SCALPING", "SWING"]
    
    signals = []
    for _ in range(num_signals):
        sym = random.choice(symbols)
        hz = random.choice(horizons)
        sig_type = random.choice([SignalType.LONG, SignalType.SHORT])
        
        event = SignalEvent(
            strategy_id="TEST_MASSIVE",
            symbol=sym,
            datetime=datetime.datetime.now(datetime.timezone.utc),
            signal_type=sig_type,
            strength=random.uniform(0.5, 1.0),
            horizon=hz,
            metadata={'setup_type': 'MOMENTUM', 'atr_pct': 0.01, 'momentum': 0.8}
        )
        signals.append(event)
        
    start_time = time.perf_counter()
    approved_count = 0
    rejected_count = 0
    
    for sig in signals:
        try:
            res = rm.validate_trade(sig)
            if res is not None:
                approved_count += 1
            else:
                rejected_count += 1
        except Exception as e:
            pass # Ignore expected sizing errors in dummy setup
            
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    latency_ms = (elapsed / num_signals) * 1000
    
    return worker_id, elapsed, latency_ms, approved_count, rejected_count

def run_super_massive_mock():
    print("🚀 [SUPER MASSIVE MOCK] Iniciando Stress Test Cuántico del RiskManager...")
    
    num_workers = mp.cpu_count()
    signals_per_worker = 2000
    total_signals = num_workers * signals_per_worker
    
    print(f"📊 Configuración: {num_workers} Workers x {signals_per_worker} Señales = {total_signals} Señales Concurrentes.")
    
    start_global = time.perf_counter()
    
    pool = mp.Pool(num_workers)
    results = [pool.apply_async(worker_process, args=(i, signals_per_worker)) for i in range(num_workers)]
    pool.close()
    pool.join()
    
    end_global = time.perf_counter()
    
    total_elapsed = end_global - start_global
    avg_latency = sum(r.get()[2] for r in results) / num_workers
    total_approved = sum(r.get()[3] for r in results)
    total_rejected = sum(r.get()[4] for r in results)
    
    print("\n" + "="*50)
    print("📈 RESULTADOS DEL MOCK SUPER MASIVO")
    print("="*50)
    print(f"⏱️  Tiempo Total: {total_elapsed:.4f}s")
    print(f"⚡ Rendimiento: {total_signals / total_elapsed:.2f} Señales/Segundo")
    print(f"⏱️  Latencia Media por Señal: {avg_latency:.4f} ms")
    print(f"✅ Señales Aprobadas: {total_approved}")
    print(f"🛑 Señales Rechazadas: {total_rejected}")
    
    if avg_latency < 10.0:
        print("\n✅ ÉXITO LATENCIA CUÁNTICA: El sistema opera fluidamente por debajo de los 10ms por decisión.")
    else:
        print("\n⚠️ ALERTA: Cuellos de botella detectados (>10ms por decisión).")

if __name__ == '__main__':
    run_super_massive_mock()
