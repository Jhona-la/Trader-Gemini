import time
import psutil
import os
import sys
import numpy as np
import polars as pl
from unittest.mock import MagicMock

# Añadir el path para importar
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from strategies.components.feature_engineering import FeatureEngineering
from utils.logger import logger

def measure_complexity():
    fe = FeatureEngineering()
    
    # Mock data provider
    data_provider = MagicMock()
    data_provider.get_order_flow_metrics.return_value = {}
    data_provider.get_derivatives_metrics.return_value = {}
    
    print("\n" + "="*50)
    print("🔬 TEST EMPÍRICO DE COMPLEJIDAD (O(N)) EN FEATURES")
    print("="*50)
    
    for n_bars in [100, 500, 1000, 5000]:
        # Generar datos simulados
        timestamps = np.arange(n_bars, dtype=np.float64) * 60000
        open_p = np.linspace(60000, 61000, n_bars)
        close_p = open_p + np.random.normal(0, 10, n_bars)
        high_p = np.maximum(open_p, close_p) + 5
        low_p = np.minimum(open_p, close_p) - 5
        volume_p = np.random.uniform(1, 100, n_bars)
        
        bars = np.core.records.fromarrays(
            [timestamps, open_p, high_p, low_p, close_p, volume_p],
            names='timestamp,open,high,low,close,volume'
        )
        
        # Test 1: Full computation
        start = time.perf_counter()
        df1 = fe.prepare_features(bars, data_provider=data_provider, symbol="BTCUSDT")
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Check C-Contiguity (El problema del Puente Roto)
        if isinstance(df1, pl.DataFrame):
            try:
                # Polars to Numpy
                np_arr = df1.to_numpy()
                is_contiguous = np_arr.flags['C_CONTIGUOUS']
                copy_msg = "❌ COPIA DE MEMORIA (Not Contiguous)" if not is_contiguous else "✅ ZERO-COPY"
            except Exception as e:
                copy_msg = f"Error: {e}"
        else:
            copy_msg = "Unknown"
            
        print(f"N_BARS: {n_bars:<5} | Latency: {latency_ms:>7.2f} ms | Mem: {copy_msg}")

def check_metal_loading():
    print("\n" + "="*50)
    print("🛡️ AUDITORÍA DE BINARIOS NATIVOS (RUST / CYTHON)")
    print("="*50)
    
    import importlib
    modules = [
        "core.rust_core.nano_core", 
        "core.nano_core", 
        "core.nano_portfolio", 
        "core.dark_alpha_queue",
        "core.mev_rbf_engine"
    ]
    
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            print(f"✅ {mod:<30} | Cargado desde: {m.__file__}")
            if m.__file__.endswith('.py'):
                print(f"   ❌ ALERTA: Fallback silencioso a Python Vanilla!")
            elif '.pyd' in m.__file__ or '.so' in m.__file__:
                print(f"   ⚡ BINARIO NATIVO ACTIVO")
        except ImportError as e:
            print(f"❌ {mod:<30} | FAILED TO LOAD: {e}")

if __name__ == "__main__":
    check_metal_loading()
    measure_complexity()
