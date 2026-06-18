import time
import pandas as np
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vectorized_backtest import run_vectorized_simulation, create_feature_matrices

def generate_mock_data(n_bars=5000):
    """Generate 5000 random bars to simulate 3.5 days of 1m data."""
    import pandas as pd
    
    np.random.seed(42)
    start_price = 65000.0
    returns = np.random.normal(0, 0.001, n_bars)
    close_prices = start_price * np.exp(np.cumsum(returns))
    high_prices = close_prices * (1 + np.random.uniform(0, 0.002, n_bars))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.002, n_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    return df

if __name__ == "__main__":
    print("🚀 Iniciando prueba de Velocidad del Quantum Vectorized Engine (C-Level Numba)...")
    
    df = generate_mock_data(10000) # 10,000 bars (~7 days of 1-minute data)
    print(f"✅ Generados {len(df)} eventos de mercado.")
    
    t0 = time.time()
    open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr = create_feature_matrices(df)
    t1 = time.time()
    print(f"⚡ Tiempo de Vectorización (Pandas/NumPy): {(t1-t0)*1000:.2f} ms")
    
    # Parámetros aleatorios
    sl_pct = 0.005
    tp_pct = 0.015
    rsi_oversold = 30.0
    rsi_overbought = 70.0
    ml_kelly_fraction = 1.0
    compounding_growth_factor = 1.0
    
    # PRIMERA EJECUCIÓN (LENTA) - Compilación de Python a Lenguaje Máquina LLVM
    print("🔨 Compilando a código C/Máquina (JIT)...")
    t_compile_start = time.time()
    res = run_vectorized_simulation(
        open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr,
        sl_pct, tp_pct, rsi_oversold, rsi_overbought,
        ml_kelly_fraction, compounding_growth_factor
    )
    t_compile_end = time.time()
    print(f"⏱️ Tiempo de Compilación Inicial: {(t_compile_end - t_compile_start)*1000:.2f} ms")
    
    # SEGUNDA EJECUCIÓN (VELOCIDAD CUÁNTICA)
    print("\n🚀 Ejecutando a velocidad de C (C-Level Array Processing)...")
    runs = 1000
    t_sim_start = time.time()
    for _ in range(runs):
        res = run_vectorized_simulation(
            open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr,
            sl_pct, tp_pct, rsi_oversold, rsi_overbought,
            ml_kelly_fraction, compounding_growth_factor
        )
    t_sim_end = time.time()
    
    total_ms = (t_sim_end - t_sim_start) * 1000
    ms_per_run = total_ms / runs
    
    print(f"\n📊 RESULTADOS (10,000 velas procesadas por run):")
    print(f"  - Tiempo Total ({runs} iteraciones): {total_ms:.2f} ms")
    print(f"  - Tiempo por Backtest Completo: {ms_per_run:.4f} ms")
    print(f"  - Velas procesadas por segundo: {10000 / (ms_per_run / 1000):,.0f} velas/segundo")
    print(f"  - Aceleración respecto a Event-Driven (~40s): ¡{40000 / ms_per_run:,.0f}x más rápido!")
    
    print(f"\n📈 PnL Simulado Final: ${res[0]:.2f} | Trades: {res[1]} | WinRate: {res[2]:.1f}%")
