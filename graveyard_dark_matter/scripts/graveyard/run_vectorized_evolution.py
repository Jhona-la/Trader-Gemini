import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

# ==============================================================================
# 🚀 PHASE 35: QUANTUM VECTORIZED BACKTESTER & GENETIC EVOLUTION
# ==============================================================================
# Motor puro Desacoplado: No importa Config ni Logger para evitar FileLocks.
# Simulamos barras sintéticas de alta precisión.
# ==============================================================================

@njit(cache=True)
def vectorized_portfolio_simulation(
    prices,            # shape: (T, N)
    ml_signals,        # shape: (T, N) - from 0.0 to 1.0 (confidence)
    tech_signals,      # shape: (T, N) - -1, 0, 1
    maker_fee,         # float
    taker_fee,         # float
    initial_capital,   # float
    leverage,          # float
    max_positions,     # int
    kelly_fraction     # float (Aggressiveness for sizing)
):
    T, N = prices.shape
    capital = initial_capital
    
    positions = np.zeros(N, dtype=np.int32)
    entry_prices = np.zeros(N, dtype=np.float32)
    position_sizes = np.zeros(N, dtype=np.float32)
    pnl_history = np.zeros(T, dtype=np.float32)
    
    ML_LONG_THRESH = 0.65
    ML_SHORT_THRESH = 0.35
    
    for t in range(1, T):
        active_pos_count = np.sum(np.abs(positions))
        
        for i in range(N):
            current_price = prices[t, i]
            pos = positions[i]
            
            if pos != 0:
                pnl_pct = 0.0
                if pos == 1:
                    pnl_pct = (current_price - entry_prices[i]) / entry_prices[i]
                elif pos == -1:
                    pnl_pct = (entry_prices[i] - current_price) / entry_prices[i]
                
                tp = 0.02
                sl = -0.01
                exit_signal = False
                
                if pos == 1 and ml_signals[t, i] < 0.45:
                    exit_signal = True
                elif pos == -1 and ml_signals[t, i] > 0.55:
                    exit_signal = True
                elif pnl_pct >= tp or pnl_pct <= sl:
                    exit_signal = True
                    
                if exit_signal:
                    gross_pnl = position_sizes[i] * pnl_pct * leverage
                    fee = (position_sizes[i] * leverage) * taker_fee
                    net_pnl = gross_pnl - fee
                    capital += net_pnl
                    
                    positions[i] = 0
                    entry_prices[i] = 0.0
                    position_sizes[i] = 0.0
                    active_pos_count -= 1
            
            if positions[i] == 0 and active_pos_count < max_positions:
                conf = ml_signals[t, i]
                tech = tech_signals[t, i]
                
                intent = 0
                if conf > ML_LONG_THRESH and tech >= 0:
                    intent = 1
                elif conf < ML_SHORT_THRESH and tech <= 0:
                    intent = -1
                    
                if intent != 0:
                    bet_size = capital * kelly_fraction
                    if bet_size > capital * 0.5:
                        bet_size = capital * 0.5
                        
                    positions[i] = intent
                    entry_prices[i] = current_price
                    position_sizes[i] = bet_size
                    
                    fee = (bet_size * leverage) * maker_fee
                    capital -= fee
                    active_pos_count += 1
                    
        pnl_history[t] = capital
        if capital <= 0:
            pnl_history[t:] = 0.0
            break
            
    return pnl_history

def build_synthetic_matrix(T=8640, N=15):
    """
    Genera una simulación sintética muy realista de T barras para N monedas.
    8640 barras de 5 minutos = 30 días exactos.
    """
    print(f"⏳ Construyendo Synthetic Matrix Global ({T} ticks, {N} monedas)...")
    t0 = time.time()
    
    prices = np.zeros((T, N), dtype=np.float32)
    # Start all prices at $100
    prices[0, :] = 100.0
    
    # Generate random walk with volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, size=(T, N))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0)).astype(np.float32)
    
    # Generate mock ML signals correlated with future returns (simulating an edge)
    ml_signals = np.full((T, N), 0.5, dtype=np.float32)
    tech_signals = np.zeros((T, N), dtype=np.int8)
    
    # Introduce alpha: ML predicts slightly correctly
    # If next return is positive, higher chance of conf > 0.65
    future_ret = np.roll(returns, shift=-1, axis=0)
    future_ret[-1, :] = 0.0
    
    # Alpha = 0.1 (10% edge)
    alpha_signal = future_ret * 50 + 0.5 
    ml_signals = np.clip(alpha_signal + np.random.normal(0, 0.2, size=(T,N)), 0.0, 1.0).astype(np.float32)
    
    # Tech signals based on momentum
    momentum = np.roll(returns, shift=1, axis=0)
    momentum[0, :] = 0.0
    tech_signals[momentum > 0.001] = 1
    tech_signals[momentum < -0.001] = -1
    
    print(f"✅ Matrix construida en {time.time() - t0:.2f}s. Shape: {prices.shape}")
    return prices, tech_signals, ml_signals, T, N

def run_vectorized_evaluation(params):
    kelly_fraction = params['kelly_fraction']
    max_positions = params['max_positions']
    
    global GLOBAL_PRICES, GLOBAL_ML_SIGNALS, GLOBAL_TECH_SIGNALS
    
    t0 = time.time()
    pnl = vectorized_portfolio_simulation(
        GLOBAL_PRICES,
        GLOBAL_ML_SIGNALS,
        GLOBAL_TECH_SIGNALS,
        maker_fee=0.0, 
        taker_fee=0.0005,
        initial_capital=13.0,
        leverage=10.0,
        max_positions=max_positions,
        kelly_fraction=kelly_fraction
    )
    t1 = time.time()
    
    final_capital = pnl[-1]
    return final_capital, (t1 - t0)

GLOBAL_PRICES = None
GLOBAL_ML_SIGNALS = None
GLOBAL_TECH_SIGNALS = None

def init_worker(prices, ml, tech):
    global GLOBAL_PRICES, GLOBAL_ML_SIGNALS, GLOBAL_TECH_SIGNALS
    GLOBAL_PRICES = prices
    GLOBAL_ML_SIGNALS = ml
    GLOBAL_TECH_SIGNALS = tech

def run_massive_evolution():
    print("🧬 Iniciando Motor de Evolución Cuántica (Fase 35) 🧬")
    print("📡 Modo Sintético: Desacoplado para evitar interrupción del Bot en vivo")
    
    # 30 días, 15 monedas
    prices, tech_signals, ml_signals, T, N = build_synthetic_matrix(T=8640, N=15)
    
    # Compile the JIT by running it once
    print("⚙️ Compilando núcleo Numba (Warm-up JIT)...")
    _ = vectorized_portfolio_simulation(prices, ml_signals, tech_signals, 0.0, 0.0005, 13.0, 10.0, 5, 0.1)
    
    print("🚀 Iniciando Montecarlo Genético Massivo (1,000 Generaciones)...")
    
    generations = []
    for _ in range(1000):
        generations.append({
            'kelly_fraction': np.random.uniform(0.01, 0.20),
            'max_positions': np.random.randint(2, 10)
        })
        
    best_capital = 0
    best_params = None
    
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(prices, ml_signals, tech_signals)) as executor:
        futures = {executor.submit(run_vectorized_evaluation, g): g for g in generations}
        
        count = 0
        for future in as_completed(futures):
            final_cap, exec_time = future.result()
            params = futures[future]
            
            if final_cap > best_capital:
                best_capital = final_cap
                best_params = params
                
            count += 1
            if count % 100 == 0:
                print(f"🧬 Evaluados {count}/1000... Mejor Capital: ${best_capital:.2f} (Kelly: {best_params['kelly_fraction']:.2f})")
                
    total_time = time.time() - t_start
    print("="*60)
    print(f"🏆 SANTO GRIAL ENCONTRADO 🏆")
    print(f"Mejor Capital Final: ${best_capital:.2f} (desde $13.0)")
    print(f"Crecimiento: {((best_capital - 13.0)/13.0)*100:.2f}%")
    print(f"Parámetros Ganadores: {best_params}")
    print(f"Tiempo Total de Simulación: {total_time:.2f}s para 1000 backtests de 30 días (15 símbolos)")
    print(f"Velocidad: {(total_time/1000)*1000:.2f} ms por backtest completo")
    print("="*60)

if __name__ == "__main__":
    run_massive_evolution()
