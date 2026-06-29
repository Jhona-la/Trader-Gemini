
@njit(parallel=True, fastmath=True)
def monte_carlo_future_projection(
    current_price: float,
    bar_volatility: float,
    drift: float,
    tp_pct: float,
    sl_pct: float,
    direction: int,
    max_steps: int,
    n_simulations: int
) -> float:
    """
    Simulador Monte Carlo Brownian Motion.
    Proyecta n_simulations trayectorias futuras de precios durante max_steps velas.
    Retorna la probabilidad de éxito (Win Rate estimado) del trade.
    """
    wins = 0
    
    tp_price = current_price * (1.0 + (tp_pct * direction))
    sl_price = current_price * (1.0 - (sl_pct * direction))
    
    # Pre-calcular factores fijos para velocidad extrema
    sqrt_dt = 1.0 # 1 step = 1 bar
    
    for i in prange(n_simulations):
        price = current_price
        hit_win = False
        hit_loss = False
        
        # Generar aleatorios para toda la trayectoria de una vez (para esta simulación)
        for step in range(max_steps):
            # Geometric Brownian Motion step
            # Z = variable normal estandar
            # random_step genera un float pseudo-aleatorio. Como Numba no tiene random normal directo 
            # en modo parallel sin lock, podemos usar una aproximación o simplemente uniform convertida
            # Numba's np.random.normal es thread-safe en prange si se maneja la semilla.
            Z = np.random.normal(0.0, 1.0)
            
            # S_t = S_{t-1} * exp((mu - sigma^2 / 2)dt + sigma * Z * sqrt(dt))
            # Para velocidades extremas usamos aproximación lineal simple para velas pequeñas:
            price_change = price * (drift + bar_volatility * Z)
            price += price_change
            
            if direction == 1:
                if price >= tp_price:
                    hit_win = True
                    break
                elif price <= sl_price:
                    hit_loss = True
                    break
            else: # SHORT
                if price <= tp_price:
                    hit_win = True
                    break
                elif price >= sl_price:
                    hit_loss = True
                    break
                    
        # Si sobrevive el tiempo máximo sin tocar SL, lo evaluamos por PnL flotante final
        if not hit_win and not hit_loss:
            if direction == 1:
                if price > current_price:
                    hit_win = True
            else:
                if price < current_price:
                    hit_win = True
                    
        if hit_win:
            wins += 1
            
    return float(wins) / float(n_simulations)
