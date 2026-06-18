import numpy as np

def calculate_omni_fitness(
    pnls: np.ndarray, 
    win_rate: float, 
    max_dd: float, 
    trades: int,
    starting_capital: float = 13.0
) -> float:
    """
    Función de supervivencia extrema para 13 USD.
    Calcula el Fitness Global basado en las métricas obtenidas.
    """
    if trades == 0 or len(pnls) == 0:
        return -9999.0
        
    current_cap = starting_capital
    lowest_cap = starting_capital
    
    current_losing_streak = 0
    max_losing_streak = 0
    
    wins = []
    losses = []
    
    for pnl_pct in pnls:
        # Trade size: usamos 50% del capital actual
        size_usd = current_cap * 0.50
        # Apalancado a 10x
        leveraged_size = size_usd * 10
        # Ganancia neta
        profit = leveraged_size * pnl_pct
        # Actualizamos capital
        current_cap += profit
        
        # Track Streaks & Win/Loss Arrays
        if pnl_pct > 0:
            current_losing_streak = 0
            wins.append(pnl_pct)
        else:
            current_losing_streak += 1
            if current_losing_streak > max_losing_streak:
                max_losing_streak = current_losing_streak
            losses.append(pnl_pct)
        
        if current_cap < lowest_cap:
            lowest_cap = current_cap
            
        if current_cap <= 5.0:
            # Bancarrota técnica para Binance (Mínimo trade ~5 USDT)
            return -9999.0

    # 1. FORENSIC METRIC: Max Losing Streak Penalty
    if max_losing_streak > 3:
        # Ruin risk for $13 account is extremely high if we lose > 3 times in a row.
        return -8000.0
            
    # Requisitos rigurosos para el Omni-Evolver
    if win_rate < 70.0:
        return -5000.0 + win_rate # Castigo fuerte si WR < 70%
        
    if lowest_cap < 8.0:
        # Penalizamos si en algún momento bajó a menos de 8 USD (Drawdown Severo)
        return -1000.0
        
    # 2. FORENSIC METRIC: Mathematical Expectancy
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    win_rate_dec = win_rate / 100.0
    loss_rate_dec = 1.0 - win_rate_dec
    
    expectancy = (win_rate_dec * avg_win) + (loss_rate_dec * avg_loss)
    
    # 3. FORENSIC METRIC: Net Margin Vulnerability (Slippage Immunity)
    # Average winner must be significantly above fees and slippage (e.g. 0.15% threshold)
    slippage_penalty = 0.0
    if avg_win < 0.0015:  # 0.15%
        # Penalizamos las ganancias microscópicas que no sobreviven a Binance
        slippage_penalty = -1000.0
        
    # 4. FORENSIC METRIC: Sortino Ratio
    # Penalizamos solo la volatilidad a la baja (downside deviation)
    downside_returns = [p for p in pnls if p < 0]
    downside_dev = np.std(downside_returns) if downside_returns else 0.0001
    if downside_dev == 0:
        downside_dev = 0.0001
        
    total_return = (current_cap - starting_capital) / starting_capital
    sortino_ratio = total_return / downside_dev
    
    # Score Base: Crecimiento Exponencial (Ratio de capital final)
    growth_ratio = current_cap / starting_capital
    
    # Bonos y Penalidades Finas
    trade_frequency_bonus = np.log1p(trades) * 2.0
    expectancy_bonus = expectancy * 10000.0  # Amplificamos la esperanza
    
    omni_score = (growth_ratio * 100.0) + trade_frequency_bonus + expectancy_bonus + slippage_penalty + (sortino_ratio * 10.0) - (max_dd * 500.0)
    
    return float(omni_score)
