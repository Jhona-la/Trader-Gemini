import math

def calculate_exponential_path(initial_capital: float, target_capital: float, days: int, trades_per_day: int, win_rate: float, fee_rate: float, leverage: float):
    print("="*60)
    print(f"🚀 OMNISCIENT COMPOUNDING ENGINE - {days} DAY CHALLENGE")
    print("="*60)
    print(f"Capital Inicial: ${initial_capital:.2f}")
    print(f"Capital Objetivo: ${target_capital:.2f}")
    print(f"Días: {days}")
    print(f"Trades Diarios Estimados: {trades_per_day}")
    print(f"Total Trades Estimados: {days * trades_per_day}")
    print(f"Apalancamiento: {leverage}x")
    print(f"Win Rate Estimado: {win_rate*100:.1f}%")
    print(f"Taker Fee (Binance): {fee_rate*100:.3f}% (x2 para ida y vuelta = {fee_rate*2*100:.3f}%)")
    print("-" * 60)

    # Calculate required compound growth per trade
    # Final = Initial * (1 + growth_per_trade) ^ total_trades
    total_trades = days * trades_per_day
    required_total_multiplier = target_capital / initial_capital
    required_growth_per_trade = math.pow(required_total_multiplier, 1/total_trades) - 1
    
    print(f"Crecimiento NETO Requerido por Trade: +{required_growth_per_trade*100:.4f}% de la cuenta")
    
    # Let's say we risk 'R' per trade.
    # Expected Value per trade = (Win Rate * Reward) - (Loss Rate * Risk)
    # We want Expected Value = required_growth_per_trade
    # Reward = Risk * Reward_Ratio
    # EV = (WR * Risk * RR) - ((1 - WR) * Risk)
    # EV = Risk * ((WR * RR) - (1 - WR))
    
    # Let's fix Risk to 1.5% of the account (very safe for $13, prevents liquidations)
    risk_pct = 0.015 
    loss_rate = 1 - win_rate
    
    if win_rate < 1.0:
        required_rr = (required_growth_per_trade / risk_pct + loss_rate) / win_rate
    else:
        required_rr = (required_growth_per_trade / risk_pct) / win_rate
        
    reward_pct = risk_pct * required_rr
    
    print(f"\n🎯 PARÁMETROS ESTRICTOS DE GESTIÓN (Con {risk_pct*100:.1f}% Risk per Trade):")
    print(f"  Risk (Stop Loss de cuenta): -{risk_pct*100:.2f}%")
    print(f"  Reward (Target Neto de cuenta): +{reward_pct*100:.4f}%")
    print(f"  Risk/Reward Requerido: 1 : {required_rr:.2f}")
    
    # Translate account growth to actual price movement using leverage
    # If we use 100% of our balance as margin (not recommended, but it's $13)
    margin_utilization = 0.80 # use 80% of $13 to leave room for fees/fluctuations
    
    # True target price movement
    # reward_pct (account) = (price_movement * leverage * margin_utilization)
    price_movement_target = reward_pct / (leverage * margin_utilization)
    price_movement_stop = risk_pct / (leverage * margin_utilization)
    
    print(f"\n📊 PARÁMETROS EN EL MERCADO (con {leverage}x y {margin_utilization*100:.0f}% Margin):")
    print(f"  Movimiento Precio TP (Neto): +{price_movement_target*100:.4f}%")
    print(f"  Movimiento Precio SL: -{price_movement_stop*100:.4f}%")
    
    # Add fees to find the BRUTO movement needed to hit the NETO target
    # 2x fee because entry and exit
    total_fee_impact = fee_rate * 2
    bruto_target = price_movement_target + total_fee_impact
    
    print(f"\n⚖️ AJUSTE DE COMISIONES (La cruda realidad):")
    print(f"  Para ganar +{price_movement_target*100:.4f}% limpio, el mercado debe moverse:")
    print(f"  TARGET BRUTO REQUERIDO (TP): +{bruto_target*100:.4f}%")
    print(f"  (Esto es el Minimum Viable Net + Expected Growth)")
    print("="*60)

if __name__ == "__main__":
    # Scenarios
    # Scalping heavy: 15 trades a day, 80% win rate
    calculate_exponential_path(
        initial_capital=13.0, 
        target_capital=26.0, 
        days=15, 
        trades_per_day=15, 
        win_rate=0.80, 
        fee_rate=0.0004, # 0.04% taker VIP 0
        leverage=10.0
    )
    
    # Perfect Scalping (as user requested "100% de WR en scalping")
    calculate_exponential_path(
        initial_capital=13.0, 
        target_capital=26.0, 
        days=15, 
        trades_per_day=10, 
        win_rate=1.0, 
        fee_rate=0.0004, 
        leverage=10.0
    )
