import sys
import os

def calculate_expectancy(win_rate, tp_pct, sl_pct, fee_pct=0.0004, slippage_pct=0.0005, trail_to_be_prob=0.4):
    """
    Simula la esperanza matemática real del sistema actual.
    
    tp_pct: Take Profit percentaje (ej: 0.015)
    sl_pct: Stop Loss percentaje (ej: 0.006)
    trail_to_be_prob: Probabilidad de que el precio llegue a 50% de TP y se devuelva a Breakeven.
    fee_pct: Total fee ida y vuelta (Taker)
    slippage_pct: Slippage total ida y vuelta
    """
    
    # Probabilidad de Tocar TP real
    real_win_rate = win_rate * (1 - trail_to_be_prob)
    
    # Probabilidad de tocar Break-Even (y pagar fees!)
    be_rate = win_rate * trail_to_be_prob
    
    # Probabilidad de Loss
    loss_rate = 1.0 - win_rate
    
    # PnL neto por evento
    net_win = tp_pct - fee_pct - slippage_pct
    net_be = 0.001 - fee_pct - slippage_pct  # Breakeven en +0.1%
    net_loss = -sl_pct - fee_pct - slippage_pct
    
    expectancy = (real_win_rate * net_win) + (be_rate * net_be) + (loss_rate * net_loss)
    
    return expectancy, real_win_rate, be_rate, loss_rate, net_win, net_be, net_loss

if __name__ == "__main__":
    # Caso actual del Bot: WinRate Teórico 60%, TP 1.5%, SL 0.6%
    tp = 0.015
    sl = 0.006
    wr = 0.60
    
    exp, rw, be, lr, nw, nb, nl = calculate_expectancy(wr, tp, sl)
    
    print("=========================================")
    print("🧠 DIAGNÓSTICO MATEMÁTICO DE EXPECTATIVA")
    print("=========================================")
    print(f"TP Objetivo: +{tp*100:.2f}% | SL Objetivo: -{sl*100:.2f}%")
    print(f"Neto Ganancia (tras fees/slip): +{nw*100:.3f}%")
    print(f"Neto Break-Even (tras fees): {nb*100:.3f}%")
    print(f"Neto Pérdida (tras fees): {nl*100:.3f}%")
    print("---")
    print(f"Prob. Llegar al TP Real: {rw*100:.1f}%")
    print(f"Prob. Ahogado por Trailing a BE: {be*100:.1f}%")
    print(f"Prob. Tocar SL (Ruido/Loss): {lr*100:.1f}%")
    print("---")
    print(f"🔥 EXPECTATIVA MATEMÁTICA REAL POR TRADE: {exp*100:.4f}%")
    if exp < 0:
        print("❌ SISTEMA DISEÑADO PARA PERDER (Expectativa Negativa)")
    else:
        print("✅ SISTEMA DISEÑADO PARA GANAR")
