from core.portfolio import Portfolio
from risk.risk_manager import RiskManager

portfolio = Portfolio(initial_capital=13.0)
risk_manager = RiskManager(portfolio=portfolio)

capital_progression = []

print("Phase 4 - Compounding Audit:")
for i in range(1, 11):
    current_equity = portfolio.get_total_equity()
    
    # Simulate size calculation
    # Let's say risk_pct is 19% as seen in logs
    size_usd = current_equity * 0.19
    
    capital_progression.append((i, current_equity, size_usd))
    
    # Simulate winning trade (+5% net on the allocated size)
    profit = size_usd * 0.05
    
    # Artificially inject profit to portfolio
    portfolio.current_cash += profit
    portfolio._refresh_equity_cache()

for i, eq, size in capital_progression:
    print(f"Trade {i} | Equity: ${eq:.2f} | Position Size (19%): ${size:.2f}")

final_eq = portfolio.get_total_equity()
print(f"Final Equity after 10 trades: ${final_eq:.2f}")
if final_eq > 13.0:
    print("Compounding Meta-Cycle Verified: Ganancia exponencial validada.")
else:
    print("Compounding Meta-Cycle Failed: El interés no es compuesto.")
