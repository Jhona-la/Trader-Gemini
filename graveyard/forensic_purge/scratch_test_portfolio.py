import sys
import os
import asyncio
from core.portfolio import Portfolio

async def test():
    portfolio = Portfolio(initial_capital=13.0)
    print(f"Init Cash: {portfolio.current_cash}")
    print(f"Equity: {getattr(portfolio, '_equity_cache', 'None')}")
    
    avail = portfolio.get_available_cash(horizon="SCALPING")
    print(f"Available Cash (Scalping): {avail}")
    
    # Fake some trade history
    portfolio.trade_history = {
        'scalping': [
            {'net_pnl': 1.0}, {'net_pnl': -0.5},
            {'net_pnl': 1.5}, {'net_pnl': -0.2},
            {'net_pnl': 1.0}, {'net_pnl': -0.5},
            {'net_pnl': 1.5}, {'net_pnl': -0.2},
            {'net_pnl': 1.0}, {'net_pnl': -0.5},
        ]
    }
    kelly = portfolio.get_smart_kelly_sizing("BNB/USDT", "technical_v1", is_micro_account=True, horizon="SCALPING")
    print(f"Kelly Fraction (Micro): {kelly}")

if __name__ == "__main__":
    asyncio.run(test())
