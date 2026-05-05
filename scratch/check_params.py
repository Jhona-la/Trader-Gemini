import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'
from config import Config
Config.TELEGRAM_ENABLED = False
Config.EMAIL_ENABLED = False
Config.DISCORD_ENABLED = False
if hasattr(Config, 'Observability'):
    Config.Observability.TELEGRAM_ENABLED = False
    Config.Observability.DISCORD_ENABLED = False
    Config.Observability.EMAIL_ENABLED = False

from core.portfolio import Portfolio
from risk.risk_manager import RiskManager

p = Portfolio(initial_capital=13.0, auto_save=False)
rm = RiskManager(max_concurrent_positions=2, portfolio=p)
hp = rm.horizon_params['SCALPING']
print("horizon_params SCALPING:")
print(f"  stop_loss_pct: {hp['stop_loss_pct']}  ({hp['stop_loss_pct']*100:.2f}%)")
print(f"  take_profit_pct: {hp['take_profit_pct']}  ({hp['take_profit_pct']*100:.2f}%)")
print(f"  leverage: {hp['leverage']}")
print(f"Config sl_pct: {Config.Strategies.SCALPING_PARAMS['sl_pct']}")
print(f"Config tp_pct: {Config.Strategies.SCALPING_PARAMS['tp_pct']}")
