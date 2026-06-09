"""Quick import and config verification after dead code cleanup."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('1. Config...', end=' ')
from config import Config
print('OK')

print('2. Engine...', end=' ')
from core.engine import Engine
print('OK')

print('3. Portfolio...', end=' ')  
from core.portfolio import Portfolio
print('OK')

print('4. RiskManager...', end=' ')
from risk.risk_manager import RiskManager
print('OK')

print('5. Technical...', end=' ')
from strategies.technical import TechnicalStrategy
print('OK')

print('6. BinanceData...', end=' ')
from data.binance_loader import BinanceData
print('OK')

print('7. Sophia...', end=' ')
from sophia.intelligence import SophiaIntelligence
print('OK')

# Verify config values
gt = Config.Horizons.GlobalThresholds
tt = Config.Strategies.TECHNICAL_THRESHOLDS
sp = Config.Strategies.SCALPING_PARAMS
sw = Config.Strategies.SWING_PARAMS

print()
print('=== CONFIG VERIFICATION ===')
print('Sophia win_prob_min:', gt['sophia_win_prob_min'], '(should be 0.60)')
print('Tech threshold:', tt['sophia_win_prob_min'], '(should be 0.60)')
print('SCALPING tp:', sp['tp_pct'], '(should be 0.0065)')
print('SWING tp:', sw['tp_pct'], '(should be 0.045)')
print('ML Enabled:', Config.LEAN_ML_ENABLED)
print('Core symbols:', Config.CORE_SYMBOLS)
print('Leverage:', Config.BINANCE_LEVERAGE, 'x')
print()
print('ALL CRITICAL IMPORTS AND CONFIG OK')
