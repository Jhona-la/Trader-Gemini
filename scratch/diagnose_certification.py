import sys
import os
sys.path.append(os.getcwd())

from risk.risk_manager import RiskManager
from core.events import SignalEvent
from core.enums import SignalType
from datetime import datetime, timezone
import logging

# Set up logging to print to stdout so we can see what's happening
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
for name in logging.root.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.DEBUG)

# Replicate MockPortfolio
class MockPortfolio:
    def __init__(self):
        self.positions = {}
        self.virtual_ledger = {}
        self.current_cash = 13.0
        self.used_margin = 0.0
        self.pending_cash = 0.0
        self.db = None
        self._last_prices = {}
        self.relative_strength_scores = {}
    def get_horizon_position(self, symbol, horizon):
        return None
    def get_total_equity(self):
        return 10000.0
    def get_available_cash(self, horizon="SCALPING"):
        return 10000.0
    def reserve_cash(self, amount, horizon="SCALPING", order_id=""):
        print(f"[MOCK PORTFOLIO] reserve_cash called: amount={amount}")
        return True
    def has_position_for_horizon(self, symbol, horizon):
        return False
    def get_smart_kelly_sizing(self, symbol, strategy_id, is_micro, horizon):
        return 0.1
    def get_setup_performance(self, setup_type):
        return {"win_rate": 0.6}
    def get_strategy_metrics(self, strategy_id):
        return {"merit_factor": 1.0}
    def get_allocation_multiplier(self, symbol, is_long):
        return 1.0
    def get_statistics(self):
        return {}

class FastKillSwitch:
    def check_status(self): return True
    activation_reason = ""

class FastSHSMonitor:
    def get_shs(self): return 100.0

risk_manager = RiskManager()
risk_manager.portfolio = MockPortfolio()
risk_manager.kill_switch = FastKillSwitch()
risk_manager.shs_monitor = FastSHSMonitor()

# Let's bypass cooldowns
from utils.cooldown_manager import cooldown_manager
cooldown_manager.SCALPING_SYMBOL_COOLDOWN = 0.0
cooldown_manager.SCALPING_PATTERN_COOLDOWN = 0.0
cooldown_manager.GLOBAL_COOLDOWN = 0.0
cooldown_manager.STRATEGY_COOLDOWN = 0.0

# Let's create a SignalEvent like the one in certification_of_perfection.py
sig = SignalEvent(
    strategy_id="[SCL]_HYBRID_SCALPING.MEAN_REV",
    setup_type="MEAN_REV",
    symbol="SYM_0/USDT",
    datetime=datetime.now(timezone.utc),
    signal_type=SignalType.LONG,
    strength=0.95,
    horizon="SCALPING",
    priority=1,
    tp_pct=0.015,
    sl_pct=0.02,
    current_price=1000.0,
    metadata={}
)

print("Running generate_order...")
try:
    order = risk_manager.generate_order(sig, 1000.0)
    print(f"Result: {order}")
except Exception as e:
    import traceback
    traceback.print_exc()
