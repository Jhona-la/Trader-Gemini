import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.kill_switch import KillSwitch
from core.portfolio import Portfolio

p = Portfolio(initial_capital=13.0, auto_save=False)
ks = KillSwitch(p)
print(f"is_killed={ks.is_killed}")
print(f"has check={hasattr(ks, 'check_kill_conditions')}")
print(f"attrs: {[a for a in dir(ks) if not a.startswith('_')]}")
