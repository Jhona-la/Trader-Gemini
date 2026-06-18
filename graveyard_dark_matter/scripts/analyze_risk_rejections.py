import sys, os, io, contextlib
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timezone
from queue import Queue
from core.events import SignalEvent, SignalType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from config import Config

print("=" * 70)
print("🔍 RISK MANAGER ISOLATION FORENSIC AUDIT")
print("=" * 70)

# Simulate 13 USD Portfolio
events_queue = Queue()
portfolio = Portfolio(initial_capital=13.0)

# Mock data provider interface
class MockData:
    def __init__(self):
        self.is_backtest = True
    def get_latest_price(self, symbol):
        return 65000.0  # mock BTC price
    def get_active_positions(self):
        return {}

data_provider = MockData()

risk_manager = RiskManager(Config.Risk, events_queue)
# Override init so it doesn't fail
risk_manager.portfolio = portfolio

rejection_reasons = {}
trials = 500
success = 0

print(f"Injecting {trials} mock valid SCALPING signals into RiskManager...")

# Pre-heat portfolio with some history if needed
portfolio.update_market_price("BTC/USDT", 65000.0)

for i in range(trials):
    sig = SignalEvent(
        symbol="BTC/USDT",
        signal_type=SignalType.LONG,
        strength=0.8,
        strategy_id="[SCL] Technical Momentum_SCALPING",
        datetime=datetime.now(timezone.utc),
        horizon="SCALPING",
        tp_pct=0.012,
        sl_pct=0.006,
        metadata={"confidence": 0.8}
    )
    
    # Capture generate_order output to extract rejection reasons
    _f_capture = io.StringIO()
    order = None
    try:
        with contextlib.redirect_stdout(_f_capture):
            order = risk_manager.generate_order(sig, 65000.0)
    except Exception as e:
        reason = f"EXCEPTION:{type(e).__name__}"
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        continue
        
    if order is None:
        _captured = _f_capture.getvalue()
        _specific_reason = None
        for _line in _captured.strip().split('\n'):
            if '[RISK] Rejected by' in _line or '[RISK] Signal Rejected' in _line or '[RISK] Rejected' in _line:
                _specific_reason = _line.strip()
                break
            # Add general failure detection
            if 'violation' in _line.lower() or 'insufficient' in _line.lower() or 'blocked' in _line.lower() or 'floor' in _line.lower():
                 _specific_reason = _line.strip()
                 break
        
        if _specific_reason:
            rejection_reasons[_specific_reason] = rejection_reasons.get(_specific_reason, 0) + 1
        else:
            # Full dump if not standard format
            reason = f"UNKNOWN: {_captured[:100]}..."
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    else:
        success += 1
        # Need to simulate the fill so Portfolio respects concurrency/cooldown?
        # Actually without fill, Portfolio has 0 exposure, so all 500 should pass if sizing is fine!
        if getattr(order, 'quantity', 0) > 0:
            qty = order.quantity
            price = 65000.0
            margin = (qty * price) / getattr(Config, 'BINANCE_LEVERAGE', 10.0)
            portfolio.release_cash(margin)

print(f"\n--- DIAGNOSTIC RESULTS ---")
print(f"Total Trials: {trials}")
print(f"Orders Generated: {success}")
print(f"Orders Rejected: {trials - success}")
print("\n--- REJECTION FORENSIC BREAKDOWN ---")
for r, c in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"{c:4d}x | {r}")
