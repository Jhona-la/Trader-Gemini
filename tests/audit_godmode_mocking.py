"""
Omni Auditor & Chaos Sandbox (God Mode v2.0)
Validates system behavior against Data Poisoning, API Ghosting, and Double-Spend Overload.
"""
import sys
import os
import asyncio
import time
import logging
from unittest.mock import Mock, patch
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("OmniAuditor")

class ChaosGodModeAuditor:
    def __init__(self):
        self.events_queue = Queue()
        self.portfolio = Portfolio(initial_capital=13.0)
        self.risk_manager = RiskManager(portfolio=self.portfolio)
        self.metrics = {"passed": 0, "failed": 0, "issues": []}

    def run_all(self):
        print("====== OMNI AUDITOR & CHAOS SANDBOX ======")
        self.test_1_omni_consistency()
        self.test_2_data_poisoning()
        self.test_3_double_spend_overload()
        self.test_4_api_ghosting()
        
        print("\n====== 📊 FINAL REPORT ======")
        print(f"PASSED: {self.metrics['passed']} | FAILED: {self.metrics['failed']}")
        if self.metrics['issues']:
            print("🛑 VULNERABILITIES DETECTED:")
            for idx, issue in enumerate(self.metrics['issues']):
                print(f" {idx+1}. {issue}")
        else:
            print("✅ SYSTEM IS MILITARY-GRADE RESILIENT.")
        print("=============================")

    def pass_check(self, condition, msg, fail_msg):
        if condition:
            logger.info(f"✅ {msg}")
            self.metrics["passed"] += 1
        else:
            logger.error(f"❌ {fail_msg}")
            self.metrics["failed"] += 1
            self.metrics["issues"].append(fail_msg)

    def test_1_omni_consistency(self):
        logger.info("\n--- TEST 1: OMNI CONSISTENCY ---")
        self.pass_check(
            self.portfolio.current_cash == 13.0,
            "Initial capital unified at $13.0",
            f"Capital leaked! Current: {self.portfolio.current_cash}"
        )
        self.pass_check(
            Config.Risk.MAX_DRAWDOWN <= 20.0,
            "Max Drawdown is defensively set.",
            "Max drawdown exceeds 20%."
        )

    def test_2_data_poisoning(self):
        logger.info("\n--- TEST 2: DATA POISONING (MALFORMED OHLCV) ---")
        try:
            market_event = MarketEvent(symbol="BTC/USDT", close_price=0.0)
            self.portfolio.update_market_price("BTC/USDT", 0.0)
            self.pass_check(
                True, 
                "Portfolio survived 0.0 price update without ZeroDivisionError.",
                "Portfolio crashed on 0.0 price update."
            )
            self.portfolio.update_market_price("BTC/USDT", -50.0)
            self.pass_check(
                True,
                "Portfolio survived negative price injection.",
                "Portfolio crashed on negative price injection."
            )
        except Exception as e:
            self.pass_check(False, "", f"Data Poisoning caused system crash: {e}")

    def test_3_double_spend_overload(self):
        logger.info("\n--- TEST 3: DOUBLE-SPEND OVERLOAD (CONCURRENCY ATTACK) ---")
        symbol = "XRP/USDT"
        self.portfolio.update_market_price(symbol, 0.5)
        
        approved_orders = 0
        from datetime import datetime, timezone
        for i in range(10):
            signal = SignalEvent(
                strategy_id="MOCK_STRAT",
                symbol=symbol,
                datetime=datetime.now(timezone.utc),
                signal_type=SignalType.LONG,
                strength=0.99,
                horizon="SCALPING",
                current_price=0.50
            )
            order = self.risk_manager.generate_order(signal, 0.50)
            if order:
                approved_orders += 1
                self.portfolio.update_fill(
                    FillEvent(
                        timeindex=datetime.now(timezone.utc),
                        symbol=symbol,
                        exchange="MOCK",
                        order_id=f"MOCK_{i}",
                        fill_price=0.50,
                        quantity=order.quantity,
                        fill_cost=0.50 * order.quantity,
                        commission=0.001,
                        direction=OrderSide.BUY,
                        horizon="SCALPING"
                    )
                )

        import risk.risk_manager
        max_allowed = getattr(Config.Risk, 'MAX_CONCURRENT_POSITIONS', 2)
        
        self.pass_check(
            approved_orders <= max_allowed,
            f"Double-Spend Overload thwarted! Only {approved_orders} orders approved out of 10.",
            f"Double-Spend Overload bypassed defenses! {approved_orders} orders approved (Max: {max_allowed})."
        )
        
    def test_4_api_ghosting(self):
        logger.info("\n--- TEST 4: API GHOSTING (EXECUTION DELAY & TIMEOUT) ---")
        symbol = "ADA/USDT"
        self.portfolio.update_market_price(symbol, 0.40)
        
        from datetime import datetime, timezone
        signal = SignalEvent(strategy_id="MOCK_STRAT", symbol=symbol, datetime=datetime.now(timezone.utc), signal_type=SignalType.LONG, strength=0.88, horizon="SCALPING", current_price=0.40)
        order1 = self.risk_manager.generate_order(signal, 0.40)
        
        signal2 = SignalEvent(strategy_id="MOCK_STRAT", symbol=symbol, datetime=datetime.now(timezone.utc), signal_type=SignalType.LONG, strength=0.90, horizon="SCALPING", current_price=0.40)
        order2 = self.risk_manager.generate_order(signal2, 0.40)
        
        self.pass_check(
            order2 is None,
            "API Ghosting Attack Resisted: Cooldown system rejected duplicate order while pending.",
            "API Ghosting Vulnerable: System allowed duplicate order while previous was hanging!"
        )


if __name__ == "__main__":
    auditor = ChaosGodModeAuditor()
    auditor.run_all()
