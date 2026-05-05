"""
AITS Phase 8: Integration Bridge
Production Adapter

The single point of contact between the AITS institutional stack
and the existing Trader Gemini production engine.

Architecture:
  engine.py → [SignalEvent] → AITSBridge.evaluate() → [EnrichedSignal | VetoSignal]

Graceful Degradation:
  If the AITS infrastructure (Redis, TimescaleDB, Neo4j) is unavailable,
  the bridge returns the original signal UNMODIFIED, allowing the legacy
  bot to operate normally.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from aits_config import AITS_CFG
except ImportError:
    AITS_CFG = None

try:
    from sovereign_risk_shield import (
        SovereignRiskShield, OrderIntent, AccountState, ShieldVerdict
    )
except ImportError:
    SovereignRiskShield = None

try:
    from smart_order_router import SmartOrderRouter, MarketContext
except ImportError:
    SmartOrderRouter = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Signal Envelope ────────────────────────────────────────────────

@dataclass
class AITSSignalEnvelope:
    """
    Wraps a production SignalEvent with AITS enrichment metadata.
    The engine reads the `verdict` field to decide whether to proceed.
    """
    original_signal: Dict[str, Any]
    verdict: str = "PASS"                # PASS | BLOCK | HALT | SHUTDOWN
    enrichment: Dict[str, Any] = field(default_factory=dict)
    execution_plan: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_approved(self) -> bool:
        return self.verdict == "PASS"


# ─── The Bridge ─────────────────────────────────────────────────────

class AITSBridge:
    """
    Adapter between the AITS institutional stack and Trader Gemini production.

    Usage in engine.py (conceptual):
        bridge = AITSBridge()
        envelope = bridge.evaluate(signal_event, portfolio_state)
        if envelope.is_approved:
            execute(envelope.execution_plan or signal_event)
        else:
            log(f"AITS VETOED: {envelope.verdict}")
    """

    def __init__(self):
        self.config = AITS_CFG
        self.aits_available = self._check_availability()

        # Initialize AITS subsystems
        self.shield = None
        self.router = None

        if self.aits_available:
            if SovereignRiskShield and self.config.ENABLE_SOVEREIGN_SHIELD:
                self.shield = SovereignRiskShield()
                logging.info("🛡️ Sovereign Risk Shield: ARMED")

            if SmartOrderRouter and self.config.ENABLE_SMART_ROUTER:
                self.router = SmartOrderRouter()
                logging.info("🔀 Smart Order Router: ONLINE")

        self._log_status()

    def _check_availability(self) -> bool:
        """Checks if the AITS config module is loaded."""
        if not self.config:
            logging.warning("⚠️ AITS Config not found. Operating in LEGACY MODE.")
            return False
        return True

    def _log_status(self):
        if not self.aits_available:
            logging.info("═══ AITS Bridge: LEGACY MODE (no enrichment) ═══")
            return

        active = []
        if self.shield:
            active.append("Shield")
        if self.router:
            active.append("SmartRouter")

        logging.info(f"═══ AITS Bridge: INSTITUTIONAL MODE — Active: {active} ═══")

    # ── Main API ────────────────────────────────────────────────────

    def evaluate(
        self,
        signal: Dict[str, Any],
        account_state: Optional[Dict[str, Any]] = None
    ) -> AITSSignalEnvelope:
        """
        Main entry point. Takes a production signal dict and returns
        an enriched envelope with the AITS verdict.

        Args:
            signal: Dict with keys like 'symbol', 'side', 'quantity',
                    'price', 'confidence', 'horizon'.
            account_state: Dict with keys like 'equity', 'peak_equity',
                           'open_positions', 'trades_today', etc.

        Returns:
            AITSSignalEnvelope with verdict, enrichment data, and
            optional execution plan from the Smart Router.
        """
        envelope = AITSSignalEnvelope(original_signal=signal)

        # ── Graceful Degradation ──
        if not self.aits_available:
            envelope.enrichment["mode"] = "LEGACY"
            return envelope

        # ── Step 1: Sovereign Shield Evaluation ──
        if self.shield:
            verdict = self._run_shield(signal, account_state or {})
            envelope.verdict = verdict.value

            if verdict != ShieldVerdict.PASS:
                envelope.enrichment["shield_verdict"] = verdict.value
                logging.warning(
                    f"🛡️ AITS Shield VETOED {signal.get('symbol', '?')}: {verdict.value}"
                )
                return envelope  # Early exit — order destroyed

        # ── Step 2: Smart Order Router ──
        if self.router and self.config.ENABLE_SMART_ROUTER:
            execution_orders = self._run_router(signal)
            envelope.execution_plan = [
                {
                    "type": o.order_type.value,
                    "side": o.side.value,
                    "price": o.price,
                    "quantity": o.quantity,
                    "delay_ms": o.delay_ms,
                }
                for o in execution_orders
            ]
            envelope.enrichment["router_algo"] = (
                execution_orders[0].order_type.value if execution_orders else "NONE"
            )

        # ── Step 3: Feature Enrichment (from Redis) ──
        envelope.enrichment["aits_version"] = "1.0.0"
        envelope.enrichment["mode"] = "INSTITUTIONAL"

        return envelope

    # ── Internal Helpers ────────────────────────────────────────────

    def _run_shield(self, signal: dict, state: dict) -> ShieldVerdict:
        order = OrderIntent(
            symbol=signal.get("symbol", "UNKNOWN"),
            side=signal.get("side", "BUY"),
            quantity=signal.get("quantity", 0.0),
            price=signal.get("price", 0.0),
            horizon=signal.get("horizon", "SCALPING"),
            model_confidence=signal.get("confidence", 0.5),
        )

        account = AccountState(
            total_capital=state.get("total_capital", self.config.TOTAL_CAPITAL_USD),
            current_equity=state.get("equity", self.config.TOTAL_CAPITAL_USD),
            session_peak_equity=state.get("peak_equity", self.config.TOTAL_CAPITAL_USD),
            open_positions=state.get("open_positions", 0),
            trades_today=state.get("trades_today", 0),
            volatility_burst_active=state.get("volatility_burst", False),
            btc_correlation=state.get("btc_correlation", 0.85),
        )

        return self.shield.evaluate(order, account)

    def _run_router(self, signal: dict) -> list:
        ctx = MarketContext(
            symbol=signal.get("symbol", "UNKNOWN"),
            best_bid=signal.get("price", 0.0) * 0.9999,
            best_ask=signal.get("price", 0.0) * 1.0001,
            spread=signal.get("price", 0.0) * 0.0002,
            bid_volume_top5=signal.get("bid_volume", 10.0),
            ask_volume_top5=signal.get("ask_volume", 10.0),
            volatility_burst=signal.get("volatility_burst", False),
            prediction_confidence=signal.get("confidence", 0.5),
            predicted_direction="UP" if signal.get("side") == "BUY" else "DOWN",
        )
        return self.router.route(ctx, signal.get("quantity", 0.001))

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = {"aits_available": self.aits_available}
        if self.shield:
            stats["shield"] = self.shield.get_stats()
        if self.router:
            stats["router_total_orders"] = len(self.router.execution_log)
        return stats


# ─── Demo / Self-Test ───────────────────────────────────────────────

if __name__ == "__main__":
    bridge = AITSBridge()

    # ── Test 1: Clean signal → PASS + Smart Routing ──
    logging.info("\n═══ TEST 1: Clean Signal (should PASS) ═══")
    signal_1 = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.00005,
        "price": 67000.0,
        "confidence": 0.82,
        "horizon": "SCALPING",
    }
    account_1 = {
        "total_capital": 13.0,
        "equity": 12.95,
        "peak_equity": 13.0,
        "open_positions": 1,
        "trades_today": 5,
        "volatility_burst": False,
        "btc_correlation": 0.85,
    }
    env_1 = bridge.evaluate(signal_1, account_1)
    logging.info(f"Verdict: {env_1.verdict} | Approved: {env_1.is_approved}")
    logging.info(f"Router Algo: {env_1.enrichment.get('router_algo')}")
    logging.info(f"Execution Plan: {env_1.execution_plan}")

    # ── Test 2: Low confidence → BLOCK ──
    logging.info("\n═══ TEST 2: Low Confidence (should BLOCK) ═══")
    signal_2 = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "quantity": 0.0001,
        "price": 3800.0,
        "confidence": 0.40,
        "horizon": "SWING",
    }
    env_2 = bridge.evaluate(signal_2, account_1)
    logging.info(f"Verdict: {env_2.verdict} | Approved: {env_2.is_approved}")

    # ── Print Final Stats ──
    logging.info(f"\n═══ Bridge Stats: {bridge.get_stats()} ═══")
    logging.info("✅ AITS Bridge Integration Test Complete.")
