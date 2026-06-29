"""
AITS Shadow Bridge — Production Integration (Read-Only Mode)

Integrates into the Trader Gemini engine as a passive observer.
Evaluates every signal through the AITS Shield + Router but does NOT
block or modify execution. Instead, it logs what WOULD have happened,
allowing us to measure AITS accuracy in production without risking capital.

Usage in engine.py:
    from aits_research.shadow_bridge import ShadowBridge
    shadow = ShadowBridge()
    # After each signal:
    shadow.observe(signal_dict, account_state, actual_outcome)
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from aits_bridge import AITSBridge
except ImportError:
    AITSBridge = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SHADOW] %(message)s")

SHADOW_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_logs")


@dataclass
class ShadowObservation:
    timestamp: float
    symbol: str
    signal_side: str
    signal_confidence: float
    aits_verdict: str        # PASS / BLOCK / HALT / SHUTDOWN
    aits_router_algo: str    # LIMIT_MAKER / TWAP / ICEBERG / MARKET
    actual_outcome: Optional[str] = None  # WIN / LOSS / PENDING
    actual_pnl: float = 0.0


class ShadowBridge:
    """
    Passive wrapper around AITSBridge that logs decisions without acting.
    Tracks concordance rate: how often AITS agrees with actual outcomes.
    """

    def __init__(self):
        self.bridge = AITSBridge() if AITSBridge else None
        self.observations: List[ShadowObservation] = []
        self.session_start = time.time()
        os.makedirs(SHADOW_LOG_DIR, exist_ok=True)

        if self.bridge:
            logging.info("🔭 Shadow Bridge ACTIVE (read-only mode)")
        else:
            logging.warning("Shadow Bridge: AITSBridge not available")

    def observe(
        self,
        signal: Dict,
        account_state: Dict,
        actual_outcome: Optional[str] = None,
        actual_pnl: float = 0.0
    ) -> ShadowObservation:
        """
        Evaluate signal through AITS without blocking. Log the result.
        Call this after every signal generation in engine.py.
        """
        obs = ShadowObservation(
            timestamp=time.time(),
            symbol=signal["symbol"],
            signal_side=signal["side"],
            signal_confidence=signal["confidence"],
            aits_verdict="N/A",
            aits_router_algo="N/A",
            actual_outcome=actual_outcome,
            actual_pnl=actual_pnl,
        )

        if self.bridge:
            envelope = self.bridge.evaluate(signal, account_state)
            obs.aits_verdict = envelope.verdict
            obs.aits_router_algo = envelope.enrichment["router_algo"]

        self.observations.append(obs)
        return obs

    def update_outcome(self, symbol: str, outcome: str, pnl: float):
        """Updates the most recent observation for a symbol with its real outcome."""
        for obs in reversed(self.observations):
            if obs.symbol == symbol and obs.actual_outcome in (None, "PENDING"):
                obs.actual_outcome = outcome
                obs.actual_pnl = pnl
                break

    def get_concordance_report(self) -> Dict:
        """
        Measures how well AITS decisions align with actual outcomes.
        
        Concordance = cases where:
          - AITS said PASS and trade was a WIN, or
          - AITS said BLOCK and trade would have been a LOSS
        """
        resolved = [o for o in self.observations if o.actual_outcome in ("WIN", "LOSS")]
        if not resolved:
            return {"total": 0, "concordance_pct": 0.0}

        concordant = 0
        for o in resolved:
            if o.aits_verdict == "PASS" and o.actual_outcome == "WIN":
                concordant += 1
            elif o.aits_verdict in ("BLOCK", "HALT", "SHUTDOWN") and o.actual_outcome == "LOSS":
                concordant += 1

        pct = concordant / len(resolved) * 100
        return {
            "total_resolved": len(resolved),
            "concordant": concordant,
            "concordance_pct": round(pct, 1),
            "total_observed": len(self.observations),
            "pass_count": sum(1 for o in self.observations if o.aits_verdict == "PASS"),
            "block_count": sum(1 for o in self.observations if o.aits_verdict != "PASS"),
        }

    def save_session(self):
        """Persists shadow observations to disk for post-mortem analysis."""
        ts = int(self.session_start)
        path = os.path.join(SHADOW_LOG_DIR, f"shadow_session_{ts}.jsonl")
        with open(path, "w") as f:
            for obs in self.observations:
                f.write(json.dumps({
                    "ts": obs.timestamp,
                    "sym": obs.symbol,
                    "side": obs.signal_side,
                    "conf": obs.signal_confidence,
                    "verdict": obs.aits_verdict,
                    "algo": obs.aits_router_algo,
                    "outcome": obs.actual_outcome,
                    "pnl": obs.actual_pnl,
                }) + "\n")
        logging.info(f"💾 Shadow session saved: {path} ({len(self.observations)} observations)")
        return path


# ─── Self-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    shadow = ShadowBridge()
    acct = {"total_capital": 13, "equity": 12.95, "peak_equity": 13,
            "open_positions": 1, "trades_today": 3,
            "volatility_burst": False, "btc_correlation": 0.8}

    # Simulate 5 signals
    signals = [
        {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.00005, "price": 67000, "confidence": 0.80, "horizon": "SCALPING"},
        {"symbol": "ETHUSDT", "side": "SELL", "quantity": 0.001, "price": 3800, "confidence": 0.40, "horizon": "SWING"},
        {"symbol": "SOLUSDT", "side": "BUY", "quantity": 0.01, "price": 170, "confidence": 0.72, "horizon": "SCALPING"},
        {"symbol": "BTCUSDT", "side": "SELL", "quantity": 0.00003, "price": 67200, "confidence": 0.65, "horizon": "SCALPING"},
        {"symbol": "DOGEUSDT", "side": "BUY", "quantity": 50, "price": 0.15, "confidence": 0.58, "horizon": "SWING"},
    ]
    outcomes = ["WIN", "LOSS", "WIN", "LOSS", "WIN"]

    for sig, out in zip(signals, outcomes):
        obs = shadow.observe(sig, acct)
        shadow.update_outcome(sig["symbol"], out, 0.05 if out == "WIN" else -0.03)
        logging.info(f"  {sig['symbol']:12s} | AITS={obs.aits_verdict:8s} | Actual={out}")

    report = shadow.get_concordance_report()
    logging.info(f"\n{'═'*50}")
    logging.info(f"  SHADOW CONCORDANCE REPORT")
    logging.info(f"{'═'*50}")
    for k, v in report.items():
        logging.info(f"  {k:25s}: {v}")

    path = shadow.save_session()
    logging.info("✅ Shadow Bridge self-test complete.")
