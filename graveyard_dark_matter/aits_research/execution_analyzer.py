"""
AITS Phase 6: Adaptive Execution & Smart Order Routing (Capa 5)
Execution Quality Analyzer

Post-trade audit engine that measures how well the Smart Order Router
performed. Computes three institutional metrics:

1. Slippage:  Difference between the intended price and the actual fill price.
2. Implementation Shortfall (IS):  Total cost of executing vs. a theoretical
   paper benchmark.  IS = (Paper PnL) − (Actual PnL).
3. Fill Rate:  Percentage of Limit/TWAP/Iceberg child orders that were
   actually filled (relevant because Limit orders can expire unfilled).

These metrics feed back into the RL Agent (Phase 5) as additional
penalty/reward signals, enabling it to learn which execution algorithms
produce the best fills under different market regimes.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class FillReport:
    """One fill event returned by the exchange after an order executes."""
    order_id: str
    symbol: str
    intended_price: float
    fill_price: float
    quantity: float
    fee_rate: float        # e.g. 0.0002 for maker, 0.0004 for taker
    filled: bool = True    # False if the order expired unfilled


@dataclass
class QualityReport:
    """Aggregated execution quality for a batch of fills."""
    total_orders: int = 0
    filled_orders: int = 0
    fill_rate_pct: float = 0.0

    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0

    implementation_shortfall_usd: float = 0.0
    total_fees_usd: float = 0.0

    grade: str = "UNGRADED"  # A / B / C / F


class ExecutionAnalyzer:
    """
    Audits a list of FillReport objects and produces a QualityReport.
    """

    # Grading thresholds
    GRADE_A_SLIP = 1.0   # ≤1 bps average slippage
    GRADE_B_SLIP = 3.0   # ≤3 bps
    GRADE_C_SLIP = 6.0   # ≤6 bps

    def analyze(self, fills: List[FillReport]) -> QualityReport:
        if not fills:
            return QualityReport()

        report = QualityReport(total_orders=len(fills))

        slippages_bps: List[float] = []
        total_shortfall = 0.0
        total_fees = 0.0

        for f in fills:
            if f.filled:
                report.filled_orders += 1

                # Slippage in basis points
                if f.intended_price != 0:
                    slip = abs(f.fill_price - f.intended_price) / f.intended_price * 10_000
                else:
                    slip = 0.0
                slippages_bps.append(slip)

                # Implementation Shortfall = |fill − intended| * quantity
                shortfall = abs(f.fill_price - f.intended_price) * f.quantity
                total_shortfall += shortfall

                # Fees
                total_fees += f.fill_price * f.quantity * f.fee_rate

        # Fill rate
        report.fill_rate_pct = (
            (report.filled_orders / report.total_orders * 100)
            if report.total_orders else 0.0
        )

        # Slippage statistics
        if slippages_bps:
            report.avg_slippage_bps = statistics.mean(slippages_bps)
            report.max_slippage_bps = max(slippages_bps)

        report.implementation_shortfall_usd = round(total_shortfall, 6)
        report.total_fees_usd = round(total_fees, 6)

        # Assign grade
        report.grade = self._grade(report)

        return report

    def _grade(self, r: QualityReport) -> str:
        if r.fill_rate_pct < 50:
            return "F"
        if r.avg_slippage_bps <= self.GRADE_A_SLIP:
            return "A"
        elif r.avg_slippage_bps <= self.GRADE_B_SLIP:
            return "B"
        elif r.avg_slippage_bps <= self.GRADE_C_SLIP:
            return "C"
        return "F"

    def print_report(self, report: QualityReport):
        logging.info("═══════════════════════════════════════════")
        logging.info("   AITS EXECUTION QUALITY REPORT")
        logging.info("═══════════════════════════════════════════")
        logging.info(f"  Orders Sent    : {report.total_orders}")
        logging.info(f"  Orders Filled  : {report.filled_orders}")
        logging.info(f"  Fill Rate      : {report.fill_rate_pct:.1f}%")
        logging.info(f"  Avg Slippage   : {report.avg_slippage_bps:.2f} bps")
        logging.info(f"  Max Slippage   : {report.max_slippage_bps:.2f} bps")
        logging.info(f"  Impl. Shortfall: ${report.implementation_shortfall_usd:.4f}")
        logging.info(f"  Total Fees     : ${report.total_fees_usd:.4f}")
        logging.info(f"  ────────────────────────────────────────")
        logging.info(f"  GRADE          : {report.grade}")
        logging.info("═══════════════════════════════════════════")


# ─── Demo / Self-Test ───────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = ExecutionAnalyzer()

    # Simulate fills from a TWAP execution
    fills = [
        FillReport("TWAP_0", "BTCUSDT", 67000.00, 67000.05, 0.0004, 0.0002),
        FillReport("TWAP_1", "BTCUSDT", 67000.00, 67000.12, 0.0004, 0.0002),
        FillReport("TWAP_2", "BTCUSDT", 67000.00, 67000.03, 0.0004, 0.0002),
        FillReport("TWAP_3", "BTCUSDT", 67000.00, 66999.98, 0.0004, 0.0002),
        FillReport("TWAP_4", "BTCUSDT", 67000.00, 67000.08, 0.0004, 0.0002, filled=False),
    ]

    report = analyzer.analyze(fills)
    analyzer.print_report(report)
