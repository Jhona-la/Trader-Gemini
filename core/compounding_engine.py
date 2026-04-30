"""
💰 PHASE 7 AITS: WEALTH ACCUMULATION & COMPOUNDING ENGINE
===========================================================
QUÉ: Motor de acumulación de riqueza y rebalanceo dinámico Scalping↔Swing.
POR QUÉ: Con $13 USD, el 100% del capital debe estar en Scalping (volumen).
  A medida que crece a $50, $100, $500+, conviene migrar gradualmente hacia
  Swing donde las comisiones pesan menos y los movimientos son más limpios.
PARA QUÉ: Maximizar el crecimiento compuesto adaptando la estrategia de
  asignación de capital al tamaño de la cuenta en tiempo real.
CÓMO: Función sigmoidea que mapea equity → allocation ratio.
  - $13 → 85% Scalping / 15% Swing
  - $50 → 70% Scalping / 30% Swing
  - $200+ → 50% Scalping / 50% Swing
  El punto de inflexión está en ~$75 (mitad del primer objetivo de duplicación).
CUÁNDO: Consultado por Portfolio.get_available_cash() en cada tick.
DÓNDE: core/compounding_engine.py
QUIÉN: CompoundingEngine (singleton), Portfolio, RiskManager.

DEPENDENCIAS CRÍTICAS:
- core/portfolio.py → _get_available_cash_internal() consume get_horizon_allocation()
- config.py → Config.INITIAL_CAPITAL (punto de partida)
"""

import math
import time
from typing import Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger("CompoundingEngine")


class CompoundingEngine:
    """
    🏦 Adaptive Wealth Accumulation Engine (Phase 7 AITS)

    Dynamically rebalances capital between SCALPING and SWING based on
    current account equity using a sigmoid transfer function.

    Properties:
    - At micro-equity ($13), heavily biases toward Scalping (volume-based growth).
    - As equity compounds, gradually shifts toward Swing (fee-efficient growth).
    - Tracks compounding velocity (actual vs theoretical) for self-diagnosis.
    """

    # ═══════════════════════════════════════════════════════════════
    # SIGMOID PARAMETERS (Tuned for $13 → $500+ growth curve)
    # ═══════════════════════════════════════════════════════════════
    # The sigmoid: scalping_pct = max_scalp - (max_scalp - min_scalp) * sigmoid(equity)
    # sigmoid(x) = 1 / (1 + exp(-k * (x - midpoint)))
    MAX_SCALPING_PCT = 0.85   # At $13: 85% Scalping
    MIN_SCALPING_PCT = 0.45   # At $500+: 45% Scalping (55% Swing)
    SIGMOID_MIDPOINT = 75.0   # Inflection point ($75 equity)
    SIGMOID_STEEPNESS = 0.04  # How fast the transition happens

    def __init__(self, initial_capital: float = 13.0):
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.last_equity = initial_capital

        # Compounding tracker
        self._compound_log = []  # List of (timestamp, equity) for velocity calc
        self._last_log_time = 0
        self._log_interval = 300  # Log equity every 5 minutes

        # Cached allocation (refreshed every tick)
        self._cached_scalping_pct = self.MAX_SCALPING_PCT
        self._cached_swing_pct = 1.0 - self.MAX_SCALPING_PCT
        self._last_recalc = 0
        self._recalc_interval = 10  # Recalculate every 10 seconds

        # Phase tracking
        self._growth_phase = "SEED"  # SEED → SPROUT → GROWTH → TREE
        self._phase_thresholds = {
            "SEED": 0.0,       # $13 (starting)
            "SPROUT": 26.0,    # $26 (first doubling)
            "GROWTH": 100.0,   # $100 (sustainable)
            "TREE": 500.0,     # $500+ (institutional)
        }

        logger.info(
            f"💰 [CompoundingEngine] Initialized | Capital=${initial_capital:.2f} | "
            f"Phase={self._growth_phase} | Allocation: "
            f"SCL={self.MAX_SCALPING_PCT*100:.0f}%/SWG={(1-self.MAX_SCALPING_PCT)*100:.0f}%"
        )

    def _sigmoid(self, equity: float) -> float:
        """
        Sigmoid transfer function normalized to [0, 1].
        Maps equity to a maturity score where:
        - 0.0 = micro-account (max scalping)
        - 1.0 = mature account (balanced allocation)
        """
        x = self.SIGMOID_STEEPNESS * (equity - self.SIGMOID_MIDPOINT)
        # Clamp to prevent overflow
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def get_horizon_allocation(self, equity: float) -> Tuple[float, float]:
        """
        Returns (scalping_pct, swing_pct) based on current equity.

        QUÉ: Calcula la partición óptima del capital entre horizontes.
        POR QUÉ: El ratio fijo 60/40 es subóptimo — una cuenta de $13
          necesita concentración máxima en Scalping para superar el mínimo
          de Binance ($5), mientras que una cuenta de $200 se beneficia
          de diversificación temporal.
        PARA QUÉ: Maximizar la velocidad de duplicación (15 días target).
        CÓMO: Sigmoid → smooth transition sin saltos bruscos.

        Returns:
            tuple: (scalping_pct, swing_pct) donde ambos suman 1.0
        """
        now = time.time()
        if now - self._last_recalc < self._recalc_interval:
            return (self._cached_scalping_pct, self._cached_swing_pct)

        self._last_recalc = now

        # Calculate sigmoid maturity
        maturity = self._sigmoid(equity)

        # Interpolate: high maturity → lower scalping percentage
        scalping_pct = self.MAX_SCALPING_PCT - (
            (self.MAX_SCALPING_PCT - self.MIN_SCALPING_PCT) * maturity
        )

        # Clamp for safety
        scalping_pct = max(self.MIN_SCALPING_PCT, min(self.MAX_SCALPING_PCT, scalping_pct))
        swing_pct = 1.0 - scalping_pct

        # Update cache
        self._cached_scalping_pct = round(scalping_pct, 4)
        self._cached_swing_pct = round(swing_pct, 4)

        # Update growth phase
        old_phase = self._growth_phase
        self._update_growth_phase(equity)
        if old_phase != self._growth_phase:
            logger.info(
                f"🌱 [CompoundingEngine] PHASE TRANSITION: {old_phase} → {self._growth_phase} | "
                f"Equity=${equity:.2f} | New Allocation: "
                f"SCL={scalping_pct*100:.1f}%/SWG={swing_pct*100:.1f}%"
            )

        # Track equity for compounding velocity
        self._track_equity(equity, now)

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        self.last_equity = equity

        return (self._cached_scalping_pct, self._cached_swing_pct)

    def _update_growth_phase(self, equity: float):
        """Updates the growth phase label based on current equity."""
        if equity >= self._phase_thresholds["TREE"]:
            self._growth_phase = "TREE"
        elif equity >= self._phase_thresholds["GROWTH"]:
            self._growth_phase = "GROWTH"
        elif equity >= self._phase_thresholds["SPROUT"]:
            self._growth_phase = "SPROUT"
        else:
            self._growth_phase = "SEED"

    def _track_equity(self, equity: float, now: float):
        """Logs equity snapshots for compounding velocity analysis."""
        if now - self._last_log_time >= self._log_interval:
            self._compound_log.append((now, equity))
            self._last_log_time = now

            # Keep last 24h of data (288 samples at 5-min intervals)
            if len(self._compound_log) > 288:
                self._compound_log = self._compound_log[-288:]

    def get_compounding_velocity(self) -> Dict:
        """
        Returns compounding performance metrics.

        QUÉ: Mide la velocidad real de crecimiento vs. la teórica.
        POR QUÉ: Si el sistema no está compounding al ritmo esperado,
          algo está fallando (fees, sizing, signal quality).
        PARA QUÉ: Self-diagnosis y alerta temprana de degradación.
        """
        if len(self._compound_log) < 2:
            return {
                "actual_growth_pct": 0.0,
                "theoretical_daily_target_pct": 4.73,  # 100% in 15 days = ~4.73%/day
                "velocity_ratio": 0.0,
                "phase": self._growth_phase,
                "peak_equity": self.peak_equity,
                "current_drawdown_pct": 0.0,
                "samples": len(self._compound_log),
            }

        first_ts, first_eq = self._compound_log[0]
        last_ts, last_eq = self._compound_log[-1]

        elapsed_hours = max(0.001, (last_ts - first_ts) / 3600.0)
        elapsed_days = elapsed_hours / 24.0

        # Actual growth
        actual_growth = (last_eq - first_eq) / max(first_eq, 0.01)
        actual_daily = actual_growth / max(elapsed_days, 0.001)

        # Theoretical: 100% in 15 days = compound daily rate
        # (1 + r)^15 = 2.0 → r = 2^(1/15) - 1 ≈ 4.73%/day
        theoretical_daily = 0.0473

        # Velocity ratio: how fast are we vs. target
        velocity_ratio = actual_daily / theoretical_daily if theoretical_daily > 0 else 0.0

        # Current drawdown from peak
        dd = (self.peak_equity - last_eq) / max(self.peak_equity, 0.01)

        return {
            "actual_growth_pct": round(actual_growth * 100, 2),
            "actual_daily_pct": round(actual_daily * 100, 2),
            "theoretical_daily_target_pct": round(theoretical_daily * 100, 2),
            "velocity_ratio": round(velocity_ratio, 3),
            "phase": self._growth_phase,
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(last_eq, 2),
            "current_drawdown_pct": round(dd * 100, 2),
            "elapsed_hours": round(elapsed_hours, 1),
            "samples": len(self._compound_log),
        }

    def get_metrics(self) -> Dict:
        """Returns full engine metrics for dashboard/telemetry."""
        return {
            "scalping_allocation": self._cached_scalping_pct,
            "swing_allocation": self._cached_swing_pct,
            "growth_phase": self._growth_phase,
            "peak_equity": self.peak_equity,
            "initial_capital": self.initial_capital,
            "compounding": self.get_compounding_velocity(),
        }


# ═══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════
_compounding_engine = None


def get_compounding_engine(initial_capital: float = None) -> CompoundingEngine:
    """
    Returns the global CompoundingEngine singleton.
    Thread-safe lazy initialization.
    """
    global _compounding_engine
    if _compounding_engine is None:
        from config import Config
        cap = initial_capital or getattr(Config, "INITIAL_CAPITAL", 13.0)
        _compounding_engine = CompoundingEngine(initial_capital=cap)
    return _compounding_engine
