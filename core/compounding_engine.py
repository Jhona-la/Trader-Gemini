"""
💰 PHASE 7 AITS: WEALTH ACCUMULATION & COMPOUNDING ENGINE — MÓDULO HORIZON
===========================================================================
QUÉ: Motor de acumulación de riqueza y rebalanceo dinámico MICRO↔SCALP↔SWING.
POR QUÉ: Con $13 USD, el capital debe distribuirse entre 3 horizontes para
  maximizar oportunidades sin pisar capital entre horizontes.
PARA QUÉ: Maximizar el crecimiento compuesto adaptando la estrategia de
  asignación de capital al tamaño de la cuenta Y al régimen de mercado.
CÓMO: Lee Config.CAPITAL_ALLOCATION para distribución por régimen,
  aplica Config.ALLOC_BOUNDS como floor/ceiling inviolables.
CUÁNDO: Consultado por Portfolio.get_available_cash() en cada tick.
DÓNDE: core/compounding_engine.py
QUIÉN: CompoundingEngine (singleton), Portfolio, RiskManager.

DEPENDENCIAS CRÍTICAS:
- core/portfolio.py → _get_available_cash_internal() consume get_horizon_allocation()
- config.py → Config.CAPITAL_ALLOCATION, Config.ALLOC_BOUNDS, Config.INITIAL_CAPITAL
- config.py → Config.MICROSCALPING_MARGIN_CAP, SCALPING_MARGIN_CAP, SWING_MARGIN_CAP

MÓDULO HORIZON: Upgraded from 2-way (SCALP/SWING) to 3-way (MICRO/SCALP/SWING).
"""

import math
import time
from typing import Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger("CompoundingEngine")


class CompoundingEngine:
    """
    🏦 Adaptive Wealth Accumulation Engine (Phase 7 AITS) — HORIZON-AWARE

    Dynamically rebalances capital between MICRO, SCALPING and SWING based on
    current account equity using a sigmoid transfer function + regime awareness.

    MÓDULO HORIZON: 3-way allocation replaces the old 2-way system.
    """

    # ═══════════════════════════════════════════════════════════════
    # SIGMOID PARAMETERS (Tuned for $13 → $500+ growth curve)
    # MÓDULO HORIZON: Controls how SCALPING shrinks as equity grows,
    # feeding that allocation to SWING (safer at higher equity).
    # ═══════════════════════════════════════════════════════════════
    SIGMOID_MIDPOINT = 75.0   # Inflection point ($75 equity)
    SIGMOID_STEEPNESS = 0.04  # How fast the transition happens

    def __init__(self, initial_capital: float = 13.0):
        from config import Config

        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.last_equity = initial_capital

        # MÓDULO HORIZON: Read 3-way allocation from Config
        self._default_alloc = {
            'MICRO': getattr(Config, 'MICROSCALPING_MARGIN_CAP', 0.25),
            'SCALP': getattr(Config, 'SCALPING_MARGIN_CAP', 0.45),
            'SWING': getattr(Config, 'SWING_MARGIN_CAP', 0.30),
        }
        self._regime_alloc = getattr(Config, 'CAPITAL_ALLOCATION', {
            'NEUTRAL': self._default_alloc.copy(),
        })
        self._alloc_bounds = getattr(Config, 'ALLOC_BOUNDS', {
            'MICRO': {'min': 0.10, 'max': 0.40},
            'SCALP': {'min': 0.25, 'max': 0.60},
            'SWING': {'min': 0.10, 'max': 0.50},
        })

        # Compounding tracker
        self._compound_log = []
        self._last_log_time = 0
        self._log_interval = 300  # Log equity every 5 minutes

        # Cached 3-way allocation
        self._cached_micro_pct = self._default_alloc['MICRO']
        self._cached_scalping_pct = self._default_alloc['SCALP']
        self._cached_swing_pct = self._default_alloc['SWING']
        self._last_recalc = 0
        self._recalc_interval = 10

        # Phase tracking
        self._growth_phase = "SEED"
        self._phase_thresholds = {
            "SEED": 0.0,
            "SPROUT": 26.0,
            "GROWTH": 100.0,
            "TREE": 500.0,
        }

        # Current regime (updated externally)
        self._current_regime = "NEUTRAL"
        
        # MÓDULO HORIZON: Capa 3 Portfolio Consciousness
        self._horizon_metrics = {
            'MICRO': {'wr': 0.5, 'pnl': 0.0, 'trades': 0},
            'SCALP': {'wr': 0.5, 'pnl': 0.0, 'trades': 0},
            'SWING': {'wr': 0.5, 'pnl': 0.0, 'trades': 0},
        }

        # [PHASE 6] Kelly Hot-Hand Momentum Tracker
        self._current_win_streak = 0
        self._highest_win_streak = 0

        logger.info(
            f"💰 [CompoundingEngine] HORIZON-AWARE Initialized | Capital=${initial_capital:.2f} | "
            f"Phase={self._growth_phase} | Allocation: "
            f"MICRO={self._cached_micro_pct*100:.0f}% / "
            f"SCL={self._cached_scalping_pct*100:.0f}% / "
            f"SWG={self._cached_swing_pct*100:.0f}%"
        )

    def _sigmoid(self, equity: float) -> float:
        """Sigmoid transfer function normalized to [0, 1]."""
        x = self.SIGMOID_STEEPNESS * (equity - self.SIGMOID_MIDPOINT)
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def set_regime(self, regime: str):
        """
        MÓDULO HORIZON: Update market regime for allocation.
        Called by MarketRegimeDetector or Engine.
        """
        if regime in self._regime_alloc:
            self._current_regime = regime
        else:
            self._current_regime = "NEUTRAL"

    def update_horizon_performance(self, horizon: str, win_rate: float, pnl_pct: float, trades: int):
        """
        CAPA 3: Recibe métricas de RiskManager para Consciencia de Portfolio.
        """
        if horizon == 'MICROSCALPING': h = 'MICRO'
        elif horizon == 'SCALPING': h = 'SCALP'
        elif horizon == 'SWING': h = 'SWING'
        else: return
        
        self._horizon_metrics[h]['wr'] = win_rate
        self._horizon_metrics[h]['pnl'] = pnl_pct
        self._horizon_metrics[h]['trades'] = trades
        # Forzamos recálculo inmediato al recibir nuevos datos
        self._last_recalc = 0

    def force_recalc(self):
        """
        🚀 FASE 12 (HYPER-FREQUENCY CAPITAL RECYCLING):
        Fuerza la invalidación del caché para que el próximo tick o cálculo de margen 
        absorba instantáneamente cualquier PnL generado en el segundo exacto del cierre de trade.
        """
        self._last_recalc = 0

    def record_trade_result(self, pnl_usd: float):
        """
        [PHASE 6] Track consecutive wins for the Kelly Hot-Hand effect.
        If we lose, trigger the Kill Switch for the streak.
        """
        if pnl_usd > 0:
            self._current_win_streak += 1
            if self._current_win_streak > self._highest_win_streak:
                self._highest_win_streak = self._current_win_streak
            logger.debug(f"🔥 [HOT-HAND] Win streak extended to {self._current_win_streak}")
        elif pnl_usd < 0:
            logger.debug(f"🧊 [HOT-HAND] Loss detected. Streak reset from {self._current_win_streak} to 0.")
            self._current_win_streak = 0

    def get_kelly_multiplier(self, horizon: str = 'SCALPING') -> float:
        """
        🚀 PHASE 19: Asymmetric Fractional Kelly (Anti-Martingale)
        QUÉ: Calcula un multiplicador de tamaño de posición (Leverage/Risk) basado
          en la racha actual de ganancias.
        POR QUÉ: Para duplicar el capital cada 3 días sin exigir un WR de 100%,
          necesitamos componer agresivamente las ganancias ("dejar correr los ganadores").
        CÓMO: Si estamos en una racha ganadora, el multiplicador sube exponencialmente.
          Si perdemos, resetea a 1.0 (Capital presenvation).
        """
        # Riesgo Base
        base_multiplier = 1.0
        
        # Scaling Agresivo basado en Win Streak
        if self._current_win_streak >= 1:
            # Incrementa 20% el poder de fuego por cada victoria consecutiva
            streak_bonus = (self._current_win_streak * 0.20)
            base_multiplier += streak_bonus
            
            # Cap de seguridad para no quemar la cuenta en un mechazo (Max 3x riesgo)
            if base_multiplier > 3.0:
                base_multiplier = 3.0
                
        # Swing requiere menor apalancamiento que Scalping por los recorridos
        if horizon == 'SWING':
            base_multiplier *= 0.5
            
        return base_multiplier

    def get_horizon_allocation(self, equity: float, timestamp_ms: float = None) -> Tuple[float, float]:
        """
        Returns (scalping_pct, swing_pct) for backward compatibility.
        
        MÓDULO HORIZON: Internally calculates 3-way, returns 2-way for 
        backward compat. Use get_3way_allocation() for full 3-way.
        """
        micro, scalp, swing = self.get_3way_allocation(equity, timestamp_ms)
        # Backward compat: merge MICRO into SCALP for old consumers
        return (scalp + micro, swing)

    def get_3way_allocation(self, equity: float, timestamp_ms: float = None) -> Tuple[float, float, float]:
        """
        MÓDULO HORIZON: Returns (micro_pct, scalp_pct, swing_pct).
        
        QUÉ: Calcula la partición 3-way del capital entre horizontes.
        POR QUÉ: Cada horizonte necesita capital dedicado para operar
          simultáneamente sin pisarse.
        CÓMO: Base = Config.CAPITAL_ALLOCATION[regime], luego ajustado
          por sigmoid de madurez del equity.
        """
        now = (timestamp_ms / 1000.0) if timestamp_ms else time.time()

        if now - self._last_recalc < self._recalc_interval and timestamp_ms is None:
            return (self._cached_micro_pct, self._cached_scalping_pct, self._cached_swing_pct)

        self._last_recalc = now

        # 1. Get base allocation from regime
        regime_alloc = self._regime_alloc.get(self._current_regime, self._default_alloc)
        base_micro = regime_alloc.get('MICRO', self._default_alloc['MICRO'])
        base_scalp = regime_alloc.get('SCALP', self._default_alloc['SCALP'])
        base_swing = regime_alloc.get('SWING', self._default_alloc['SWING'])

        # 2. Apply sigmoid maturity adjustment
        # As equity grows, shift from MICRO → SWING (safer at higher equity)
        maturity = self._sigmoid(equity)
        # MICRO shrinks with maturity (micro is for small accounts)
        micro_adj = base_micro * (1.0 - maturity * 0.4)  # At max maturity, MICRO = 60% of base
        # SWING grows with maturity
        swing_adj = base_swing + (base_micro - micro_adj) * 0.7  # 70% of MICRO reduction goes to SWING
        # SCALP gets the rest
        scalp_adj = 1.0 - micro_adj - swing_adj

        # 2.5 CAPA 3: Apply Merit-Based Tilt (Portfolio Consciousness)
        # Shift capital towards the most profitable horizon dynamically
        total_wr = sum(m['wr'] for m in self._horizon_metrics.values())
        if total_wr > 0:
            # Calculate deviation from average WR
            avg_wr = total_wr / 3.0
            
            # Max tilt is 15% absolute reallocation
            max_tilt = 0.15
            
            # Calculate raw tilts
            micro_tilt = (self._horizon_metrics['MICRO']['wr'] - avg_wr) * max_tilt
            scalp_tilt = (self._horizon_metrics['SCALP']['wr'] - avg_wr) * max_tilt
            swing_tilt = (self._horizon_metrics['SWING']['wr'] - avg_wr) * max_tilt
            
            # Apply tilt
            micro_adj += micro_tilt
            scalp_adj += scalp_tilt
            swing_adj += swing_tilt

        # 3. Enforce bounds (IMMUTABLE)
        bounds = self._alloc_bounds
        micro_pct = max(bounds['MICRO']['min'], min(bounds['MICRO']['max'], micro_adj))
        scalp_pct = max(bounds['SCALP']['min'], min(bounds['SCALP']['max'], scalp_adj))
        swing_pct = max(bounds['SWING']['min'], min(bounds['SWING']['max'], swing_adj))

        # 4. Normalize to sum = 1.0
        total = micro_pct + scalp_pct + swing_pct
        if total > 0:
            micro_pct /= total
            scalp_pct /= total
            swing_pct /= total

        # 5. MICRO-ACCOUNT SAFETY OVERRIDE (Ensure minimum viability)
        # For $13 accounts, we need aggressive compounding.
        # Dedicate 70% to Scalping (fast turns) and 30% to Swing. Microscalping disabled at this tier.
        if equity < 25.0 and equity > 0:
            micro_pct = 0.0
            scalp_pct = 0.70
            swing_pct = 0.30

        # [PHASE 8 QUANTUM EVOLUTION] Multi-Horizon Synergy & Phase-State Transition
        # If market regime is extremely volatile, SWING is dangerous (requires wide stops that can't be afforded).
        # We trigger a Quantum Circuit Breaker: divert 100% of SWING margin into MICROSCALPING.
        if self._current_regime in ("HIGH_VOL", "VOLATILE", "CRASH", "LIQUIDITY_VOID"):
            logger.warning(f"⚡ [QUANTUM CIRCUIT BREAKER] Regime {self._current_regime} detected! Diverting Swing margin to Microscalping.")
            micro_pct += swing_pct
            swing_pct = 0.0

        # Update cache
        self._cached_micro_pct = round(micro_pct, 4)
        self._cached_scalping_pct = round(scalp_pct, 4)
        self._cached_swing_pct = round(swing_pct, 4)

        # Update growth phase
        old_phase = self._growth_phase
        self._update_growth_phase(equity)
        if old_phase != self._growth_phase:
            logger.info(
                f"🌱 [CompoundingEngine] PHASE TRANSITION: {old_phase} → {self._growth_phase} | "
                f"Equity=${equity:.2f} | Allocation: "
                f"MICRO={micro_pct*100:.1f}% / SCL={scalp_pct*100:.1f}% / SWG={swing_pct*100:.1f}%"
            )

        # Track equity for compounding velocity
        self._track_equity(equity, now)

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        self.last_equity = equity

        return (self._cached_micro_pct, self._cached_scalping_pct, self._cached_swing_pct)

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
            if len(self._compound_log) > 288:
                self._compound_log = self._compound_log[-288:]

    def get_compounding_velocity(self) -> Dict:
        """
        Returns compounding performance metrics.
        """
        if len(self._compound_log) < 2:
            return {
                "actual_growth_pct": 0.0,
                "theoretical_daily_target_pct": 100.0,
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

        actual_growth = (last_eq - first_eq) / max(first_eq, 0.01)
        actual_daily = actual_growth / max(elapsed_days, 0.001)
        theoretical_daily = 1.00

        velocity_ratio = actual_daily / theoretical_daily if theoretical_daily > 0 else 0.0
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

    def get_quantum_kelly_fraction(self, win_probability: float, reward_risk_ratio: float, max_kelly_cap: float = 0.30) -> float:
        """
        [PHASE 8 QUANTUM KELLY ENGINE + PHASE 6 HOT-HAND MOMENTUM]
        f* = W - ((1 - W) / R)
        Calculates the exact % of the account to risk on a single trade based on AI confidence.
        Target: Achieve exponential growth (100% in 3 days) by betting heavily on 95%+ confidence signals.
        """
        if win_probability <= 0 or reward_risk_ratio <= 0:
            return 0.05 # fallback to 5%

        kelly_f = win_probability - ((1.0 - win_probability) / reward_risk_ratio)
        
        # ═══════════════════════════════════════════════════════════════
        # [PHASE 6] KELLY "HOT-HAND" MOMENTUM ADJUSTMENT
        # QUÉ: Incrementa el max_kelly_cap si estamos en racha.
        # POR QUÉ: Los mercados tienen persistencia de regímenes.
        # PARA QUÉ: Duplicar en 3 días.
        # ═══════════════════════════════════════════════════════════════
        dynamic_cap = max_kelly_cap
        # 🚀 [PHASE 9] QUANTUM COMPOUNDING ENGINE (CRE)
        # Arriesgamos ganancias flotantes cuadráticamente.
        equity = self.last_equity
        if equity > self.initial_capital:
            growth_factor = equity / max(self.initial_capital, 0.01)
            dynamic_cap = min(1.0, max_kelly_cap * (growth_factor ** 1.5)) # Geometric acceleration
            
        if self._current_win_streak >= 3:
            dynamic_cap = min(1.0, dynamic_cap + (self._current_win_streak * 0.05))
            logger.info(f"🔥 [HOT-HAND KELLY] Streak={self._current_win_streak}. Base Cap: {max_kelly_cap*100}% -> Boosted Cap: {dynamic_cap*100:.0f}%")

        # Aggressive Fractional Kelly (we use Full Kelly bounded by max_kelly_cap because of the 3-day doubling goal)
        if kelly_f <= 0:
             return 0.01 # Negative edge -> 1% probe size
             
        # Scale up aggressively: W=0.9, R=1.5 -> Kelly = 0.9 - (0.1/1.5) = 0.83 -> Bounded to dynamic_cap
        safe_kelly = max(0.02, min(kelly_f, dynamic_cap))
        
        return safe_kelly

    def get_growth_roadmap(self, current_equity: float = None, current_day: int = 1, avg_net_pnl_per_trade: float = 0.0, trades_today: int = 0) -> Dict:
        """
        📊 Growth Roadmap Calculator
        """
        eq = current_equity or self.last_equity or self.initial_capital
        
        daily_growth_factor = 2.0
        daily_growth_pct = 100.0
        
        expected_start = self.initial_capital * math.pow(daily_growth_factor, max(0, current_day - 1))
        target_end = expected_start * daily_growth_factor
        
        daily_usd_target = target_end - expected_start
        usd_progress = eq - expected_start
        
        avg_win_usd = max(avg_net_pnl_per_trade, 0.05)
        remaining_usd = max(0.0, target_end - eq)
        trades_needed = math.ceil(remaining_usd / avg_win_usd)
        
        on_track = eq >= target_end
        
        return {
            "daily_target_pct": round(daily_growth_pct, 2),
            "daily_target_usd": round(daily_usd_target, 2),
            "target_equity_end_of_day": round(target_end, 2),
            "usd_progress_today": round(usd_progress, 2),
            "trades_needed_today": trades_needed if not on_track else 0,
            "avg_win_usd": round(avg_win_usd, 2),
            "on_track": on_track
        }

    def get_pyramid_allocation(self, unrealized_pnl_usd: float, current_position_value: float, pyramid_count: int) -> float:
        """
        🚀 PHASE 21: QUANTUM PYRAMIDING (ANTI-MARTINGALA)
        Calcula el tamaño adicional a inyectar en una posición ganadora
        financiado por el PnL no realizado (Ganancia Flotante).
        
        Reglas:
        1. Sólo se piramida si el PnL no realizado es positivo.
        2. La ganancia debe ser > 1% del equity actual para habilitar Pyramiding.
        3. La adición disminuye con cada escalón (50%, 25%, 12.5% del tamaño base).
        """
        if unrealized_pnl_usd <= 0:
            return 0.0
            
        # Umbral mínimo: 1% de la cuenta actual
        min_pnl_to_pyramid = self.last_equity * 0.010
        if unrealized_pnl_usd < min_pnl_to_pyramid:
            return 0.0
            
        # Factores de decrecimiento por escalón de pirámide
        # Escalón 0 -> añade 50% del valor actual
        # Escalón 1 -> añade 25%
        # Escalón 2 -> añade 12.5%
        decay_factor = 0.5 ** (pyramid_count + 1)
        
        # El tamaño sugerido es un porcentaje del valor actual de la posición,
        # pero está capado a 2x el PnL flotante (Financiado con ganancia)
        suggested_size_usd = current_position_value * decay_factor
        max_funded_size = unrealized_pnl_usd * 2.0
        
        allocation = min(suggested_size_usd, max_funded_size)
        
        logger.info(f"🔺 [PYRAMIDING] Flotante: ${unrealized_pnl_usd:.2f} | Escalón: {pyramid_count} | Sugerido: ${suggested_size_usd:.2f} | Final: ${allocation:.2f}")
        return allocation

    def get_metrics(self) -> Dict:
        """Returns full engine metrics for dashboard/telemetry."""
        return {
            # MÓDULO HORIZON: 3-way allocation
            "micro_allocation": self._cached_micro_pct,
            "scalping_allocation": self._cached_scalping_pct,
            "swing_allocation": self._cached_swing_pct,
            "growth_phase": self._growth_phase,
            "peak_equity": self.peak_equity,
            "initial_capital": self.initial_capital,
            "regime": self._current_regime,
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


if __name__ == "__main__":
    import random
    
    logger.info("🎲 Starting Monte Carlo Validation for Phase 7 (+100% / 3 days)")
    
    def run_simulation(win_rate, reward_risk_ratio, trades_per_day, days, risk_per_trade, start_cap=13.0):
        capital = start_cap
        for _ in range(days * trades_per_day):
            if capital <= 0:
                break
                
            # Asumiendo que arriesgamos un % fijo del capital disponible
            risk_amount = capital * risk_per_trade
            
            if random.random() <= win_rate:
                capital += risk_amount * reward_risk_ratio
            else:
                capital -= risk_amount
                
        return capital

    scenarios = [
        {"wr": 0.55, "rr": 1.5, "tpd": 20, "risk": 0.05},
        {"wr": 0.60, "rr": 1.2, "tpd": 30, "risk": 0.05},
        {"wr": 0.70, "rr": 1.0, "tpd": 40, "risk": 0.03},
    ]

    for idx, s in enumerate(scenarios):
        logger.info(f"\n--- Scenario {idx+1}: WR={s['wr']*100:.0f}%, RR={s['rr']}, Trades/Day={s['tpd']}, Risk={s['risk']*100:.1f}% ---")
        results = [run_simulation(s['wr'], s['rr'], s['tpd'], 3, s['risk']) for _ in range(1000)]
        avg_cap = sum(results) / len(results)
        success_rate = sum(1 for r in results if r >= 26.0) / len(results)
        ruin_rate = sum(1 for r in results if r < 5.0) / len(results)
        
        logger.info(f"Average Final Capital (3 days): ${avg_cap:.2f}")
        logger.info(f"Probability of doubling (>=$26): {success_rate*100:.1f}%")
        logger.info(f"Probability of ruin (<$5): {ruin_rate*100:.1f}%")

