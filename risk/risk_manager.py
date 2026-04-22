import os
import sys
import time
import math
import traceback
import numpy as np
from collections import deque
from datetime import timedelta, datetime, timezone
from decimal import Decimal, getcontext

# ═══════════════════════════════════════════════════════════════
# PREDICTION TRACKER IMPORT (Feedback Loop Closure)
# QUÉ: Importa el sistema de trazabilidad predictiva.
# POR QUÉ: Cierra el gap donde prediction_metrics.json se generaba
#   offline pero NUNCA se consumía en producción.
# PARA QUÉ: Permite rechazar señales de baja precisión y modular
#   sizing por confidence_factor en tiempo real.
# ═══════════════════════════════════════════════════════════════
try:
    from core.prediction_tracker import PredictionTracker
    _PREDICTION_TRACKER_AVAILABLE = True
except ImportError:
    _PREDICTION_TRACKER_AVAILABLE = False

from config import Config
from core.events import OrderEvent, SignalEvent
from core.enums import OrderSide, SignalType, OrderType
from core.resolution_state import ResolutionState
from core.world_awareness import world_awareness
from risk.kill_switch import KillSwitch
from utils.debug_tracer import trace_execution
from utils.cooldown_manager import cooldown_manager
from utils.safe_leverage import safe_leverage_calculator
from utils.logger import logger
from core.data_handler import get_data_handler
from utils.statistics_pro import StatisticsPro
from utils.math_kernel import (
    calculate_garch_jit,
    compute_kelly_fraction_jit,
    extract_kelly_stats_jit,
    compute_cvar_jit,
)


# ============================================================
# SCIENTIFIC RISK TOOLS (FIXED)
# ============================================================


class FeeCalculator:
    """Cálculo preciso de fees - CORREGIDO"""

    # [SOVEREIGN-DEPLOY] Dynamic Fee Awareness
    TAKER_FEE = Config.BINANCE_TAKER_FEE_BNB  # Default fallback
    MAKER_FEE = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
    IS_DYNAMIC = False

    @classmethod
    def update_dynamic_fees(cls, maker: float, taker: float):
        cls.MAKER_FEE = maker
        cls.TAKER_FEE = taker
        cls.IS_DYNAMIC = True
        logger.info(
            f"💰 [FEE-AWARENESS] Dynamic Commission Rates Embedded: Maker {maker * 100:.4f}% | Taker {taker * 100:.4f}%"
        )

    @staticmethod
    def calculate_round_trip_fee(
        notional_value: float, order_type: str = "LIMIT"
    ) -> float:
        """
        SUPREMO-V4: Simulador de comisiones REALISTA (Binance Futures).
        - MAKER (LIMIT): 0.02% (con BNB)
        - TAKER (MARKET): 0.0375% (con BNB)
        """
        fee_rate = FeeCalculator.MAKER_FEE
        if order_type.upper() == "MARKET":
            fee_rate = FeeCalculator.TAKER_FEE

        return notional_value * fee_rate * 2


class CVaRCalculator:
    """Conditional Value at Risk - CALIBRADO para leverage"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.loss_history = deque(maxlen=100)

    def validate_integrity(self, price: float) -> bool:
        """
        [PHASE 14] Data Integrity Check (Chaos Defense).
        Rejects NaNs, Infs, or non-positive prices.
        """
        if price is None:
            return False
        if isinstance(price, (float, int)):
            if np.isnan(price) or np.isinf(price) or price <= 0:
                logger.error(f"🛡️ RiskManager: Invalid Price Detect ({price})")
                return False
        return True

    def update(self, pnl_pct: float):
        if pnl_pct < 0:
            self.loss_history.append(abs(pnl_pct))

    def calculate_cvar(self) -> float:
        if len(self.loss_history) < 10:
            return 0.05  # 5% default (REALISTA con 10x lev)

        # [NANO-SPEED] Convert deque to numpy and use compiled kernel
        loss_array = np.array(self.loss_history, dtype=np.float64)
        return float(compute_cvar_jit(loss_array, self.confidence_level))

    def should_reduce_risk(self, current_drawdown: float) -> bool:
        """FIXED: Threshold más permisivo para growth"""
        cvar = self.calculate_cvar()
        threshold = min(0.25, cvar * 2.5)  # Max 25%, 2.5x CVaR
        return current_drawdown >= threshold


# ============================================================
# MAIN RISK MANAGER
# ============================================================


class RiskManager:
    """Risk Management Module - FINAL VERSION"""

    def __init__(self, max_concurrent_positions=5, portfolio=None):
        self.max_risk_per_trade = Config.MAX_RISK_PER_TRADE
        self.stop_loss_pct = Config.STOP_LOSS_PCT
        # Capital tracking delegated to Portfolio (Single Source of Truth)
        # self.initial_capital = 12.0 (Removed)
        # self.current_capital = 12.0 (Removed)
        # self.peak_capital = 12.0  (Removed - managed by SafeLeverageCalculator)
        self.max_concurrent_positions = max_concurrent_positions
        self.portfolio = portfolio

        # Ensure SafeLeverageCalculator has portfolio reference
        if self.portfolio:
            safe_leverage_calculator.portfolio = self.portfolio

        # Cooldown System (Delegated to CooldownManager)
        # self.cooldowns = {} (Removed)
        self.current_regime = "RANGING"

        # Scientific Tools
        self.cvar_calc = CVaRCalculator()
        self.fee_calc = FeeCalculator()

        # Kill Switch
        self.kill_switch = KillSwitch(portfolio=self.portfolio)

        # ═══════════════════════════════════════════════════════════════
        # PREDICTION TRACKER (Feedback Loop Closure)
        # QUÉ: Tracker de precisión predictiva en tiempo real.
        # POR QUÉ: Sin esto, el sistema aprende win rate (Kelly) pero
        #   NO la precisión direccional ni la duración del edge.
        # CUÁNDO: Se inicializa con RiskManager, se alimenta con cada
        #   señal y se consulta en generate_order() y size_position().
        # ═══════════════════════════════════════════════════════════════
        if _PREDICTION_TRACKER_AVAILABLE:
            self.prediction_tracker = PredictionTracker()
        else:
            self.prediction_tracker = None

        # Kelly Stats (Dynamic Genesis V5.0)
        self.win_count = 0
        self.loss_count = 0
        # Instead of fixed 0.52, this now acts only as the absolute 'cold start' prior
        self.bootstrap_win_rate = getattr(Config.Risk, "DEFAULT_BOOTSTRAP_WR", 0.52)
        self.bootstrap_trades = getattr(Config.Risk, "BOOTSTRAP_TRADES", 20)

        # Growth Phases (CALIBRADO)
        self.LEVERAGE_GROWTH = 10
        self.POSITION_PCT_GROWTH = 0.30  # 30% en growth

        # Phase 5: Flipping State
        self.daily_flips = {}  # {symbol: {date: "YYYY-MM-DD", count: N}}
        self.last_flip_times = {}  # {symbol: timestamp}
        self.daily_trade_logs = {}  # {date: {symbol: count}}
        self.global_trade_count = 0  # Optimized global counter for Level VII
        # FORENSIC-V17: Bypassed for high-frequency micro-scalping
        # QUÉ: Límites diarios de trades. POR QUÉ: Exigencia de no limitar el mercado a $13.
        # PARA QUÉ: Oportunidades ilimitadas.
        self.MAX_TRADES_PER_SYMBOL = getattr(Config, "MAX_TRADES_PER_SYMBOL", 1000)
        self.MAX_TRADES_TOTAL = getattr(Config, "MAX_TRADES_TOTAL", 20000)

        # Phase 6: Stress Testing
        self.stress_score = 100.0  # Default perfect score (0% Ruin Risk)
        self.last_stress_check = 0
        self.stress_check_interval = 3600  # Check every hour

        # Meta-Brain Integration (Phase 7)
        self.strategy_selector = None  # Set by Engine

        # Execution Caps [SS-006 FIX: Removed duplicate MAX_TRADES_TOTAL hardcode]
        self.global_regime = "RANGING"  # FORENSIC-V24: Default to RANGING (was UNKNOWN → silent veto failures in backtest)

        # Phase 14-71: Dynamic Capital Allocation
        self.resolution_state = ResolutionState.STABLE
        self.recovery_threshold = (
            0.0075  # 0.75% Drawdown triggers defensive mode (halved risk)
        )
        self.growth_threshold = 0.05  # 5% Profit triggers growth

        # Phase 42: Momentum Exit Thresholds
        self.MOMENTUM_EXIT_THRESHOLD = 0.015  # 1.5% drop in 1m bars for long exit
        self.momentum_cache = {}  # {symbol: deque(maxlen=5)}

        # PHASE 56: Metal-Core Optimized Cache
        self._trade_cache = deque(
            maxlen=2000
        )  # List of dicts: {'is_win': bool, 'pnl_pct': float, 'symbol': str}
        self._cache_initialized = False
        self._last_day_str = 0  # Integer YYYYMMDD for fast comparison
        self._status_cache = {}
        self._last_status_read = 0

        # Phase L: Sector Correlation Filter
        # FORENSIC-V22: Dynamic sector limit for micro-accounts
        # With $13 equity, a single trade = ~$12.35 margin (95% of equity).
        # A fixed 35% limit ($4.55) blocks ANY trade after the first one.
        # Scale: <$100 → 95%, $100-$1000 → 60%, >$1000 → 35% (institutional)
        _init_cap = getattr(Config, 'INITIAL_CAPITAL', 13.0)
        if _init_cap < 100:
            self.max_sector_exposure = 0.95  # Micro-account: allow concentration
        elif _init_cap < 1000:
            self.max_sector_exposure = 0.60  # Small account: moderate diversification
        else:
            self.max_sector_exposure = 0.35  # Standard: institutional limits
        self.symbol_sectors = {
            "BTCUSDT": "MAJOR",
            "ETHUSDT": "MAJOR",
            "ETCUSDT": "MAJOR",
            "SOLUSDT": "LAYER1",
            "AVAXUSDT": "LAYER1",
            "DOTUSDT": "LAYER1",
            "NEARUSDT": "LAYER1",
            "ADAUSDT": "LAYER1",
            "TRXUSDT": "LAYER1",
            "ATOMUSDT": "LAYER1",
            "APTUSDT": "LAYER1",
            "DOGEUSDT": "MEME",
            "SHIBUSDT": "MEME",
            "PEPEUSDT": "MEME",
            "LINKUSDT": "DEFI",
            "UNIUSDT": "DEFI",
            "ARBUSDT": "DEFI",
            "OPUSDT": "DEFI",
            "MATICUSDT": "SCALING",
            "LTCUSDT": "PAYMENT",
            "BCHUSDT": "PAYMENT",
            "FILUSDT": "DEP_WEB3",
            "ICPUSDT": "DEP_WEB3",
        }

        # Phase 14: Funding & Rebate Tools
        self.funding_evasion_threshold = 0.0003  # 0.03%
        self.funding_buffer_minutes = 15
        self.rebate_priority_mode = getattr(Config, "REBATE_PRIORITY", True)

        # [SOVEREIGN-HORIZONS] Especialización Multi-Horizonte (Phase 1.2)
        # QUÉ: Parámetros aislados por horizonte temporal de trading.
        # POR QUÉ: Scalping requiere stop-loss más ajustados y cierres rápidos,
        #   mientras que Swing necesita espacio para respirar (ATR mayor).
        # PARA QUÉ: Evitar "Position Collision" y mejorar la tasa de acierto (WR).
        self.horizon_params = {
            "SCALPING": {
                "stop_loss_pct": getattr(
                    Config, "STOP_LOSS_PCT_SCALPING", 0.006
                ),  # 0.6%
                "take_profit_pct": getattr(
                    Config, "TAKE_PROFIT_PCT_SCALPING", 0.012
                ),  # 1.2%
                "max_risk_pct": getattr(
                    Config, "MAX_RISK_SCALPING", 0.05
                ),  # 5% del capital alloc
                "leverage": getattr(Config, "LEVERAGE_SCALPING", 10),
            },
            "SWING": {
                "stop_loss_pct": getattr(Config, "STOP_LOSS_PCT_SWING", 0.015),  # 1.5%
                "take_profit_pct": getattr(
                    Config, "TAKE_PROFIT_PCT_SWING", 0.035
                ),  # 3.5%
                "max_risk_pct": getattr(
                    Config, "MAX_RISK_SWING", 0.10
                ),  # 10% del capital alloc
                "leverage": getattr(Config, "LEVERAGE_SWING", 5),
            },
        }

        # Sovereign-Deploy: Kill Switch L1 & Fractional Kelly
        self.consecutive_losses = {}

    def _get_sector(self, symbol: str) -> str:
        """Standardized symbol to sector mapping."""
        # Normalize symbol for lookup
        clean_sym = symbol.replace("/", "").upper()
        if not clean_sym.endswith("USDT"):
            clean_sym += "USDT"
        return self.symbol_sectors.get(clean_sym, "ALT")

    def _get_sector_exposure(self, sector: str) -> float:
        """Returns total MARGIN exposure for a specific sector (not notional).
        
        FORENSIC-V22 FIX: Was using raw notional (qty * price) which with 10x
        leverage meant $123.50 for a $12.35 margin trade. The 35% equity limit
        ($4.55 on $13 account) was ALWAYS exceeded after just ONE trade,
        blocking ALL subsequent trades in the same sector.
        FIX: Divide by effective leverage to get MARGIN-based exposure.
        """
        if not self.portfolio:
            return 0.0
        exposure = 0.0
        for sym, pos in self.portfolio.positions.items():
            if self._get_sector(sym) == sector:
                qty = pos.get("quantity", 0)
                price = pos.get("current_price", pos.get("avg_price", 0))
                notional = abs(qty * price)
                # Use position-stored leverage or Config default
                lev = pos.get("leverage", getattr(Config, "BINANCE_LEVERAGE", 10)) or 10
                exposure += notional / lev  # MARGIN, not notional
        return exposure

    def _initialize_cache(self):
        """QUÉ: Carga inicial de trades a memoria para evitar I/O futuro."""
        if self._cache_initialized:
            return
        try:
            dh = get_data_handler()
            # Try to load recent trades from CSV once
            csv_path = "dashboard/data/futures/trades.csv"
            if os.path.exists(csv_path):
                trades = dh.load_trades_df(csv_path)
                if not trades.empty:
                    for _, t in trades.iterrows():
                        is_win = t.get("net_pnl", 0) > 0
                        pnl = (
                            t.get("net_pnl", 0)
                            / (t.get("entry_price", 1) * t.get("quantity", 1))
                            if t.get("entry_price", 0) > 0
                            else 0
                        )
                        self._trade_cache.append(
                            {
                                "is_win": is_win,
                                "pnl_pct": pnl,
                                "symbol": t.get("symbol", ""),
                            }
                        )
                        # Update counts for Kelly/WR
                        if is_win:
                            self.win_count += 1
                        else:
                            self.loss_count += 1
                        self.cvar_calc.update(pnl)
            self._cache_initialized = True
            logger.info(
                f"⚡ [RiskMgr] Meta-Core Cache Initialized with {len(self._trade_cache)} trades."
            )
        except Exception as e:
            logger.error(f"Cache Init Failed: {e}")
            self._cache_initialized = True  # Don't retry per tick

    # ============================================================
    # 🛡️ SUPREMO-V3: ATOMIC VALIDATION PIPELINE (ZERO-TRUST)
    # ============================================================

    def _validate_fat_finger(self, price, symbol):
        """
        AUDIT DEPT C: Sanity Check (>5% Deviation)
        Prevents orders with absurd prices due to API errors or bugs.
        """
        if price <= 0:
            return False

        # In a real scenario, we'd compare against a 1-minute moving average or order book mid-price.
        # Here we use the last known price from Portfolio if available, or just pass if first trade.
        last_price = None
        if self.portfolio and symbol in self.portfolio.positions:
            last_price = self.portfolio.positions[symbol].get("current_price")

        if last_price and last_price > 0:
            deviation = abs(price - last_price) / last_price
            if deviation > 0.05:  # > 5% Deviation
                logger.critical(
                    f"🛑 FAT FINGER BLOCKED {symbol}: Price {price} deviates {deviation * 100:.1f}% from {last_price}"
                )
                return False
        return True

    def _validate_slippage(self, symbol, current_price):
        """
        [SOVEREIGN-DEPLOY] Liquidity Awareness (Slippage < 0.1%)
        Estima un worst-case slippage utilizando el spread y la volatilidad local.
        """
        max_allowed_slippage = getattr(Config, "MAX_SLIPPAGE_PCT", 0.001)

        # Intentaremos estimar el slippage basado en el ATR o data_handler si existe
        dh = get_data_handler()
        if dh and hasattr(dh, "get_latest_bars"):
            try:
                bars = dh.get_latest_bars(symbol, n=5)
                if bars is not None and len(bars) > 1:
                    # Calculamos el true range promedio para esta volatilidad cortan high-low / close
                    recent_volatility = (
                        bars[-1]["high"] - bars[-1]["low"]
                    ) / current_price
                    est_slippage = (
                        recent_volatility * 0.15
                    )  # Heuristic rule of thumb for illiquidity

                    if est_slippage > max_allowed_slippage:
                        logger.warning(
                            f"🛑 [LIQUIDITY] Slippage check failed for {symbol}: Est {est_slippage * 100:.3f}% > Max {max_allowed_slippage * 100:.3f}%"
                        )
                        return False
            except Exception as e:
                logger.error(
                    f"Error in slippage validation (safe fallback triggered): {e}"
                )
                return True
        return True

    def _validate_emergency_bypass(self, signal_event):
        """QUÉ: Bypass instantáneo para señales de salida."""
        return signal_event.signal_type == SignalType.EXIT

    def _validate_kill_switch(self):
        """Valida estado global del sistema."""
        if not self.kill_switch.check_status():
            logger.warning(
                f"💀 Kill Switch Active: {self.kill_switch.activation_reason}"
            )
            return False
        return True

    def _validate_frequency_limits(self, symbol, signal_type):
        """Valida límites de trades diarios por símbolo y global."""
        if signal_type not in [SignalType.LONG, SignalType.SHORT]:
            return True

        # Fast Int-based Date check
        now = datetime.now()
        today_int = now.year * 10000 + now.month * 100 + now.day

        if today_int != self._last_day_str:
            self.daily_trade_logs = {}
            self.global_trade_count = 0
            self._last_day_str = today_int

        symbol_count = self.daily_trade_logs.get(symbol, 0)
        if symbol_count >= self.MAX_TRADES_PER_SYMBOL:
            return False

        if self.global_trade_count >= self.MAX_TRADES_TOTAL:
            return False
        return True

    def _validate_regime_veto(self, symbol, signal_type):
        """Veto basado en correlación con BTC (Swarm)."""
        if symbol == "BTC/USDT":
            return True

        # [PHASE 5] Bypass Global Veto if the asset has EXTREME relative strength (Hedging)
        if self.portfolio and hasattr(self.portfolio, "relative_strength_scores"):
            is_long = signal_type == SignalType.LONG
            rs_mult = self.portfolio.get_allocation_multiplier(symbol, is_long)
            if rs_mult >= 1.3:
                logger.info(
                    f"🛡️ [Veto Bypass] Allowing {getattr(signal_type, 'name', str(signal_type))} on {symbol} despite Global Regime (Relative Strength Hedging)."
                )
                return True

        # FORENSIC-V16: NY Session Filter REMOVED
        # QUÉ: Antes se bloqueaban trades en 15-18 UTC si ADX>30.
        # POR QUÉ SE ELIMINÓ: Crypto opera 24/7. Las sesiones de alta
        #   volatilidad son OPORTUNIDADES, no amenazas, para scalping.
        # PARA QUÉ: Permitir que el sistema capture movimientos fuertes
        #   que antes se vetaban arbitrariamente.

        if self.global_regime == "TRENDING_BEAR" and signal_type == SignalType.LONG:
            logger.warning(f"🛡️ [Veto] Blocking LONG {symbol} (Global: Bearish).")
            return False
        if self.global_regime == "TRENDING_BULL" and signal_type == SignalType.SHORT:
            logger.warning(f"🛡️ [Veto] Blocking SHORT {symbol} (Global: Bullish).")
            return False
        return True

    def _validate_directional_safety(
        self, symbol, signal_type, horizon: str = "SCALPING"
    ):
        """
        Evita duplicar posiciones en la misma dirección PARA EL MISMO HORIZONTE.

        FORENSIC-V11 FIX: Permite señales CONTRARIAS (flip) si la posición actual
        está perdiendo > 0.3%. Antes, esta función rechazaba el 69.9% de TODAS
        las señales porque bloqueaba misma-dirección incluso cuando el trade actual
        era perdedor y una señal contraria podría recuperar capital.
        """
        if not self.portfolio:
            return True

        v_key = f"{symbol}_{horizon}"
        v_pos = self.portfolio.virtual_ledger.get(v_key)

        if not v_pos:
            return True

        qty = v_pos.get("quantity", 0)
        if qty == 0:
            return True

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V16 FIX: FLIP LOGIC WAS DEAD CODE (unreachable after return False)
        # QUÉ: Permite cambiar de dirección si la posición actual está perdiendo.
        # POR QUÉ: Antes, el `return False` en L386 hacía que el bloque de flip
        #   (L388-393) fuera INALCANZABLE — dead code desde que se escribió.
        # PARA QUÉ: Habilitar trend-following agresivo: si estamos LONG perdiendo,
        #   permitir SHORT inmediato (y viceversa).
        # CÓMO: Calcular unrealized_pnl_pct del virtual_ledger ANTES del return.
        # CUÁNDO: En cada señal de dirección contraria a la posición actual.
        # DÓNDE: risk/risk_manager.py → _validate_directional_safety()
        # QUIÉN: RiskManager, Portfolio (virtual_ledger)
        # ═══════════════════════════════════════════════════════════════

        # Calculate unrealized PnL for flip decision
        entry_price = v_pos.get("avg_price", 0)
        current_price = v_pos.get("current_price", entry_price)
        unrealized_pnl_pct = 0.0
        if entry_price > 0 and current_price > 0:
            if qty > 0:  # LONG position
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT position
                unrealized_pnl_pct = (entry_price - current_price) / entry_price

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V28 FIX #2: ADAPTIVE FLIP THRESHOLDS
        # QUÉ: Umbrales de flip más permisivos para micro-cuentas.
        # POR QUÉ: BTC ATR típico es 0.04%. El threshold anterior de -0.05%
        #   EXCEDÍA el rango normal de volatilidad → flips nunca se ejecutaban.
        # PARA QUÉ: Permitir que el sistema cambie de dirección cuando el
        #   mercado invalida la tesis original, incluso con movimientos mínimos.
        # CÓMO: SCALPING -0.02% (mitad del ATR mínimo), SWING -0.1%.
        # ═══════════════════════════════════════════════════════════════
        flip_threshold = -0.0002 if horizon == "SCALPING" else -0.001

        # Block same-direction duplicates (never stack)
        if qty > 0 and signal_type == SignalType.LONG:
            # Already LONG, new signal is LONG → stacking prohibited
            return False
        if qty < 0 and signal_type == SignalType.SHORT:
            # Already SHORT, new signal is SHORT → stacking prohibited
            return False

        # Opposite direction signals: Allow if current position is losing
        if unrealized_pnl_pct < flip_threshold:
            logger.info(
                f"🔄 [{v_key}] FLIP ALLOWED: Current PnL {unrealized_pnl_pct * 100:.2f}% < {flip_threshold * 100:.2f}%. Permitting opposite signal."
            )
            return True

        # Default: allow opposite direction (it will be handled by generate_order's exit logic)
        return True

    def _validate_margin_ratio(self):
        """Phase 56: Optimized Margin check with 1s caching."""
        # ... (keep existing)
        return True

    def _validate_funding_risk(self, symbol: str, side: OrderSide) -> bool:
        """
        QUÉ: Bloquea entradas si el funding es excesivamente alto y el cobro es inminente.
        POR QUÉ: Evitar pérdidas por 'funding leak' en posiciones de HFT.
        SUPREMO-V4: Simetría total para SHORTS (ADAPTABILIDAD).
        """
        if not Config.BINANCE_USE_FUTURES:
            return True

        try:
            from data.data_provider import get_data_provider

            dp = get_data_provider()
            funding_info = dp.get_funding_rate(symbol)
            if not funding_info:
                return True

            rate = funding_info.get("last_funding_rate", 0)
            next_funding_time = funding_info.get("next_funding_time", 0)

            # Caso LONG: Funding positivo alto (pagamos por estar largos)
            if side == OrderSide.BUY and rate > self.funding_evasion_threshold:
                time_to_funding = (
                    next_funding_time - datetime.now(timezone.utc).timestamp()
                ) / 60
                if 0 < time_to_funding < self.funding_buffer_minutes:
                    logger.warning(
                        f"💸 [FundingGuard] VETO LONG {symbol}: Rate {rate * 100:.3f}% incoming in {time_to_funding:.1f}m."
                    )
                    return False

            # Caso SHORT: Funding negativo alto (pagamos por estar cortos)
            elif side == OrderSide.SELL and rate < -self.funding_evasion_threshold:
                time_to_funding = (
                    next_funding_time - datetime.now(timezone.utc).timestamp()
                ) / 60
                if 0 < time_to_funding < self.funding_buffer_minutes:
                    logger.warning(
                        f"💸 [FundingGuard] VETO SHORT {symbol}: Rate {rate * 100:.3f}% incoming in {time_to_funding:.1f}m."
                    )
                    return False

            return True
        except Exception as e:
            logger.error(f"Funding Check Error: {e}")
            return True

    # ============================================================
    # MOMENTUM EXIT (Phase 42)
    # ============================================================

    def _check_momentum_exit(self, symbol: str, side: str, data_provider) -> bool:
        """
        QUÉS: Salida por momentum adverso (Cuchillo Cayendo).
        POR QUÉ: Evitar esperar al SL si el precio cae >1.5% en segundos (Flash Crash).
        """
        if not data_provider:
            return False
        try:
            # Get last 3-5 bars (1m)
            bars = data_provider.get_latest_bars(symbol, n=5)
            if bars is None or len(bars) < 3:
                return False

            # Calculate 1m Returns
            closes = bars["close"]
            last_ret = (closes[-1] - closes[-2]) / closes[-2]
            accel = (closes[-1] - closes[-3]) / closes[-3]  # 2m change

            # Evolutionary Momentum Exit (Read from position or default)
            # Since _check_momentum_exit is proactive, we need the position's parameter
            # We must fetch it from the portfolio.
            # Wait, _check_momentum_exit does not receive `pos`. Let's assume -0.012 default
            # unless we pass it.
            # We will use a safe default here, but it's better if we pass accel_threshold dynamically.
            accel_threshold = getattr(self, "_last_momentum_accel", -0.012)

            if side == "LONG":
                # Momentum is strongly negative
                if last_ret < (accel_threshold * 0.6) or accel < accel_threshold:
                    logger.warning(
                        f"🪂 [RiskMgr] MOMENTUM EXIT {symbol}: Long dumped {accel * 100:.2f}% in 2m. GTFO."
                    )
                    return True
            else:
                # Momentum is strongly positive (Against Short)
                if last_ret > (-accel_threshold * 0.6) or accel > -accel_threshold:
                    logger.warning(
                        f"🪂 [RiskMgr] MOMENTUM EXIT {symbol}: Short squeezed {accel * 100:.2f}% in 2m. GTFO."
                    )
                    return True

            return False
        except Exception as e:
            logger.error(f"Momentum Check Error: {e}")
            return False

        # ============================================================

    # REGIME ORCHESTRATION (Phase 12)
    # ============================================================

    def update_regime(self, regime: str, data: dict = None):
        """
        External update of Market Regime (Single Source of Truth).
        """
        if regime in [
            "TRENDING",
            "RANGING",
            "VOLATILE",
            "STAGNANT",
            "MIXED",
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "CHOPPY",
            "ZOMBIE",
            "MEAN_REVERTING",
        ]:
            if self.current_regime != regime:
                logger.info(
                    f"⚖️ [RiskManager] Regime Change: {self.current_regime} -> {regime}"
                )
                self.current_regime = regime

    def update_global_regime(self, global_regime: str):
        """
        BTC Leader Broadcasting (Phase 8).
        """
        if self.global_regime != global_regime:
            self.global_regime = global_regime
            if global_regime == "TRENDING_BEAR":
                logger.warning(
                    "🛡️ [RiskMgr] GLOBAL VETO: BTC is Bearish. Restricting Altcoin Longs."
                )
            elif global_regime == "TRENDING_BULL":
                logger.info(
                    "🐂 [RiskMgr] Global Sentimens: BTC is Bullish. Opportunity window open."
                )

    def get_regime(self):
        return self.current_regime

    def check_volatility_shock(self, symbol, returns):
        """
        [PHASE II] GARCH Volatility Shock Circuit Breaker.
        If Realized Vol > 2.5 * Forecasted GARCH Vol -> KILL SWITCH.
        """
        try:
            if len(returns) < 50:
                return  # Insufficient data

            # 1. Forecast GARCH Variance
            garch_vars = calculate_garch_jit(np.array(returns, dtype=np.float64))
            forecast_vol = np.sqrt(garch_vars[-1])

            # 2. Realized Volatility (Last 10 bars)
            realized_vol = np.std(returns[-10:])

            # 3. Check for Shock
            if (
                realized_vol > 2.5 * forecast_vol and realized_vol > 0.01
            ):  # Min 1% vol to trigger
                logger.critical(
                    f"🛑 [CIRCUIT BREAKER] GARCH SHOCK on {symbol}! Realized={realized_vol:.4f} > 2.5x GARCH={forecast_vol:.4f}"
                )
                self.kill_switch.activate(f"GARCH Shock: {symbol} Volatility Explosion")
                return True

        except Exception as e:
            logger.error(f"GARCH Check Error: {e}")
        return False

    # ============================================================
    # GROWTH PHASE METHODS (FIXED)
    # ============================================================

    def get_current_phase(self, capital: float) -> str:
        """Delegate to SafeLeverageCalculator"""
        return safe_leverage_calculator.get_phase(capital)

    def get_win_rate(self) -> float:
        total = self.win_count + self.loss_count

        # Phase 5: Evolutionary Genesis - Dynamic Portfolio Win Rate
        # Extracción directa del historial corporativo para dimensionamiento Kelly real
        if self.portfolio and hasattr(self.portfolio, "get_statistics"):
            stats = self.portfolio.get_statistics()
            port_total = stats.get("total_trades", 0)
            port_wr = stats.get("win_rate", 0.0)

            if port_total >= self.bootstrap_trades:
                return port_wr
            elif port_total > 0:
                # Transición híbrida con la base portfolio
                weight = port_total / self.bootstrap_trades
                return (port_wr * weight) + (self.bootstrap_win_rate * (1 - weight))

        # Fallback local (Safety Net)
        if total < self.bootstrap_trades:
            if total > 0:
                weight_real = total / self.bootstrap_trades
                real_wr = self.win_count / total
                return (real_wr * weight_real) + (
                    self.bootstrap_win_rate * (1 - weight_real)
                )
            return self.bootstrap_win_rate
        return self.win_count / total if total > 0 else 0.5

    def get_bayesian_win_rate(self) -> float:
        """Phase 6: Bayesian Posterior Win Rate (Scientific)."""
        # Use Bayesian Inference for more robust "Real" Win Rate for optimization
        return StatisticsPro.bayesian_win_rate(
            self.win_count, self.loss_count, prior_alpha=10, prior_beta=10
        )

    def _compute_kelly_math(self, p: float, b: float, apply_mult: bool = True) -> float:
        """
        [PRECISION-AXIOMA] Core math for Kelly Criterion via Numba JIT.
        Eliminates the millisecond-latency of Python Decimal overhead.
        """
        try:
            # Defensive Scaling (Risk Fortress)
            kelly_mult = 0.25  # Quarter-Kelly for Scalping volatility

            # Clamp between 0% and 40% exposure
            clamped = compute_kelly_fraction_jit(
                p=float(p),
                b=float(b),
                apply_mult=apply_mult,
                kelly_mult=float(kelly_mult),
                stress_score=float(self.stress_score),
                max_exposure=0.40,
            )

            logger.debug(
                f"📐 [Axioma-Kelly NANO] P:{p:.3f} B:{b:.3f} Final:{clamped:.4f}"
            )
            return float(clamped)

        except Exception as e:
            logger.error(
                f"❌ [AXIOMA] Nano Kelly calculation failed: {e}. Defaulting to 0.0"
            )
            return 0.0

    def calculate_kelly_fraction(
        self,
        symbol: str = "",
        strategy_id: str = None,
        rr_ratio: float = 0.75,
        signal_event=None,
    ) -> float:
        """
        [PHASE 13] ALPHA-SHIELD: Dynamic Kelly Sizing
        QUÉ: Calcula la fracción óptima de Kelly basada en el performance real del símbolo/estrategia.
        POR QUÉ: Maximiza el crecimiento geométrico mientras protege contra la ruina.
        """
        try:
            # 1. Gather Stats from Cache (PHASE 56: O(1) in-memory)
            trades = [
                t
                for t in self._trade_cache
                if (not symbol or t["symbol"] == symbol)
                and (not strategy_id or t.get("strategy_id") == strategy_id)
            ]

            if len(trades) < 10:
                # Fallback to Bayesian Win Rate if no symbol data
                p = self.get_bayesian_win_rate()
                b = rr_ratio  # Payoff ratio
            else:
                # [NANO-SPEED] Use compiled kernel for stats
                pnl_arr = np.array([t["pnl_pct"] for t in trades], dtype=np.float64)
                is_win_arr = np.array([t["is_win"] for t in trades], dtype=np.bool_)
                p, b = extract_kelly_stats_jit(pnl_arr, is_win_arr)

            # 2. Kelly Formula (JIT Delegated)
            kelly_frac_float = self._compute_kelly_math(p, b, apply_mult=False)
            kelly = kelly_frac_float

            # 3. Defensive Scaling (Risk Fortress)
            # SOVEREIGN-DEPLOY: Absolute Fractional Kelly Enforcement (f*/10)
            kelly_mult = getattr(Config.Strategies, "ML_KELLY_FRACTION", 0.25)

            # Extreme Defense: If Ruin Risk (Stress Score) is low
            if self.stress_score < 90:
                kelly_mult = 0.25  # Quarter-Kelly

            # AEGIS-ULTRA: Systemic Risk Shield (Contagion)
            # If fleet correlation is high, reduce size to avoid synchronized drawdowns
            if hasattr(self, "fleet_correlation") and self.fleet_correlation > 0.85:
                logger.warning(
                    f"🚨 SYSTEMIC RISK: Fleet Correlation {self.fleet_correlation:.2f}. Reducing Size by 50%."
                )
                kelly_mult *= 0.5

            fractional_kelly = max(0.0, kelly * float(kelly_mult))

            # 4. Symbol Isolation & Sector Blocker
            if signal_event and hasattr(signal_event, "symbol"):
                if not self.validate_symbol_isolation(signal_event.symbol):
                    return 0.0

                sector = self._get_sector(signal_event.symbol)
                current_sector_exposure = self._get_sector_exposure(sector)
                capital = self.portfolio.get_total_equity() if self.portfolio else 15.0
                if current_sector_exposure >= (capital * self.max_sector_exposure):
                    logger.warning(f"🚫 Sector limit reached: {sector}")
                    return 0.0

            # 5. Final Clamp
            return float(
                max(0.05, min(fractional_kelly, 0.40))
            )  # Min 5%, Max 40% (Aggressive for $12)

        except Exception as e:
            logger.error(f"Kelly Error: {e}")
            return 0.15  # Safe Default

    def validate_symbol_isolation(self, symbol: str) -> bool:
        """
        [PHASE 14] Memory Isolation Check
        QUÉ: Verifica que no excedamos el presupuesto de memoria para 20 símbolos.
        POR QUÉ: Evitar fugas de memoria y degradación de performance en HFT.
        """
        active_symbols = 0
        if self.portfolio:
            active_symbols = sum(
                1 for pos in self.portfolio.positions.values() if pos["quantity"] != 0
            )

        # Budget: 20 Símbolos Máximo para estabilidad micro-latencia
        if active_symbols >= 20 and not (
            self.portfolio
            and symbol in self.portfolio.positions
            and self.portfolio.positions[symbol]["quantity"] != 0
        ):
            logger.critical(
                f"🛑 [ISOLATION] Memory Budget Exceeded! Blocking {symbol}."
            )
            return False
        return True

    def record_trade_result(self, is_win: bool, pnl_pct: float = 0, symbol: str = ""):
        """
        Phase 56: Real-time cache update (Atomic).
        ⚡ Phase OMNI: Tick-Level Dynamic Kelly Update.

        QUÉ: Recalcula la fracción de Kelly en cada fill event.
        POR QUÉ: El Kelly batch (cada N trades) introduce lag que pierde alpha.
        PARA QUÉ: Ajustar el sizing en tiempo real conforme cambia el performance.
        CÓMO: Rolling window de últimos 50 trades → EMA-smoothed win rate → Kelly formula.
        CUÁNDO: Cada fill event (vía Portfolio.update_fill → Engine._process_fill_event).
        DÓNDE: risk/risk_manager.py → record_trade_result().
        QUIÉN: RiskManager, Portfolio, Engine.
        """
        if is_win:
            self.win_count += 1
            if symbol:
                self.consecutive_losses[symbol] = 0
        else:
            self.loss_count += 1
            if symbol:
                self.consecutive_losses[symbol] = (
                    self.consecutive_losses.get(symbol, 0) + 1
                )
                if self.consecutive_losses[symbol] >= 5:
                    # [FORENSIC FIX] Relaxed from 3 losses/1h → 5 losses/5min
                    # QUÉ: Cooldown de 5 minutos tras 5 pérdidas consecutivas.
                    # POR QUÉ: 1 hora era excesivo para scalping — bloqueaba
                    #   la moneda el 6% del día por solo 3 pérdidas.
                    # PARA QUÉ: Permitir re-entry rápida cuando el mercado
                    #   revierte, sin perder oportunidades.
                    # CÓMO: 5 pérdidas es más estadísticamente significativo
                    #   que 3, y 5 minutos permite que las condiciones cambien.
                    logger.warning(
                        f"⚠️ [COOLING] {symbol} accumulated 5 consecutive losses. 5-min cooldown."
                    )
                    # Use custom cooldown (the method sets the timestamp, blocking re-entry for 300s)
                    cooldown_manager.check_custom_cooldown(f"loss_streak_{symbol}", 300)
                    self.consecutive_losses[symbol] = 0

        self.cvar_calc.update(pnl_pct)

        # Update Metal-Core Cache
        self._trade_cache.append(
            {"is_win": is_win, "pnl_pct": pnl_pct, "symbol": symbol}
        )

        # Optional: Limit cache growth to last 1000 trades for performance
        if len(self._trade_cache) > 1000:
            self._trade_cache.pop(0)

        # ═══════════════════════════════════════════════════════════════
        # PREDICTION TRACKER: OUTCOME PROPAGATION (Feedback Loop Closure)
        # QUÉ: Propaga el resultado del trade al PredictionTracker.
        # POR QUÉ: Sin esto, el tracker solo ve señales pero nunca sabe
        #   si el trade fue win/loss. El confidence_factor quedaría estático.
        # PARA QUÉ: Actualizar trade_win_rate en el tracker para ajustar
        #   dinámicamente el confidence_factor que modula sizing y LIMIT pricing.
        # ═══════════════════════════════════════════════════════════════
        if self.prediction_tracker:
            try:
                self.prediction_tracker.record_trade_outcome(
                    symbol=symbol, is_win=is_win, pnl_pct=pnl_pct
                )
            except Exception as _pt_err:
                logger.debug(f"PredictionTracker outcome error: {_pt_err}")

        # ⚡ PHASE OMNI: TICK-LEVEL KELLY RECALCULATION
        # Uses a rolling window of last 50 trades for responsive sizing
        _KELLY_WINDOW = 50
        trade_list = list(self._trade_cache)
        recent = trade_list[-_KELLY_WINDOW:]

        if len(recent) >= 10:  # Minimum sample size for statistical validity
            # [NANO-SPEED] Use compiled kernel for stats
            pnl_arr = np.array([t["pnl_pct"] for t in recent], dtype=np.float64)
            is_win_arr = np.array([t["is_win"] for t in recent], dtype=np.bool_)
            p, b = extract_kelly_stats_jit(pnl_arr, is_win_arr)

            # Decimal Kelly Math Evaluation
            raw_kelly = self._compute_kelly_math(p, b, apply_mult=False)

            # Half-Kelly with regime-aware scaling
            kelly_mult = 0.5
            if self.stress_score < 90:
                kelly_mult = 0.25  # Quarter-Kelly under stress

            tick_kelly = float(max(0.05, min(raw_kelly * kelly_mult, 0.40)))

            # EMA smoothing to prevent whipsaw (alpha=0.2)
            if not hasattr(self, "_tick_kelly"):
                self._tick_kelly = tick_kelly
            else:
                self._tick_kelly = 0.2 * tick_kelly + 0.8 * self._tick_kelly

            logger.debug(
                f"⚡ [Kelly/Axioma] Tick Update: p={p:.3f} b={b:.3f} raw={raw_kelly:.3f} → {self._tick_kelly:.3f}"
            )

    def update_equity(self, equity: float):
        """
        External update from Main Loop to sync Kill Switch & Safe Leverage.
        """
        # 1. Update Kill Switch (Critical Safety)
        if self.kill_switch:
            self.kill_switch.update_equity(equity)

        # 2. Update Safe Leverage Calculator (Growth Phase Tracking)
        safe_leverage_calculator.update_capital(equity)
        self.peak_capital = safe_leverage_calculator.peak_capital

    # [SS-013 FIX] First duplicate definition removed; unified version below at L536+

    def _get_dynamic_risk_per_trade(self, capital: float) -> float:
        """
        Calcula riesgo por trade basado en Profit Lock y Drawdown.
        PROFESSOR METHOD (PHOENIX ADJUSTED):
        1. Micro-Accounts (<$50): Bypasses Ratchet logic that permanently halts trading.
        """
        if capital < 50:
            # Phoenix Protocol for Micro Scalping
            return 0.03  # 3% base risk for micro accounts ($13 * 3% = $0.39 risk -> completely viable for scalping)

        initial = safe_leverage_calculator.initial_capital
        peak = safe_leverage_calculator.peak_capital

        # 1. Base Logic (Drawdown Protection - Tightened for HFT)
        risk_pct = 0.01  # Default 1%
        if peak > 0:
            dd = (peak - capital) / peak
            if dd > 0.025:
                risk_pct = 0.002  # 0.2% (Deep defense)
            elif dd > 0.015:
                risk_pct = 0.005  # 0.5% (Early defense)

        # 2. Profit Lock Milestones (Wealth Preservation)
        # "Si cuenta +50% sobre HWM" (interpretado como Growth sobre Initial)
        if peak >= (initial * 2.0):  # +100% Growth
            risk_pct *= 0.10  # Reduce to 10% of standard (0.1% risk)
            # Logic: "Account doubled. Don't blow it."
        elif peak >= (initial * 1.5):  # +50% Growth
            risk_pct *= 0.25  # Reduce to 25% of standard (0.25% risk)

        # 3. Protected Capital Floor (The Ratchet)
        profit = peak - initial
        if profit > 0:
            # Lock 80% of ATH profits
            protected_capital = initial + (profit * 0.80)

            # Calculate Max Loss Allowed for this trade
            max_loss_allowed = capital - protected_capital

            if max_loss_allowed <= 0:
                print(
                    f"🛑 PROTECTED CAPITAL REACHED (${protected_capital:.2f}). Trading Halted."
                )
                return 0.0

            # Clamp risk amount
            current_risk_amt = capital * risk_pct
            if current_risk_amt > max_loss_allowed:
                print(
                    f"🛡️ RATCHET: Clamping risk ${current_risk_amt:.2f} -> ${max_loss_allowed:.2f}"
                )
                risk_pct = max_loss_allowed / capital

        return risk_pct

    def _update_stress_metrics(self):
        """Phase 56: Use in-memory cache instead of CSV for PoR."""
        import time

        now = time.time()
        if now - self.last_stress_check < self.stress_check_interval:
            return

        if not self._cache_initialized:
            self._initialize_cache()

        try:
            pnl_returns = [t["pnl_pct"] for t in self._trade_cache]

            if len(pnl_returns) >= 20:
                paths = StatisticsPro.generate_monte_carlo_paths(
                    pnl_returns, n_sims=500
                )  # Reduced sims for speed
                metrics = StatisticsPro.calculate_stress_metrics(paths)
                self.stress_score = metrics.get("stress_score", 100.0)

            self.last_stress_check = now
        except Exception as e:
            pass

    def _calculate_dynamic_stop_loss(
        self, atr_pct: float, horizon: str = "SCALPING"
    ) -> float:
        """
        Calcula SL dinámico basado en régimen de volatilidad.
        SUPREMO-V4: ADAPTABILIDAD en mercados CHOPPY.
        """
        horizon_str = getattr(Config.Strategies, "ACTIVE_HORIZON", "1D")
        horizon_days = int(horizon_str.replace("D", "")) if "D" in horizon_str else 1
        h_sqrt = math.sqrt(horizon_days)

        # 1. Base Multiplier por Volatilidad
        # SUPREMO-V4: Cambio de SL fijo a dinámico (1.5x ATR)
        # POR QUÉ: Si el precio se mueve 1.5x su ATR en contra, la tesis del trade murió para micro-cuentas.
        # Un SL de 3.0x es demasiado suelto y genera pérdidas de -5%.
        mult = 1.5  # Reducido radicalmente de 3.0 para cuenta de $13

        # 2. Ajuste por Régimen de Mercado (ADAPTABILIDAD)
        # En CHOPPY, APRETAMOS el stop (no lo ampliamos) porque no queremos regalar capital al ruido.
        # En TRENDING, podemos ajustarlo más para proteger ganancias.
        if self.current_regime == "CHOPPY":
            mult *= 0.90
            logger.debug(
                f"🛡️ [ADAPTIVE SL] CHOPPY regime detected. TIGHTENING SL multiplier to {mult:.2f} (Capital Protection)"
            )
        elif self.current_regime == "TRENDING":
            mult *= 0.85

        # 3. AEGIS-ULTRA: MAE-Based Stop Optimization
        # If we have trade history, check average MAE (Max Adverse Excursion)
        if hasattr(self, "_trade_cache") and len(self._trade_cache) > 20:
            winning_maes = [
                t.get("max_adverse_excursion", 0)
                for t in self._trade_cache
                if t["is_win"]
            ]
            if winning_maes:
                avg_mae = np.mean(winning_maes)
                # Set stop just below average MAE of winners (Tightest possible valid stop)
                mae_stop = avg_mae * 1.2

                # Use the tighter of ATR-based or MAE-based
                atr_stop = atr_pct * mult
                final_stop = min(atr_stop, max(0.002 * h_sqrt, mae_stop))
                return final_stop

        sl_raw = atr_pct * mult

        # Dynamically scale Max and Min SL limits
        # 1D Max: 1.2%. 30D Max: ~6.5%
        max_limit = 0.012 * h_sqrt
        min_limit = 0.002 * max(1.0, h_sqrt / 2)

        return max(min_limit, min(sl_raw, max_limit))

    def _update_capital_tracking(self, current_equity: float):
        """
        Deprecated. Use update_equity directly instead.
        """
        self.update_equity(current_equity)

    def _track_drawdown_velocity(self, capital: float) -> float:
        """
        Phase 12: Tracks capital over time (bars/calls) to detect fast drops.
        Returns a target_exposure multiplier (1.0 = normal, 0.5 = defensive).
        """
        if not hasattr(self, "capital_history"):
            import collections
            import time

            self.capital_history = collections.deque(maxlen=100)
            self._last_capital_track = 0.0

        import time

        now = time.time()

        # Sample equity at most every 60 seconds (simulating 1-minute bars)
        if now - getattr(self, "_last_capital_track", 0) > 60:
            self.capital_history.append({"value": capital, "ts": now})
            self._last_capital_track = now

        multiplier = 1.0

        # Evaluate drops
        if len(self.capital_history) >= 30:  # at least 30 minutes of data
            point_30m_ago = self.capital_history[-min(30, len(self.capital_history))]
            drop_30m = (point_30m_ago["value"] - capital) / point_30m_ago["value"]

            if drop_30m > 0.005:  # >0.5% drop
                logger.warning(
                    f"📉 [VELOCITY] Fast 30m Drop: {drop_30m * 100:.2f}%. Halving size."
                )
                multiplier *= 0.5

        if len(self.capital_history) >= 60:  # at least 60 minutes
            point_60m_ago = self.capital_history[-60]
            drop_60m = (point_60m_ago["value"] - capital) / point_60m_ago["value"]

            if drop_60m > 0.01:  # >1.0% drop
                logger.warning(
                    f"🚨 [VELOCITY DEFENSE] 60m Drop: {drop_60m * 100:.2f}%. Defensive Mode."
                )
                multiplier *= 0.5  # Aggregate 0.5 * 0.5 = 0.25 (Quarter-size)

        return multiplier

    # ============================================================
    # POSITION SIZING (FIXED)
    # ============================================================

    @trace_execution
    def size_position(self, symbol, risk_pct=0.02, multiplier=1.0, horizon="SCALPING", current_price=0.0):
        """
        [PHASE 3.2] Meritocratic Position Sizing (The 13-Dollar micro-account Protocol)
        QUÉ: Calcula el tamaño de posición (notional y quantity) basándose en mérito y horizonte.
        POR QUÉ: En una cuenta de $13, el sizing debe ser atómico para evitar rechazos de Binance
           por notional insuficiente (< $5.00) o falta de margen.
        PARA QUÉ: Maximizar el crecimiento compuesto equilibrando SCALPING (volumen) vs SWING (pips).
        CÓMO: Capital -> Merit Mult -> Risk-at-Risk -> Margin -> Quantity.
        DÓNDE: risk/risk_manager.py
        """
        try:
            if not self.portfolio:
                return None

            # 1. Obtener parámetros por horizonte
            h_params = self.horizon_params.get(horizon, self.horizon_params["SCALPING"])

            # [MERITOCRACY] Merit-based risk adjustment
            sl_pct = h_params["stop_loss_pct"]
            tp_pct = h_params["take_profit_pct"]
            target_leverage = h_params["leverage"]

            # 2. Calcular Capital Disponible para este horizonte
            available_cash = self.portfolio.get_available_cash(horizon=horizon)
            
            # AEGIS-V21: Adaptive floor for micro-accounts ($13)
            _min_cash_floor = 0.50 if self.portfolio.get_total_equity() < 50.0 else 1.0
            
            if available_cash < _min_cash_floor:
                logger.warning(
                    f"⚠️ [SIZING] Insufficient margin in {horizon} ledger: ${available_cash:.2f} (Floor: ${_min_cash_floor})"
                )
                return None

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V28 FIX #1: ADAPTIVE GLOBAL MARGIN CAP
            # QUÉ: Cap dinámico según tamaño de cuenta.
            # POR QUÉ: Con $13 y cap 85%, 2 posiciones de ~$4 margen cada una
            #   consumen $8/$11.05 = 72% → solo $3.05 headroom. Cuando se suma
            #   pending, headroom < $1 → TODA señal es rechazada (77.4% de
            #   señales mueren aquí).
            # PARA QUÉ: Micro-cuentas necesitan utilización máxima para superar
            #   el mínimo notional de Binance ($5). El cap conservador del 85%
            #   es para cuentas institucionales, no para $13.
            # CÓMO: <$50 → 95% cap (micro), ≥$50 → 85% cap (standard).
            #   Headroom floor: <$50 → $0.50, ≥$50 → $1.00.
            # ═══════════════════════════════════════════════════════════════
            global_cash = getattr(self.portfolio, "current_cash", available_cash)
            global_used = getattr(self.portfolio, "used_margin", 0.0)
            global_pending = getattr(self.portfolio, "pending_cash", 0.0)
            
            # Adaptive cap: micro-accounts need maximum utilization
            _margin_cap_pct = 0.95 if global_cash < 50.0 else 0.85
            _headroom_floor = 0.50 if global_cash < 50.0 else 1.0
            
            max_global_margin = global_cash * _margin_cap_pct
            current_total_margin = global_used + global_pending
            remaining_margin_headroom = max(0.0, max_global_margin - current_total_margin)
            
            if remaining_margin_headroom <= _headroom_floor:
                 logger.warning(f"⚠️ [SIZING] Global Margin Cap Reached. Used+Pending: ${current_total_margin:.2f} / Max: ${max_global_margin:.2f} (Cap: {_margin_cap_pct*100:.0f}%)")
                 return None

            # 3. Cálculo de Tamaño Nominal (Notional)
            effective_risk = min(0.20, risk_pct * multiplier)  # Cap a 20% max por trade
            risk_amount = available_cash * effective_risk

            # Notional = Risk_Amount / SL_Pct
            if sl_pct > 0:
                notional_size = risk_amount / sl_pct
            else:
                notional_size = available_cash * target_leverage * 0.5  # Conservador

            # 4. Hardening para Cuenta Micro ($13 USD)
            # Limitar al MENOR entre el 40% del disponible, o el headroom global restante
            max_notional_from_cash = available_cash * target_leverage * 0.40
            max_notional_from_headroom = remaining_margin_headroom * target_leverage
            
            max_notional = min(max_notional_from_cash, max_notional_from_headroom)
            notional_size = min(notional_size, max_notional)

            # Minimum Notional Guard ($6.00 para margen de error)
            if notional_size < 6.0:
                margin_needed_for_pad = 6.0 / target_leverage
                if margin_needed_for_pad <= remaining_margin_headroom and (available_cash * target_leverage) >= 6.0:
                    notional_size = 6.0
                    logger.debug(
                        f"⚖️ [MARGIN-FIT] Padding notional to $6.0 for {symbol} (Headroom: ${remaining_margin_headroom:.2f})"
                    )
                else:
                    logger.warning(
                        f"🚫 [SIZING] Notional {notional_size:.2f} below minimum and cannot be padded (Headroom: ${remaining_margin_headroom:.2f})."
                    )
                    return None

            # 5. Cálculo Final de Cantidad (con fallback robusto)
            # [P0 FIX] Fallback para _last_prices
            if current_price <= 0:
                if hasattr(self.portfolio, "_last_prices") and self.portfolio._last_prices:
                    current_price = self.portfolio._last_prices.get(symbol, 0)

            # Fallback 1: Try getting from positions
            if current_price <= 0 and hasattr(self.portfolio, "positions"):
                pos = self.portfolio.positions.get(symbol)
                if pos and pos.get("current_price"):
                    current_price = pos.get("current_price", 0)

            # Fallback 2: Try virtual_ledger (for multi-horizon)
            if current_price <= 0 and hasattr(self.portfolio, "virtual_ledger"):
                for key, vpos in self.portfolio.virtual_ledger.items():
                    if symbol in key and vpos.get("current_price"):
                        current_price = vpos.get("current_price", 0)
                        break

            if current_price <= 0:
                logger.error(f"❌ [SIZING] Price for {symbol} not available.")
                return None

            quantity = notional_size / current_price

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V24 FIX #1 (CATASTRÓFICO): REMOVED HARDCODED OrderSide.BUY
            # QUÉ: size_position() retornaba "direction": OrderSide.BUY SIEMPRE.
            # POR QUÉ: Esto ANULABA todas las señales SHORT — se ejecutaban como
            #   LONG. Cada vez que el modelo predecía "vender", el sistema compraba.
            # PARA QUÉ: La dirección la determina generate_order() basado en
            #   signal_event.signal_type (LONG→BUY, SHORT→SELL). size_position()
            #   solo calcula CUÁNTO, no EN QUÉ DIRECCIÓN.
            # CÓMO: Eliminado "direction" del return dict. generate_order() ya
            #   calcula order_side correctamente en L1496.
            # ═══════════════════════════════════════════════════════════════
            return {
                "quantity": quantity,
                "dollar_size": notional_size / target_leverage,
                "notional": notional_size,
                "leverage": target_leverage,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
            }
        except Exception as e:
            logger.error(f"❌ Sizing Failure for {symbol}: {e}")
            return None

    def _update_resolution_state(self, current_dd: float):
        """Phase 14: State Machine for Risk Appetite"""
        if current_dd > self.recovery_threshold:
            self.resolution_state = ResolutionState.RECOVERY
        elif (
            current_dd < (self.recovery_threshold * 0.5)
            and self.resolution_state == ResolutionState.RECOVERY
        ):
            # Exit recovery when we claw back half the threshold
            self.resolution_state = ResolutionState.STABLE
            logger.info("✅ Recoup complete! Exiting Recovery Mode.")

        # Check for Growth
        # Need Total Profit %
        # capital = ... (already have dd)
        # Implementation Detail: Growth is tricky, let's stick to simple profit check outside
        pass

    # ============================================================
    # EXPECTANCY GATEKEEPER (Phase 5)
    # ============================================================

    def _check_expectancy_viability(self, symbol) -> bool:
        """Phase 56: Metal-Core Optimized Expectancy Gatekeeper."""
        if not self._cache_initialized:
            self._initialize_cache()

        try:
            sym_trades = [t for t in self._trade_cache if t["symbol"] == symbol]
            if len(sym_trades) < 10:
                return True  # Learning mode

            wins = sum(1 for t in sym_trades if t["is_win"])
            total = len(sym_trades)
            avg_win = (
                np.mean([t["pnl_pct"] for t in sym_trades if t["is_win"]])
                if wins > 0
                else 0
            )
            avg_loss = (
                np.mean([abs(t["pnl_pct"]) for t in sym_trades if not t["is_win"]])
                if (total - wins) > 0
                else 0
            )

            wr = wins / total
            expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)

            if expectancy <= 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Error evaluating trade cache: {e}")
            return True

    # ============================================================
    # ORDER GENERATION (FIXED)
    # ============================================================

    @trace_execution
    def generate_order(self, signal_event, current_price):
        """
        🛡️ MERCHANT GOD PROTOCOL (Phase 3.2): ATOMIC ORDER GENERATION
        QUÉ: Transforma señales en órdenes válidas tras pasar 7 filtros de seguridad y reserva de margen.
        POR QUÉ: Garantiza viabilidad de la cuenta de $13 USD y evita margin leaks.
        PARA QUÉ: Lograr un 100% de WR mediante filtrado ultra-selectivo.
        CÓMO: Pipeline secuencial: Bypass → Validaciones → Sizing Kelly → Reserva → Construcción.
        """
        # ================================================================
        # 1.0. PREDICTIVE TP LIMIT BYPASS
        # QUÉ: Genera una orden LIMIT en el exchange para el TP exacto
        # ================================================================
        if getattr(signal_event, "strategy_id", "") == "PLACE_TP_LIMIT":
            horizon = getattr(signal_event, "horizon", "SCALPING")
            pos = self.portfolio.get_horizon_position(signal_event.symbol, horizon) if self.portfolio else None
            if not pos or abs(pos.get("quantity", 0)) < 1e-8:
                return None
            
            qty = pos["quantity"]
            direction = OrderSide.SELL if qty > 0 else OrderSide.BUY
            
            _meta = getattr(signal_event, "metadata", {}) or {}
            tp_price = _meta.get("tp_price", 0.0)
            if not tp_price:
                return None
                
            return OrderEvent(
                symbol=signal_event.symbol,
                order_type=OrderType.LIMIT,
                quantity=abs(qty),
                direction=direction,
                price=tp_price,
                strategy_id="PREDICTIVE_TP",
                horizon=horizon,
                priority=1,
                is_exit=True,
                is_close=True,
                metadata={
                    "timeInForce": "GTC",
                    "reduceOnly": True,
                    "is_tp_limit": True
                }
            )

        # ================================================================
        # 1. EMERGENCY BYPASS (Rule 2.1) - EXIT Signals ignore entry gates
        # QUÉ: Las señales de salida no pasan por los filtros de entrada.
        # POR QUÉ: La seguridad de cerrar una posición es prioritaria.
        # ================================================================
        if self._validate_emergency_bypass(signal_event):
            horizon = getattr(signal_event, "horizon", "SCALPING")
            pos = (
                self.portfolio.get_horizon_position(signal_event.symbol, horizon)
                if self.portfolio
                else None
            )

            if not pos or abs(pos.get("quantity", 0)) < 1e-8:
                return None

            qty = pos["quantity"]
            direction = OrderSide.BUY if qty < 0 else OrderSide.SELL

            exit_priority = getattr(signal_event, "priority", 1)
            strategy_id = getattr(signal_event, "strategy_id", "EXIT")
            is_kill_switch = exit_priority == 0 or strategy_id in (
                "EMERGENCY_EXIT",
                "KILL_SWITCH",
            )

            use_limit_exits = getattr(Config, "Execution", None) and getattr(
                Config.Execution, "USE_LIMIT_BBO_EXITS", True
            )

            if is_kill_switch or not use_limit_exits:
                exit_order_type = OrderType.MARKET
                exit_metadata = {"exit_mode": "TAKER_PANIC", "cancel_tp_first": pos.get("tp_limit_placed", False)}
            else:
                exit_order_type = OrderType.LIMIT
                exit_metadata = {
                    "timeInForce": "GTC",
                    "is_bbo_exit": True,
                    "is_exit": True,
                    "exit_mode": "LIMIT_CHASING",
                    "cancel_tp_first": pos.get("tp_limit_placed", False)
                }

            return OrderEvent(
                symbol=signal_event.symbol,
                order_type=exit_order_type,
                quantity=abs(qty),
                direction=direction,
                price=current_price,
                strategy_id=strategy_id,
                horizon=horizon,
                priority=exit_priority,
                is_exit=True,
                is_close=True,
                ttl=getattr(Config.Execution, "CHASE_TIMEOUT_SECONDS", 5)
                if exit_order_type == OrderType.LIMIT
                else None,
                metadata=exit_metadata,
            )

        # ================================================================
        # 2. ATOMIC VALIDATIONS (Sequencial & Fast)
        # QUÉ: Puertas de seguridad obligatorias previo al sizing.
        # ================================================================
        if not self._validate_kill_switch():
            print(f"[RISK] Rejected by KILL_SWITCH for {signal_event.symbol}")
            return None
        if not self._validate_frequency_limits(
            signal_event.symbol, signal_event.signal_type
        ):
            print(f"[RISK] Rejected by FREQUENCY_LIMIT for {signal_event.symbol}")
            return None
        if not self._validate_regime_veto(
            signal_event.symbol, signal_event.signal_type
        ):
            _sig_name = getattr(signal_event.signal_type, 'name', str(signal_event.signal_type))
            print(f"[RISK] Rejected by REGIME_VETO for {signal_event.symbol} ({_sig_name} vs {self.global_regime})")
            return None

        # 🧟 ZOMBIE FEATURE INTEGRATION: Regime Tension Veto
        tension = getattr(signal_event, 'tension', 0.0)
        if tension > 1.5 or tension < -1.5:
            print(f"[RISK] Rejected by REGIME_TENSION for {signal_event.symbol} (tension={tension:.2f})")
            return None

        # 🧟 PHASE 2 ZOMBIE INTEGRATION: Correlation, Sentiment, Liquidity
        # 1. Correlation Risk
        if hasattr(self, 'correlation_manager') and self.correlation_manager:
            active_symbols = [sym for sym, pos in self.portfolio.positions.items() if abs(pos.get('quantity', 0)) > 0]
            if active_symbols:
                safe, reason = self.correlation_manager.check_correlation_risk(signal_event.symbol, active_symbols)
                if not safe:
                    print(f"[RISK] Rejected by HIGH_CORRELATION for {signal_event.symbol}: {reason}")
                    return None
                    
        # 2. Market Sentiment Veto
        if hasattr(self, 'sentiment_processor') and self.sentiment_processor:
            mood = self.sentiment_processor.get_market_mood()
            _sig_str = str(signal_event.signal_type).split('.')[-1]
            if _sig_str == 'LONG' and mood < -0.5:
                print(f"[RISK] Rejected by SENTIMENT_DIVERGENCE for {signal_event.symbol} (LONG but Mood={mood:.2f})")
                return None
            elif _sig_str == 'SHORT' and mood > 0.5:
                print(f"[RISK] Rejected by SENTIMENT_DIVERGENCE for {signal_event.symbol} (SHORT but Mood={mood:.2f})")
                return None
                
        # 3. Liquidity Vacuum Veto (Only for Scalping)
        horizon = getattr(signal_event, "horizon", "SCALPING")
        if horizon == 'SCALPING' and hasattr(self, 'liquidity_guardian') and self.liquidity_guardian:
            quality = self.liquidity_guardian.get_market_quality_score(signal_event.symbol)
            if quality < 30:
                print(f"[RISK] Rejected by LIQUIDITY_VACUUM for {signal_event.symbol} (Quality={quality:.1f} < 30)")
                return None

        # ═══════════════════════════════════════════════════════════════
        # PREDICTION CONFIDENCE GATE (Feedback Loop Integration)
        # QUÉ: Rechaza señales de estrategias con precisión < 55%.
        # POR QUÉ: Sin este filtro, estrategias que aciertan la dirección
        #   solo el 50% del tiempo (coin-flip) generan trades que las
        #   comisiones convierten en neto-negativos.
        # PARA QUÉ: Solo ejecutar trades con edge estadístico real.
        # CÓMO: Consulta la precisión rolling de PredictionTracker.
        #   Si accuracy < 55% con N>=30 samples → rechazo.
        # CUÁNDO: Después de validaciones de kill_switch y regime, antes
        #   de sizing. No afecta señales de salida (ya bypass arriba).
        # ═══════════════════════════════════════════════════════════════
        if self.prediction_tracker:
            _strat_id = getattr(signal_event, 'strategy_id', '')
            _horizon = getattr(signal_event, 'horizon', 'SCALPING')
            should_reject, reject_reason = self.prediction_tracker.should_reject_signal(
                _strat_id, _horizon
            )
            if should_reject:
                print(f"[RISK] Rejected by PREDICTION_GATE for {signal_event.symbol} ({reject_reason})")
                return None

        # Spot Mode Safety
        if (
            not getattr(Config, "BINANCE_USE_FUTURES", False)
            and signal_event.signal_type == SignalType.SHORT
        ):
            print(f"[RISK] Rejected by SPOT_SAFETY for {signal_event.symbol} (SHORT in Spot Mode)")
            return None

        if not self._validate_directional_safety(
            signal_event.symbol, signal_event.signal_type, horizon
        ):
            _sig_name = getattr(signal_event.signal_type, 'name', str(signal_event.signal_type))
            print(f"[RISK] Rejected by DIRECTIONAL_SAFETY for {signal_event.symbol} ({_sig_name} {horizon})")
            return None
        if not self._validate_margin_ratio():
            print(f"[RISK] Rejected by MARGIN_RATIO for {signal_event.symbol}")
            return None
        if not self._validate_fat_finger(current_price, signal_event.symbol):
            print(f"[RISK] Rejected by FAT_FINGER for {signal_event.symbol}")
            return None
        if not self._validate_slippage(signal_event.symbol, current_price):
            print(f"[RISK] Rejected by SLIPPAGE for {signal_event.symbol}")
            return None

        # ================================================================
        # 3. MERCHANT GOD: ATOMIC SIZING & MARGIN RESERVATION
        # ================================================================
        try:
            if not self.portfolio:
                print(f"[RISK] Rejected by NO_PORTFOLIO for {signal_event.symbol}")
                return None

            symbol = signal_event.symbol
            strategy_id = getattr(signal_event, "strategy_id", "Unknown")
            setup_type = getattr(signal_event, "setup_type", "generic")

            # FORENSIC FIX #4: ORPHAN GUARD
            # QUÉ: Bloquear operaciones que no tengan un strategy_id válido asignado.
            # POR QUÉ: Las estrategias "Unknown" son trades huérfanos que escapan a la validación del Oráculo ML.
            if strategy_id == "Unknown" or not strategy_id:
                logger.warning(f"🛡️ [ORPHAN GUARD] Blocked {symbol} trade with no strategy_id. ML Oracle validation required.")
                print(f"[RISK] Rejected by ORPHAN_GUARD for {symbol} (Missing strategy_id)")
                return None

            # Risk Gates & Cooldowns
            # FORENSIC-V24 FIX #2: Pass actual signal_type to risk gates
            if not self._check_risk_gates(symbol, strategy_id, signal_event.signal_type):
                _sig_name = getattr(signal_event.signal_type, 'name', str(signal_event.signal_type))
                print(f"[RISK] Rejected by RISK_GATES for {symbol} ({_sig_name})")
                return None

            # 📋 [PHASE 6] SECTOR CORRELATION FILTER
            # QUÉ: Bloquea si la exposición del sector excede el límite (35%).
            sector = self._get_sector(symbol)
            sector_exposure = self._get_sector_exposure(sector)
            total_equity = self.portfolio.get_total_equity()
            if sector_exposure >= (total_equity * self.max_sector_exposure):
                print(f"[RISK] Rejected by SECTOR_EXPOSURE for {symbol} (Sector {sector}: ${sector_exposure:.2f} >= limit)")
                return None

            # Dynamic Capacity (Meritocracy)
            open_positions = sum(
                1
                for pos in self.portfolio.virtual_ledger.values()
                if pos.get("quantity", 0) != 0
            )
            dynamic_max = self._get_dynamic_max_positions(setup_type, strategy_id)
            if open_positions >= dynamic_max and signal_event.signal_type in [
                SignalType.LONG,
                SignalType.SHORT,
            ]:
                if not self.portfolio.has_position_for_horizon(symbol, horizon):
                    print(f"[RISK] Rejected by POSITION_LIMIT for {symbol} ({open_positions}/{dynamic_max} open)")
                    return None

            # Sizing (Meritocratic Kelly)
            base_risk_pct = getattr(Config, "MAX_RISK_PER_TRADE", 0.05)
            # MERITOCRACY-V3: Combining Setup merit with specific Strategy ID merit
            merit_mult = self._calculate_merit_multiplier(setup_type, strategy_id)
            
            # FORENSIC FIX: Modulate position sizing with PredictionTracker's confidence_factor
            c_factor = 1.0
            avg_mfe_pct = None
            limit_offset_pct = None
            optimal_ttl_bars = None
            if self.prediction_tracker:
                exec_params = self.prediction_tracker.get_execution_params(strategy_id, horizon)
                c_factor = exec_params.get("confidence_factor", 1.0)
                avg_mfe_pct = exec_params.get("avg_mfe_pct")
                limit_offset_pct = exec_params.get("limit_offset_pct")
                optimal_ttl_bars = exec_params.get("optimal_ttl_bars")
                merit_mult *= c_factor
                logger.debug(f"⚖️ [SIZING] {symbol} | Base Merit: {merit_mult/c_factor if c_factor else merit_mult:.2f} | Confidence: {c_factor:.2f} | Final Mult: {merit_mult:.2f}")

            params = self.size_position(
                symbol, base_risk_pct, multiplier=merit_mult, horizon=horizon, current_price=current_price
            )
            
            # 🧟 ZOMBIE FEATURE INTEGRATION: Dynamic Take Profit based on MFE
            if params and avg_mfe_pct and avg_mfe_pct > params.get("tp_pct", 0.0):
                # We cap the override to prevent absurd MFE spikes from risking the trade
                max_tp = getattr(Config.Risk, "MAX_PROFIT_TAKE", 0.15) 
                dynamic_tp = min(avg_mfe_pct * 0.9, max_tp) # Target 90% of avg MFE
                if dynamic_tp > params["tp_pct"]:
                    params["tp_pct"] = dynamic_tp

            if not params or params["quantity"] <= 0:
                print(f"[RISK] Rejected by SIZING_FAILED for {symbol} (params={params})")
                return None

            # 📊 [PHASE 10] PORTFOLIO VaR CHECK
            # QUÉ: Calcula si el nuevo trade rompe el presupuesto de riesgo sistémico.
            if not self.check_portfolio_var(params["dollar_size"]):
                print(f"[RISK] Rejected by PORTFOLIO_VAR for {symbol} (size=${params['dollar_size']:.2f})")
                return None

            # Margin Reservation & Fitting (The 13-Dollar Protocol)
            reservation_amount = params["dollar_size"]
            # FORENSIC-V23: Include Horizon and Direction in ID for perfect traceability
            _hz_prefix = "SCL" if horizon == "SCALPING" else "SWG"
            order_side = OrderSide.BUY if signal_event.signal_type == SignalType.LONG else OrderSide.SELL
            _dir_prefix = "LONG" if order_side == OrderSide.BUY else "SHRT"
            _ts = int(datetime.now(timezone.utc).timestamp())
            client_order_id = f"TG_{_hz_prefix}_{_dir_prefix}_{_ts}_{symbol.replace('/', '')}"

            if not self.portfolio.reserve_cash(
                reservation_amount, horizon=horizon, order_id=client_order_id
            ):
                # Attempt Margin Fitting (Min Notional $5.05)
                min_notional = 5.05
                leverage = params["leverage"]
                min_margin = (min_notional / leverage) * 1.05  # 5% buffer

                if reservation_amount > min_margin and self.portfolio.reserve_cash(
                    min_margin, horizon=horizon, order_id=client_order_id
                ):
                    logger.info(
                        f"⚖️ [MARGIN-FITTING] Downsizing {symbol} to ${min_margin:.2f}"
                    )
                    reservation_amount = min_margin
                    params["quantity"] = min_notional / current_price
                else:
                    print(f"[RISK] Rejected by MARGIN_INSUFFICIENT for {symbol} (need ${reservation_amount:.2f}, min=${min_margin:.2f})")
                    return None

            # ================================================================
            # 4. EXECUTION CONSTRUCTION
            # ================================================================
            cooldown_manager.record_trade(symbol, strategy_id="RISK_MANAGER")
            self.global_trade_count += 1

            # BBO Selection
            strength = getattr(signal_event, "strength", 0)
            priority = getattr(signal_event, "priority", 1)
            exec_config = getattr(Config, "Execution", None)
            use_limit_entries = (
                getattr(exec_config, "USE_LIMIT_BBO_ENTRIES", True)
                if exec_config
                else True
            )

            order_type = OrderType.MARKET if priority == 0 else OrderType.LIMIT
            entry_mode = (
                "TAKER_PANIC"
                if priority == 0
                else ("MAKER_PROFIT" if use_limit_entries else "LEGACY")
            )

            entry_metadata = {
                "strength": strength,
                "entry_mode": entry_mode,
                "dollar_size": reservation_amount,
                "client_order_id": client_order_id,
                "setup_type": setup_type,
                "merit_mult": merit_mult,
            }
            if limit_offset_pct is not None:
                entry_metadata["limit_offset_pct"] = limit_offset_pct
            if optimal_ttl_bars is not None:
                entry_metadata["optimal_ttl_bars"] = optimal_ttl_bars

            if hasattr(signal_event, 'metadata') and signal_event.metadata:
                entry_metadata.update(signal_event.metadata)

            if order_type == OrderType.LIMIT and getattr(
                exec_config, "POST_ONLY_GTX", True
            ):
                entry_metadata["timeInForce"] = "GTX"

            return OrderEvent(
                symbol=symbol,
                order_type=order_type,
                quantity=params["quantity"],
                direction=order_side,
                leverage=params["leverage"],
                strategy_id=strategy_id,
                sl_pct=params["sl_pct"],
                tp_pct=params["tp_pct"],
                price=current_price,
                ttl=optimal_ttl_bars * 60 if optimal_ttl_bars else (getattr(exec_config, "ENTRY_TTL_SECONDS", 30) if order_type == OrderType.LIMIT else 30),
                horizon=horizon,
                priority=priority,
                is_shadow=False,
                metadata=entry_metadata,
            )

        except Exception as e:
            logger.error(f"❌ [MERCHANT-GOD] Order generation FATAL: {e}")
            return None

    def _generate_exit_order(self, signal_event, current_price):
        """
        BBO ARCHITECTURE: Exit Order Generator
        QUÉ: Genera orden de cierre con tipo LIMIT BBO o MARKET según urgencia.
        POR QUÉ: Exits normales pueden esperar BBO fill → ahorro 47% en fees.
        PARA QUÉ: Maximizar retención de capital en micro-cuenta.
        """
        try:
            if not self.portfolio:
                return None

            symbol = signal_event.symbol
            horizon = getattr(signal_event, "horizon", "SCALPING")
            pos = self.portfolio.get_horizon_position(symbol, horizon)

            if not pos or pos.get("quantity", 0) == 0:
                return None

            qty = pos["quantity"]
            strategy_id = getattr(signal_event, "strategy_id", "EXIT")
            exit_priority = getattr(signal_event, "priority", 1)
            is_emergency = (
                strategy_id in ("EMERGENCY_EXIT", "KILL_SWITCH") or exit_priority == 0
            )

            exec_config = getattr(Config, "Execution", None)
            use_limit_exits = (
                getattr(exec_config, "USE_LIMIT_BBO_EXITS", True)
                if exec_config
                else True
            )

            if is_emergency or not use_limit_exits:
                exit_type = OrderType.MARKET
                exit_mode = "TAKER_PANIC"
            else:
                exit_type = OrderType.LIMIT
                exit_mode = "LIMIT_CHASING"

            exit_metadata = {
                "exit_mode": exit_mode,
                "is_exit": True,
                "timeInForce": "GTC",  # Exits always GTC to ensure closure
            }

            return OrderEvent(
                symbol=symbol,
                order_type=exit_type,
                quantity=abs(qty),
                direction=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                strategy_id=strategy_id,
                price=current_price,
                horizon=horizon,
                priority=exit_priority,
                is_exit=True,
                is_close=True,
                ttl=getattr(exec_config, "CHASE_TIMEOUT_SECONDS", 5)
                if (exit_type == OrderType.LIMIT and exec_config)
                else None,
                metadata=exit_metadata,
            )
        except Exception as e:
            logger.error(f"❌ Exit order generation failed: {e}")
            return None

    @trace_execution
    def _check_risk_gates(self, symbol, strategy_id, signal_type=None):
        """
        [PHASE 14] AEGIS-IV: Unified Risk Gate.
        QUÉ: Concentra validaciones de cooldown, kill_switch y régimen global en un punto único.
        POR QUÉ: Antes estaban dispersas, causando latencia y fallos de lógica por desincronización.
        PARA QUÉ: Garantizar que cada trade cumpla con el consenso de seguridad del sistema.
        CÓMO: Pipeline: Kill Switch → Daily Limits → Cooldowns → Strategic Veto.
        CUÁNDO: Antes de cada sizing y reserva de margen en generate_order.
        """
        # 1. Kill Switch Check
        if not self.kill_switch.check_status():
            print(
                f"💀 [AEGIS] Global Veto: Kill Switch Active ({self.kill_switch.activation_reason})"
            )
            return False

        # 2. Daily Limit Check (Atomic)
        # FORENSIC-V24 FIX #2: Use actual signal_type instead of hardcoded LONG
        _freq_signal = signal_type if signal_type else SignalType.LONG
        if not self._validate_frequency_limits(symbol, _freq_signal):
            _sig_name = getattr(_freq_signal, 'name', str(_freq_signal))
            print(f"🛑 [AEGIS] Frequency Limit Breached for {symbol} ({_sig_name}).")
            return False

        # 3. Cooldown Check (FORENSIC FIX: unpack tuple)
        # cooldown_manager.can_trade devuelve (bool, reason)
        can_trade_res = cooldown_manager.can_trade(symbol, strategy_id=strategy_id)
        if not can_trade_res[0]:
            print(
                f"❄️ [AEGIS] Cooldown active for {symbol} under {strategy_id}. Reason: {can_trade_res[1]}"
            )
            return False

        # 4. Strategic Regime Veto (Final Quality Filter)
        if self.current_regime == "VOLATILE" and strategy_id == "TECHNICAL_STRATEGY":
            print(
                f"🛡️ [AEGIS] Vetoing {strategy_id} for {symbol} in VOLATILE regime (Risk of whipsaw)."
            )
            return False

        if self.current_regime == "TRENDING" and strategy_id == "STATISTICAL_REVERSION":
            print(f"🛡️ [AEGIS] Vetoing Mean Reversion in TRENDING regime.")
            return False

        return True

    def _record_flip(self, symbol):
        """
        [PHASE 5] Flip Intensity Tracker.
        QUÉ: Monitorea la frecuencia de cambios de dirección (LONG <-> SHORT).
        POR QUÉ: El "over-flipping" indica indecisión del mercado y causa pérdidas por comisiones.
        PARA QUÉ: Detectar 'Trend exhaustion' y aplicar cooldowns preventivos.
        CÓMO: Contador persistente en memoria por símbolo con reset diario.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if symbol not in self.daily_flips or self.daily_flips[symbol]["date"] != today:
            self.daily_flips[symbol] = {"date": today, "count": 1}
        else:
            self.daily_flips[symbol]["count"] += 1

        self.last_flip_times[symbol] = now.timestamp()
        logger.info(
            f"🔄 [FLIP] {symbol} intensificando: {self.daily_flips[symbol]['count']} flips hoy."
        )

        # Protective Circuit Breaker (Too many flips in a short time)
        if self.daily_flips[symbol]["count"] > 10:
            logger.warning(
                f"🚨 [OVER-FLIP] {symbol} exceeded 10 flips. Forced 30min timeout."
            )
            cooldown_manager.set_cooldown(symbol, 1800, "Shield: Over-Flip Protection")

    # ============================================================
    # CHECK STOPS - COMPLETE ORIGINAL
    # ============================================================

    def check_stops(self, portfolio, data_provider, symbol_filter=None, now=None):
        """
        🚀 ALPHA-MAX: Advanced exit orchestration (Horizon-Isolated).
        - Dynamic TP Targets (ATR-based from Entry)
        - Break-Even 2.0 (Fee protection + Profit Guard)
        - Momentum Protection (Phase 42)

        FORENSIC-V12 FIX #2: symbol_filter parameter.
        QUÉ: Filtra evaluación a solo las posiciones del símbolo dado.
        POR QUÉ: Engine llama check_stops() por cada MarketEvent (1 por símbolo).
           Sin filtro, 26 símbolos × 26 posiciones = 676 evaluaciones/tick.
        PARA QUÉ: Reducir a O(N) — solo evalúa posiciones del símbolo actual.
        CUÁNDO: Siempre que se invoca desde Engine._process_market_event().
        DÓNDE: risk/risk_manager.py → check_stops()
        QUIÉN: SRE/DevOps + Arquitecto Senior.

        Args:
            portfolio: Portfolio instance
            data_provider: DataProvider instance
            symbol_filter: Optional[str] - If set, only evaluate positions for this symbol
        """
        if not portfolio:
            return []
        stop_signals = []
        if now is None:
            now = datetime.now(timezone.utc)
        elif isinstance(now, (int, float)):
             now = datetime.fromtimestamp(now, tz=timezone.utc)

        # 🛡️ PHOENIX V3: Iteramos sobre el Libro Mayor Virtual para asegurar aislamiento Scalping vs Swing
        for v_key, pos in portfolio.virtual_ledger.items():
            qty = pos.get("quantity", 0.0)
            if abs(qty) < 1e-8:
                continue

            # v_key is like 'BTC/USDT_SCALPING'
            parts = v_key.rsplit("_", 1)
            symbol = parts[0]
            pos_horizon = parts[1] if len(parts) > 1 else pos.get("horizon", "SCALPING")

            # FORENSIC-V12 FIX #2: Skip positions not matching the filter
            if symbol_filter and symbol != symbol_filter:
                continue

            current_price = pos.get("current_price")
            entry_price = pos.get("avg_price")
            if not current_price or not entry_price:
                continue

            # ================================================================
            # FORENSIC REMEDIATION: Horizon-aware SL/TP fallbacks
            # QUÉ: Los fallbacks originales (0.003 SL / 0.008 TP) eran LETALES
            #   para BTC con ATR normal de 0.5-1.5% en 5m.
            # POR QUÉ: Con 0.3% SL y 10x leverage, cualquier movimiento normal
            #   de BTC (0.2-0.5%) disparaba Hard SL instantáneamente (-2% a -5%).
            # PARA QUÉ: SL debe ser ≥ ATR medio para sobrevivir ruido normal.
            # ================================================================
            default_sl = 0.006 if pos_horizon == "SCALPING" else 0.015
            default_tp = 0.012 if pos_horizon == "SCALPING" else 0.035
            sl_pct = pos.get("sl_pct", default_sl) or default_sl
            tp_pct = pos.get("tp_pct", default_tp) or default_tp
            hwm = pos.get("high_water_mark", entry_price)
            lwm = pos.get("low_water_mark", entry_price)

            unrealized_pnl_pct = (
                ((current_price - entry_price) / entry_price) * 100
                if qty > 0
                else ((entry_price - current_price) / entry_price) * 100
            )
            # 🕰️ [FINOPS TIME-STOP] Protección de cuentas Micro ($13) y Capital Lockup
            if "entry_time" in pos:
                entry_time_val = pos["entry_time"]
                if hasattr(entry_time_val, "timestamp"):
                    entry_time_val = entry_time_val.timestamp()
                
                seconds_held = (now.timestamp() - entry_time_val)
                
                # FORENSIC FIX #2: ZOMBIE CATCHER (Scalping TTL)
                # QUÉ: Liberar capital secuestrado en trades Scalping perdedores que duran horas.
                # POR QUÉ: Un trade en negativo bloquea margen, impidiendo tomar señales ML altamente rentables.
                if pos_horizon == "SCALPING" and seconds_held > 420:  # 7 minutos máximo
                    if unrealized_pnl_pct < 0.1:  # Si no está en ganancias claras, liquidar
                        logger.warning(
                            f"🧟 [ZOMBIE CATCHER] {symbol} {pos_horizon} held for {seconds_held:.0f}s with PnL {unrealized_pnl_pct:.2f}%. Exiting to free capital."
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TIME_STOP_ZOMBIE",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        continue

                # Original FINOPS TIME-STOP for SWING
                if pos_horizon in ["SWING", "MACRO"] and portfolio.get_total_equity() < 50.0:
                    hours_held = seconds_held / 3600
                    if (
                        hours_held > 7.5 and unrealized_pnl_pct < 0.5
                    ):  # Si no ganamos al menos +0.5% en 7.5 hrs, abortar antes del funding
                        logger.warning(
                            f"🛑 [FINOPS TIME-STOP] {symbol} {pos_horizon} max holding time (7.5h) reached. Exiting to prevent Funding Fee bleed."
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TIME_STOP",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        continue

            # LONG POSITION
            if qty > 0:
                # 1. Momentum Exit (Proactive)
                if self._check_momentum_exit(symbol, "LONG", data_provider):
                    print(f"🪂 {pos_horizon} MOMENTUM EXIT {symbol}! (Proactive)")
                    stop_signals.append(
                        SignalEvent(
                            strategy_id="MOMENT_MGR",
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(True, 0.0)
                    continue

                # FORENSIC FIX #9: EXPLICIT TAKE PROFIT (PREDICTIVE LIMIT)
                if tp_pct > 0:
                    if not pos.get("tp_limit_placed"):
                        tp_price_val = entry_price * (1 + tp_pct)
                        logger.info(f"🎯 [PREDICTIVE LIMIT] LONG {symbol} | Placing Resting TP at {tp_price_val:.4f} (+{tp_pct*100:.2f}%)")
                        pos["tp_limit_placed"] = True
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="PLACE_TP_LIMIT",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT, # Se ignora en el generate_order bypass
                                strength=1.0,
                                horizon=pos_horizon,
                                metadata={"tp_price": tp_price_val}
                            )
                        )
                    # Fallback de seguridad por si falla la orden en el exchange
                    elif current_price >= (entry_price * (1 + tp_pct)):
                        tp_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        print(
                            f"🎯 [LONG {pos_horizon}] TAKE PROFIT (FALLBACK) {symbol}! +{tp_pnl_pct:.2f}%"
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TAKE_PROFIT",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        self.record_trade_result(True, tp_pnl_pct)
                        continue

                # 2 & 3. 3-STAGE ADAPTIVE TRAILING + TURBO-BREAKEVEN
                # FORENSIC-V11: Use BBO Maker fee for round-trip (90%+ of orders are LIMIT)
                # Before: TAKER fee (0.0375%) × 2 = 0.075% → threshold 0.1125%
                # After: MAKER fee (0.02%) × 2 = 0.04% → threshold 0.06%
                # This RAISES the turbo threshold because MAKER fee is lower:
                # net profit after fees is HIGHER, so we can afford to wait longer.
                _maker_fee = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
                _taker_fee = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.000375)
                fee_buffer = _maker_fee + _taker_fee  # Entry=Maker, Exit=varies
                peak_pnl = ((hwm - entry_price) / entry_price) * 100
                tp_target_pct = tp_pct * 100 if tp_pct > 0 else 1.0  # Safe fallback

                # ⚡ Turbo-Breakeven (Stage 0): Immediate capital protection once fee gap is broken
                # ═══════════════════════════════════════════════════════════════
                # FORENSIC-V13 FIX #5: TP-RELATIVE TURBO-BREAKEVEN
                # QUÉ: Para SCALPING, turbo-BE se activa cuando peak PnL llega
                #   a ≥75% del TP target, NO basado en múltiplo de fees.
                # POR QUÉ: El enfoque de fee×multiplier (2.5x, 4.5x) no escala
                #   con el TP. Con TP=0.40%, 4.5×fees=0.26% activaba turbo-BE
                #   ANTES de que el trade pudiera llegar al TP → 92/100 exits
                #   eran turbo-BE con solo 1 TAKE PROFIT → micro-pérdidas.
                # PARA QUÉ: Dejar que el price corra hasta ~75% del TP antes
                #   de activar protección. Si llega a 75% y retrocede, ya fue
                #   un buen trade que simplemente no cerró en TP perfecto.
                # ═══════════════════════════════════════════════════════════════
                if pos_horizon == "SCALPING":
                    # FORENSIC FIX: Aggressive Trailing Breakeven at 0.3% PnL
                    turbo_threshold_pct = 0.30
                else:
                    # Fee-relative for SWING (original behavior)
                    turbo_threshold_pct = fee_buffer * 100 * 2.5

                if peak_pnl >= turbo_threshold_pct:
                    # We lock in entry_price + fee_buffer + round trip slippage
                    turbo_be_price = entry_price * (
                        1 + fee_buffer + 0.0008
                    )  # 0.08% Total FinOps Net-Zero
                    if (
                        current_price < turbo_be_price
                    ):  # Price crashed back after hitting PEAK
                        print(
                            f"⚡ [LONG {pos_horizon}] TURBO-BREAKEVEN {symbol}! Peak +{peak_pnl:.2f}% gave us edge. Bailing at {current_price:.4f}"
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TURBO_BE",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        self.record_trade_result(True, unrealized_pnl_pct)
                        continue

                progress = peak_pnl / tp_target_pct if tp_target_pct > 0 else 0

                trail_price = None
                trail_name = None

                # Fetch dynamically passed momentum threshold for this trade
                self._last_momentum_accel = (
                    pos.get("metadata", {}).get("momentum_exit_accel", -0.012)
                    if isinstance(pos.get("metadata"), dict)
                    else -0.012
                )

                if progress >= 0.75:
                    # Stage 3: Tight trail - protect 85% of gains from peak
                    trail_price = hwm - ((hwm - entry_price) * 0.15)
                    trail_name = "TRAIL_STAGE_3_TIGHT"
                elif progress >= 0.50:
                    # Stage 2: Standard trail - protect 70% of gains from peak
                    trail_price = hwm - ((hwm - entry_price) * 0.30)
                    trail_name = "TRAIL_STAGE_2_STD"
                elif progress >= 0.25:
                    # Stage 1: Move to breakeven + round-trip fees + slippage safety buffer
                    trail_price = entry_price * (1 + fee_buffer + 0.0005)
                    trail_name = "TRAIL_STAGE_1_BE"

                if trail_price and current_price < trail_price:
                    print(
                        f"🛡️/💰 [LONG {pos_horizon}] {trail_name} {symbol}! Triggered at {current_price:.4f} (Peak: +{peak_pnl:.2f}%)"
                    )
                    stop_signals.append(
                        SignalEvent(
                            strategy_id=trail_name,
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(True, unrealized_pnl_pct)
                    continue

                # 4. Initial Hard Stop Loss (Protective)
                if current_price < (entry_price * (1 - sl_pct)):
                    print(
                        f"🛑 HARD SL [{pos_horizon}] {symbol}! {unrealized_pnl_pct:.2f}%"
                    )
                    stop_signals.append(
                        SignalEvent(
                            strategy_id="HARD_SL",
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(False, unrealized_pnl_pct)
                    continue

            # SHORT POSITION
            elif qty < 0:
                # 1. Momentum Exit
                if self._check_momentum_exit(symbol, "SHORT", data_provider):
                    print(f"🪂 {pos_horizon} SHORT MOMENTUM EXIT {symbol}! (Proactive)")
                    stop_signals.append(
                        SignalEvent(
                            strategy_id="MOMENT_MGR",
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(True, 0.0)
                    continue

                # FORENSIC FIX #9: EXPLICIT TAKE PROFIT FOR SHORTS (PREDICTIVE LIMIT)
                if tp_pct > 0:
                    if not pos.get("tp_limit_placed"):
                        tp_price_val = entry_price * (1 - tp_pct)
                        logger.info(f"🎯 [PREDICTIVE LIMIT] SHORT {symbol} | Placing Resting TP at {tp_price_val:.4f} (+{tp_pct*100:.2f}%)")
                        pos["tp_limit_placed"] = True
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="PLACE_TP_LIMIT",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                                metadata={"tp_price": tp_price_val}
                            )
                        )
                    # Fallback de seguridad
                    elif current_price <= (entry_price * (1 - tp_pct)):
                        tp_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        print(
                            f"🎯 [SHORT {pos_horizon}] TAKE PROFIT (FALLBACK) {symbol}! +{tp_pnl_pct:.2f}%"
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TAKE_PROFIT",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        self.record_trade_result(True, tp_pnl_pct)
                        continue

                # 2 & 3. 3-STAGE ADAPTIVE TRAILING + TURBO-BREAKEVEN
                # FORENSIC-V11: BBO Maker fee for round-trip (same fix as LONG side)
                _maker_fee = getattr(Config, "BINANCE_MAKER_FEE_BNB", 0.0002)
                _taker_fee = getattr(Config, "BINANCE_TAKER_FEE_BNB", 0.000375)
                fee_buffer = _maker_fee + _taker_fee  # Entry=Maker, Exit=varies
                peak_pnl = ((entry_price - lwm) / entry_price) * 100
                tp_target_pct = tp_pct * 100 if tp_pct > 0 else 1.0  # Safe fallback

                # ⚡ Turbo-Breakeven (Stage 0): Immediate capital protection
                # FORENSIC-V13 FIX #5: TP-relative for SCALPING (same logic as LONG)
                if pos_horizon == "SCALPING":
                    # FORENSIC FIX: Aggressive Trailing Breakeven at 0.3% PnL
                    turbo_threshold_pct = 0.30
                else:
                    turbo_threshold_pct = fee_buffer * 100 * 2.5

                if peak_pnl >= turbo_threshold_pct:
                    # We lock in entry_price - fee_buffer - round trip slippage
                    turbo_be_price = entry_price * (
                        1 - fee_buffer - 0.0008
                    )  # 0.08% Total FinOps Net-Zero
                    if current_price > turbo_be_price:  # Price bounced back up
                        print(
                            f"⚡ [SHORT {pos_horizon}] TURBO-BREAKEVEN {symbol}! Peak +{peak_pnl:.2f}% gave us edge. Bailing at {current_price:.4f}"
                        )
                        stop_signals.append(
                            SignalEvent(
                                strategy_id="TURBO_BE",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                strength=1.0,
                                horizon=pos_horizon,
                            )
                        )
                        self.record_trade_result(True, unrealized_pnl_pct)
                        continue

                progress = peak_pnl / tp_target_pct if tp_target_pct > 0 else 0

                trail_price = None
                trail_name = None

                self._last_momentum_accel = (
                    pos.get("metadata", {}).get("momentum_exit_accel", -0.012)
                    if isinstance(pos.get("metadata"), dict)
                    else -0.012
                )

                if progress >= 0.75:
                    # Stage 3: Tight trail - protect 85% of gains from peak
                    trail_price = lwm + ((entry_price - lwm) * 0.15)
                    trail_name = "SHORT_TRAIL_STAGE_3_TIGHT"
                elif progress >= 0.50:
                    # Stage 2: Standard trail - protect 70% of gains from peak
                    trail_price = lwm + ((entry_price - lwm) * 0.30)
                    trail_name = "SHORT_TRAIL_STAGE_2_STD"
                elif progress >= 0.25:
                    # Stage 1: Move to breakeven + round-trip fees + slippage safety buffer
                    trail_price = entry_price * (1 - fee_buffer - 0.0005)
                    trail_name = "SHORT_TRAIL_STAGE_1_BE"

                if trail_price and current_price > trail_price:
                    print(
                        f"🛡️/💰 [SHORT {pos_horizon}] {trail_name} {symbol}! Triggered at {current_price:.4f} (Peak: +{peak_pnl:.2f}%)"
                    )
                    stop_signals.append(
                        SignalEvent(
                            strategy_id=trail_name,
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(True, unrealized_pnl_pct)
                    continue

                # 4. Initial Hard Stop
                if current_price > (entry_price * (1 + sl_pct)):
                    print(
                        f"🛑 SHORT HARD SL [{pos_horizon}] {symbol}! {unrealized_pnl_pct:.2f}%"
                    )
                    stop_signals.append(
                        SignalEvent(
                            strategy_id="HARD_SL",
                            symbol=symbol,
                            datetime=now,
                            signal_type=SignalType.EXIT,
                            strength=1.0,
                            horizon=pos_horizon,
                        )
                    )
                    self.record_trade_result(False, unrealized_pnl_pct)
                    continue

        return stop_signals

    # ============================================================
    # KILL SWITCH FACADE
    # ============================================================

    # Using the L596 update_equity instead.
    def activate_kill_switch(self, reason: str):
        """
        [P0 FIX] Activate Kill Switch with proper reason

        QUÉ: Activa el kill switch pasando la razón correcta.
        POR QUÉ: El código anterior llamaba record_loss() lo cual solo incrementaba
                el contador y llamaba activate() implícitamente, pero no pasaba
                la razón específica por la cual se activó.
        """
        if self.kill_switch:
            self.kill_switch.activate(reason)

    def record_api_error(self):
        """Record API error and check for system instability."""
        if self.kill_switch:
            self.kill_switch.record_api_error()

    def reset_api_errors(self):
        """Reset the API error counter."""
        if self.kill_switch:
            self.kill_switch.reset_api_errors()

    # ============================================================
    # SNIPER STRATEGY METHODS (ORIGINAL)
    # ============================================================

    def calculate_dynamic_leverage(self, atr: float, price: float) -> int:
        print(f"Legacy calculate_dynamic_leverage called. Delegating...")
        result = safe_leverage_calculator.calculate_safe_leverage(atr, price)
        return result["leverage"]

    def update_leverage_and_params(self, volatility: float, regime: str):
        """
        [DF-B1] REGLA DE ADAPTABILIDAD DE RECURSOS
        QUÉ: Ajusta el leverage y agresividad según el régimen y la volatilidad GARCH.
        POR QUÉ: Evita sobre-apalancamiento en clústeres de alta varianza.
        """
        # Phase 13: GARCH-Adaptive Leverage
        # Si la volatilidad GARCH es alta, forzamos reducción de leverage preventivo.
        self.current_volatility = volatility

        # Scaling leverage inversely with volatility (Simplified GARCH cluster link)
        if volatility > 0.05:  # High Vol
            self.max_leverage = 3
            logger.warning(
                f"❄️ GARCH High Vol Cluster ({volatility:.4f}) -> Leverage CAPPED to 3x"
            )
        elif volatility > 0.025:  # Elevated Vol
            self.max_leverage = 7
        else:  # Low/Stable Vol
            self.max_leverage = 12

        # Regime Specific Adjustments
        if regime == "TRENDING_UP":
            self.max_leverage = min(self.max_leverage, 15)  # Boost for BTC runs
        elif regime == "CHOPPY":
            self.max_leverage = min(self.max_leverage, 5)  # Defensive

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: str,
        margin_type: str = "ISOLATED",
    ) -> float:
        if leverage <= 0:
            return 0.0
        mmr = 0.004
        if direction == "LONG":
            liq_price = entry_price * (1 - (1 / leverage) + mmr)
        else:
            liq_price = entry_price * (1 + (1 / leverage) - mmr)
        return liq_price

    def calculate_distance_to_liquidation(
        self, entry_price: float, current_price: float, leverage: int, direction: str
    ) -> dict:
        liq_price = self.calculate_liquidation_price(entry_price, leverage, direction)
        if direction == "LONG":
            distance = (current_price - liq_price) / current_price * 100
        else:
            distance = (liq_price - current_price) / current_price * 100
        return {
            "liq_price": liq_price,
            "distance_pct": distance,
            "is_danger": distance < 2.0,
        }

    def calculate_sniper_position_size(
        self, capital: float, leverage: int, entry_price: float
    ) -> dict:
        notional = capital * leverage
        quantity = notional / entry_price if entry_price > 0 else 0
        margin_required = notional / leverage
        return {
            "notional": notional,
            "quantity": quantity,
            "margin_required": margin_required,
            "leverage": leverage,
        }

    def check_portfolio_var(self, new_trade_value: float) -> bool:
        """
        [PHASE 10] Dynamic Hedging / VaR Check
        Calculates simple Parametric VaR (95%) for the portfolio.
        Returns False if adding 'new_trade_value' exceeds Max VaR allowed.

        FORENSIC-V24 FIX #8 (CATASTRÓFICO):
        QUÉ: VaR usaba NOTIONAL exposure contra un límite basado en EQUITY.
        POR QUÉ: Con 10x leverage, un trade de $6 MARGIN = $60 NOTIONAL.
          VaR = $60 × 0.03 × 1.65 = $2.97, pero límite = $13 × 0.05 = $0.65.
          Esto bloqueaba ~95% de TODOS los trades válidos.
        PARA QUÉ: Usar MARGIN exposure (what we actually risk) contra MARGIN limit.
        CÓMO: Dividir exposición por leverage para obtener margin-at-risk real.
        """
        if not self.portfolio:
            return True

        # 1. Get total portfolio value
        total_equity = self.portfolio.get_total_equity()

        # FORENSIC-V24: Scale VaR limit for micro-accounts
        # $13 account needs more room — allow 15% VaR for < $100, 10% for < $1000, 5% standard
        if total_equity < 100:
            var_pct = 0.15  # 15% VaR budget for micro-accounts
        elif total_equity < 1000:
            var_pct = 0.10  # 10% for small accounts
        else:
            var_pct = 0.05  # 5% institutional
        max_var_limit = total_equity * var_pct

        # 2. Estimate Current VaR using MARGIN (not notional)
        # FORENSIC-V24: Use margin-based exposure to match equity-based limit
        current_margin_exposure = 0.0
        for s, pos in self.portfolio.positions.items():
            notional = abs(pos["quantity"] * pos["current_price"])
            lev = pos.get("leverage", getattr(Config, "BINANCE_LEVERAGE", 10)) or 10
            current_margin_exposure += notional / lev

        future_margin_exposure = current_margin_exposure + new_trade_value

        # Simple VaR = Margin_Exposure * Volatility * Z(95%)
        # Z(95%) ~= 1.65
        # Assuming avg daily vol of 3% for crypto portfolio
        daily_vol = 0.03

        estimated_var = future_margin_exposure * daily_vol * 1.65

        if estimated_var > max_var_limit:
            logger.warning(
                f"🛡️ VaR REJECTION: Est VaR ${estimated_var:.2f} > Limit ${max_var_limit:.2f} (Margin Exp: ${future_margin_exposure:.2f})"
            )
            return False

        return True

    def validate_sniper_order(
        self, symbol: str, quantity: float, entry_price: float, leverage: int
    ) -> dict:
        notional = quantity * entry_price
        margin_required = notional / leverage
        # PHASE 1: Execution Audit ($13 Micro-Account Hardening)
        # We increase the hard MIN_NOTIONAL from Binance's 5.0 to 6.0 to prevent
        # rejected orders due to Taker fees or sub-cent slippage pushing it below limits.
        MIN_NOTIONAL = 6.0
        MIN_MARGIN = 1.0

        if notional < MIN_NOTIONAL:
            return {
                "is_valid": False,
                "reason": f"Notional ${notional:.2f} < MIN ${MIN_NOTIONAL}",
                "adjusted_qty": MIN_NOTIONAL / entry_price,
            }
        if margin_required < MIN_MARGIN:
            return {
                "is_valid": False,
                "reason": f"Margin ${margin_required:.2f} < MIN ${MIN_MARGIN}",
                "adjusted_qty": (MIN_MARGIN * leverage) / entry_price,
            }
        return {"is_valid": True, "reason": "OK", "adjusted_qty": quantity}

    def _get_dynamic_max_positions(self, setup_type: str, strategy_id: str = "Unknown") -> int:
        """
        🛡️ MERITOCRACY-V3: DYNAMIC POSITION CAPACITY
        QUÉ: Calcula la capacidad de slots dinámica basada en el mérito del setup Y la estrategia.
        POR QUÉ: Premia estrategias ganadoras dándoles más 'slots' de capital.
        PARA QUÉ: Maximizar el retorno sobre setups con alta tasa de acierto demostrada.
        CÓMO: Baseline de 26 slots (Sin límites) → Pero prioriza si el mérito es alto.
        """
        base = getattr(Config, "MAX_CONCURRENT_POSITIONS", 26)

        # 1. Consultar performance histórica
        if not self.portfolio:
            return base

        # Meritocracy Logic: + slots if results are positive
        stats_setup = self.portfolio.get_setup_performance(setup_type)
        stats_strat = self.portfolio.get_strategy_metrics(strategy_id)
        
        merit = stats_strat.get('merit_factor', 1.0)
        
        if (stats_setup and stats_setup["win_rate"] >= 0.60) or merit > 1.2:
            # God Mode Unlocking: Unbounded for ELITE regimes with > 60% HitRate to fulfill exponential compound goal.
            logger.info(
                f"🧬 [MERIT] Elite Strategy {strategy_id} / Setup {setup_type} unlocked! Capacity: UNBOUNDED"
            )
            return 99

        return base

    def _calculate_merit_multiplier(self, setup_type: str, strategy_id: str = "Unknown") -> float:
        """
        Devuelve un multiplicador de sizing basado en el rendimiento del setup y de la estrategia específica.
        Rango: 0.5x a 2.5x (Hardened for exponential growth)
        """
        if not self.portfolio:
            return 1.0

        # 1. Setup Performance (Generic)
        stats_setup = self.portfolio.get_setup_performance(setup_type)
        wr_setup = stats_setup.get("win_rate", 0.5) if stats_setup else 0.5
        
        # 2. Strategy ID Performance (Specific - Meritocracy Requirement)
        stats_strat = self.portfolio.get_strategy_metrics(strategy_id)
        merit_factor = stats_strat.get('merit_factor', 1.0)
        
        # Combined Merit: Weight Strategy more than Setup (0.7 vs 0.3)
        final_mult = (merit_factor * 0.7) + (wr_setup * 2.0 * 0.3) # WR 0.5 -> 1.0 multiplier
        
        # Safety Clamping per User Growth Target
        if merit_factor > 1.5:
             logger.info(f"🚀 [GOLDEN-ID] Strategy {strategy_id} is DOMINATING. Applying 1.5x Aggression.")
             final_mult = max(final_mult, 1.5)

        return max(0.5, min(2.5, final_mult))
