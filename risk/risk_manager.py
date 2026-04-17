import os
import sys
import time
import math
import traceback
import numpy as np
from collections import deque
from datetime import timedelta, datetime, timezone
from decimal import Decimal, getcontext

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
from utils.math_kernel import calculate_garch_jit, compute_kelly_fraction_jit, extract_kelly_stats_jit, compute_cvar_jit



# ============================================================
# SCIENTIFIC RISK TOOLS (FIXED)
# ============================================================

class FeeCalculator:
    """Cálculo preciso de fees - CORREGIDO"""
    # [SOVEREIGN-DEPLOY] Dynamic Fee Awareness
    TAKER_FEE = Config.BINANCE_TAKER_FEE_BNB  # Default fallback
    MAKER_FEE = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002) 
    IS_DYNAMIC = False

    @classmethod
    def update_dynamic_fees(cls, maker: float, taker: float):
        cls.MAKER_FEE = maker
        cls.TAKER_FEE = taker
        cls.IS_DYNAMIC = True
        logger.info(f"💰 [FEE-AWARENESS] Dynamic Commission Rates Embedded: Maker {maker*100:.4f}% | Taker {taker*100:.4f}%")
        
    @staticmethod
    def calculate_round_trip_fee(notional_value: float, order_type: str = 'LIMIT') -> float:
        """
        SUPREMO-V4: Simulador de comisiones REALISTA (Binance Futures).
        - MAKER (LIMIT): 0.02% (con BNB)
        - TAKER (MARKET): 0.0375% (con BNB)
        """
        fee_rate = FeeCalculator.MAKER_FEE
        if order_type.upper() == 'MARKET':
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
        if price is None: return False
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
        self.current_regime = 'RANGING'
        
        # Scientific Tools
        self.cvar_calc = CVaRCalculator()
        self.fee_calc = FeeCalculator()
        
        # Kill Switch
        self.kill_switch = KillSwitch(portfolio=self.portfolio)
        
        # Kelly Stats (Dynamic Genesis V5.0)
        self.win_count = 0
        self.loss_count = 0
        # Instead of fixed 0.52, this now acts only as the absolute 'cold start' prior
        self.bootstrap_win_rate = getattr(Config.Risk, 'DEFAULT_BOOTSTRAP_WR', 0.52)
        self.bootstrap_trades = getattr(Config.Risk, 'BOOTSTRAP_TRADES', 20)
        
        # Growth Phases (CALIBRADO)
        self.LEVERAGE_GROWTH = 10
        self.POSITION_PCT_GROWTH = 0.30  # 30% en growth
        
        # Phase 5: Flipping State
        self.daily_flips = {}  # {symbol: {date: "YYYY-MM-DD", count: N}}
        self.last_flip_times = {} # {symbol: timestamp}
        self.daily_trade_logs = {} # {date: {symbol: count}}
        self.global_trade_count = 0 # Optimized global counter for Level VII
        self.MAX_TRADES_PER_SYMBOL = getattr(Config, 'MAX_TRADES_PER_SYMBOL', 15)
        self.MAX_TRADES_TOTAL = getattr(Config, 'MAX_TRADES_TOTAL', 100)
        
        # Phase 6: Stress Testing
        self.stress_score = 100.0 # Default perfect score (0% Ruin Risk)
        self.last_stress_check = 0
        self.stress_check_interval = 3600 # Check every hour

        # Meta-Brain Integration (Phase 7)
        self.strategy_selector = None # Set by Engine
        
        # Execution Caps [SS-006 FIX: Removed duplicate MAX_TRADES_TOTAL hardcode]
        self.global_regime = 'UNKNOWN' # BTC Leader (Phase 8)
        
        # Phase 14-71: Dynamic Capital Allocation
        self.resolution_state = ResolutionState.STABLE
        self.recovery_threshold = 0.0075 # 0.75% Drawdown triggers defensive mode (halved risk)
        self.growth_threshold = 0.05    # 5% Profit triggers growth
        
        # Phase 42: Momentum Exit Thresholds
        self.MOMENTUM_EXIT_THRESHOLD = 0.015 # 1.5% drop in 1m bars for long exit
        self.momentum_cache = {} # {symbol: deque(maxlen=5)}

        # PHASE 56: Metal-Core Optimized Cache
        self._trade_cache = deque(maxlen=2000) # List of dicts: {'is_win': bool, 'pnl_pct': float, 'symbol': str}
        self._cache_initialized = False
        self._last_day_str = 0 # Integer YYYYMMDD for fast comparison
        self._status_cache = {}
        self._last_status_read = 0
        
        # Phase L: Sector Correlation Filter
        self.max_sector_exposure = 0.35 # Max 35% of capital per sector
        self.symbol_sectors = {
            "BTCUSDT": "MAJOR", "ETHUSDT": "MAJOR", "ETCUSDT": "MAJOR",
            "SOLUSDT": "LAYER1", "AVAXUSDT": "LAYER1", "DOTUSDT": "LAYER1", 
            "NEARUSDT": "LAYER1", "ADAUSDT": "LAYER1", "TRXUSDT": "LAYER1", 
            "ATOMUSDT": "LAYER1", "APTUSDT": "LAYER1",
            "DOGEUSDT": "MEME", "SHIBUSDT": "MEME", "PEPEUSDT": "MEME",
            "LINKUSDT": "DEFI", "UNIUSDT": "DEFI", "ARBUSDT": "DEFI", "OPUSDT": "DEFI",
            "MATICUSDT": "SCALING",
            "LTCUSDT": "PAYMENT", "BCHUSDT": "PAYMENT",
            "FILUSDT": "DEP_WEB3", "ICPUSDT": "DEP_WEB3"
        }
        
        # Phase 14: Funding & Rebate Tools
        self.funding_evasion_threshold = 0.0003 # 0.03%
        self.funding_buffer_minutes = 15
        self.rebate_priority_mode = getattr(Config, 'REBATE_PRIORITY', True)

        # Sovereign-Deploy: Kill Switch L1 & Fractional Kelly
        self.consecutive_losses = {}

    def _get_sector(self, symbol: str) -> str:
        """Standardized symbol to sector mapping."""
        # Normalize symbol for lookup
        clean_sym = symbol.replace('/', '').upper()
        if not clean_sym.endswith('USDT'): clean_sym += 'USDT'
        return self.symbol_sectors.get(clean_sym, "ALT")

    def _get_sector_exposure(self, sector: str) -> float:
        """Returns total notional exposure for a specific sector."""
        if not self.portfolio: return 0.0
        exposure = 0.0
        for sym, pos in self.portfolio.positions.items():
            if self._get_sector(sym) == sector:
                qty = pos.get('quantity', 0)
                price = pos.get('current_price', pos.get('avg_price', 0))
                exposure += abs(qty * price)
        return exposure
        
    def _initialize_cache(self):
        """QUÉ: Carga inicial de trades a memoria para evitar I/O futuro."""
        if self._cache_initialized: return
        try:
            dh = get_data_handler()
            # Try to load recent trades from CSV once
            csv_path = "dashboard/data/futures/trades.csv"
            if os.path.exists(csv_path):
                trades = dh.load_trades_df(csv_path)
                if not trades.empty:
                    for _, t in trades.iterrows():
                        is_win = t.get('net_pnl', 0) > 0
                        pnl = t.get('net_pnl', 0) / (t.get('entry_price', 1) * t.get('quantity', 1)) if t.get('entry_price', 0) > 0 else 0
                        self._trade_cache.append({
                            'is_win': is_win,
                            'pnl_pct': pnl,
                            'symbol': t.get('symbol', '')
                        })
                        # Update counts for Kelly/WR
                        if is_win: self.win_count += 1
                        else: self.loss_count += 1
                        self.cvar_calc.update(pnl)
            self._cache_initialized = True
            logger.info(f"⚡ [RiskMgr] Meta-Core Cache Initialized with {len(self._trade_cache)} trades.")
        except Exception as e:
            logger.error(f"Cache Init Failed: {e}")
            self._cache_initialized = True # Don't retry per tick

    # ============================================================
    # 🛡️ SUPREMO-V3: ATOMIC VALIDATION PIPELINE (ZERO-TRUST)
    # ============================================================

    def _validate_fat_finger(self, price, symbol):
        """
        AUDIT DEPT C: Sanity Check (>5% Deviation)
        Prevents orders with absurd prices due to API errors or bugs.
        """
        if price <= 0: return False
        
        # In a real scenario, we'd compare against a 1-minute moving average or order book mid-price.
        # Here we use the last known price from Portfolio if available, or just pass if first trade.
        last_price = None
        if self.portfolio and symbol in self.portfolio.positions:
             last_price = self.portfolio.positions[symbol].get('current_price')
        
        if last_price and last_price > 0:
             deviation = abs(price - last_price) / last_price
             if deviation > 0.05: # > 5% Deviation
                  logger.critical(f"🛑 FAT FINGER BLOCKED {symbol}: Price {price} deviates {deviation*100:.1f}% from {last_price}")
                  return False
        return True

    def _validate_slippage(self, symbol, current_price):
        """
        [SOVEREIGN-DEPLOY] Liquidity Awareness (Slippage < 0.1%)
        Estima un worst-case slippage utilizando el spread y la volatilidad local.
        """
        max_allowed_slippage = getattr(Config, 'MAX_SLIPPAGE_PCT', 0.001)
        
        # Intentaremos estimar el slippage basado en el ATR o data_handler si existe
        dh = get_data_handler()
        if dh and hasattr(dh, 'get_latest_bars'):
            try:
                bars = dh.get_latest_bars(symbol, n=5)
                if bars is not None and len(bars) > 1:
                    # Calculamos el true range promedio para esta volatilidad cortan high-low / close
                    recent_volatility = (bars[-1]['high'] - bars[-1]['low']) / current_price
                    est_slippage = recent_volatility * 0.15 # Heuristic rule of thumb for illiquidity
                    
                    if est_slippage > max_allowed_slippage:
                        logger.warning(f"🛑 [LIQUIDITY] Slippage check failed for {symbol}: Est {est_slippage*100:.3f}% > Max {max_allowed_slippage*100:.3f}%")
                        return False
            except Exception as e:
                logger.error(f"Error in slippage validation (safe fallback triggered): {e}")
                return True
        return True

    def _validate_emergency_bypass(self, signal_event):
        """QUÉ: Bypass instantáneo para señales de salida."""
        return signal_event.signal_type == SignalType.EXIT

    def _validate_kill_switch(self):
        """Valida estado global del sistema."""
        if not self.kill_switch.check_status():
            logger.warning(f"💀 Kill Switch Active: {self.kill_switch.activation_reason}")
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
        if symbol == 'BTC/USDT': return True
        
        # [PHASE 5] Bypass Global Veto if the asset has EXTREME relative strength (Hedging)
        if self.portfolio and hasattr(self.portfolio, 'relative_strength_scores'):
             is_long = signal_type == SignalType.LONG
             rs_mult = self.portfolio.get_allocation_multiplier(symbol, is_long)
             if rs_mult >= 1.3:
                 logger.info(f"🛡️ [Veto Bypass] Allowing {signal_type.name} on {symbol} despite Global Regime (Relative Strength Hedging).")
                 return True
                 
        # SUPREMO-V4: Filtro de horario NY (15:00 - 18:00 UTC)
        # La volatilidad agresiva de NY rompe los rangos de scalping. Reducimos exposición.
        now_utc = datetime.now(timezone.utc)
        if 15 <= now_utc.hour <= 18:
            logger.warning(f"🕒 [NY-FILTER] Critical Volatility Zone (15-18 UTC). Applying extra caution for {symbol}.")
            # Nota: El tamaño de posición se reducirá en la lógica de Kelly o allocation si es necesario,
            # pero aquí podemos aplicar un veto preventivo si el ADX es > 30 (tendencia violenta).
            dh = get_data_handler()
            if dh:
                bars = dh.get_latest_bars(symbol, n=1)
                if bars is not None and len(bars) > 0:
                    # Si el ADX es alto en NY, no operamos scalping de reversión
                    if bars[-1].get('adx', 0) > 30:
                        logger.warning(f"🛑 [NY-FILTER] High ADX ({bars[-1]['adx']:.1f}) in NY session. Vetoing {signal_type.name} {symbol}.")
                        return False

        if self.global_regime == 'TRENDING_BEAR' and signal_type == SignalType.LONG:
            logger.warning(f"🛡️ [Veto] Blocking LONG {symbol} (Global: Bearish).")
            return False
        if self.global_regime == 'TRENDING_BULL' and signal_type == SignalType.SHORT:
            logger.warning(f"🛡️ [Veto] Blocking SHORT {symbol} (Global: Bullish).")
            return False
        return True

    def _validate_directional_safety(self, symbol, signal_type, horizon: str = 'SCALPING'):
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
            
        qty = v_pos.get('quantity', 0)
        if qty == 0:
            return True
        
        # Block same-direction duplicates (never stack)
        if qty > 0 and signal_type == SignalType.LONG:
            return False
        if qty < 0 and signal_type == SignalType.SHORT:
            return False
            
        # FORENSIC-V11: Allow opposite-direction flips ONLY if losing > 0.3%
        # This prevents churning on noise but enables recovery from bad entries
        entry_price = v_pos.get('entry_price', 0)
        if entry_price > 0:
            current_price = self.portfolio.last_prices.get(symbol, entry_price)
            if qty > 0:  # Currently LONG, new signal is SHORT
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:  # Currently SHORT, new signal is LONG
                unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
            if unrealized_pnl_pct < -0.003:  # Losing > 0.3%
                logger.info(f"🔄 [{v_key}] FLIP ALLOWED: Current PnL {unrealized_pnl_pct*100:.2f}% < -0.3%. Permitting opposite signal.")
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
            if not funding_info: return True
            
            rate = funding_info.get('last_funding_rate', 0)
            next_funding_time = funding_info.get('next_funding_time', 0)
            
            # Caso LONG: Funding positivo alto (pagamos por estar largos)
            if side == OrderSide.BUY and rate > self.funding_evasion_threshold:
                time_to_funding = (next_funding_time - datetime.now(timezone.utc).timestamp()) / 60
                if 0 < time_to_funding < self.funding_buffer_minutes:
                    logger.warning(f"💸 [FundingGuard] VETO LONG {symbol}: Rate {rate*100:.3f}% incoming in {time_to_funding:.1f}m.")
                    return False
                    
            # Caso SHORT: Funding negativo alto (pagamos por estar cortos)
            elif side == OrderSide.SELL and rate < -self.funding_evasion_threshold:
                time_to_funding = (next_funding_time - datetime.now(timezone.utc).timestamp()) / 60
                if 0 < time_to_funding < self.funding_buffer_minutes:
                    logger.warning(f"💸 [FundingGuard] VETO SHORT {symbol}: Rate {rate*100:.3f}% incoming in {time_to_funding:.1f}m.")
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
            closes = bars['close']
            last_ret = (closes[-1] - closes[-2]) / closes[-2]
            accel = (closes[-1] - closes[-3]) / closes[-3] # 2m change
            
            # Evolutionary Momentum Exit (Read from position or default)
            # Since _check_momentum_exit is proactive, we need the position's parameter
            # We must fetch it from the portfolio.
            # Wait, _check_momentum_exit does not receive `pos`. Let's assume -0.012 default
            # unless we pass it.
            # We will use a safe default here, but it's better if we pass accel_threshold dynamically.
            accel_threshold = getattr(self, '_last_momentum_accel', -0.012)
            
            if side == 'LONG':
                # Momentum is strongly negative
                if last_ret < (accel_threshold * 0.6) or accel < accel_threshold:
                    logger.warning(f"🪂 [RiskMgr] MOMENTUM EXIT {symbol}: Long dumped {accel*100:.2f}% in 2m. GTFO.")
                    return True
            else:
                # Momentum is strongly positive (Against Short)
                if last_ret > (-accel_threshold * 0.6) or accel > -accel_threshold:
                    logger.warning(f"🪂 [RiskMgr] MOMENTUM EXIT {symbol}: Short squeezed {accel*100:.2f}% in 2m. GTFO.")
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
        if regime in ['TRENDING', 'RANGING', 'VOLATILE', 'STAGNANT', 'MIXED', 'TRENDING_BULL', 'TRENDING_BEAR', 'CHOPPY', 'ZOMBIE', 'MEAN_REVERTING']:
            if self.current_regime != regime:
                logger.info(f"⚖️ [RiskManager] Regime Change: {self.current_regime} -> {regime}")
                self.current_regime = regime

    def update_global_regime(self, global_regime: str):
        """
        BTC Leader Broadcasting (Phase 8).
        """
        if self.global_regime != global_regime:
            self.global_regime = global_regime
            if global_regime == 'TRENDING_BEAR':
                logger.warning("🛡️ [RiskMgr] GLOBAL VETO: BTC is Bearish. Restricting Altcoin Longs.")
            elif global_regime == 'TRENDING_BULL':
                logger.info("🐂 [RiskMgr] Global Sentimens: BTC is Bullish. Opportunity window open.")
        
    def get_regime(self):
        return self.current_regime

    def check_volatility_shock(self, symbol, returns):
        """
        [PHASE II] GARCH Volatility Shock Circuit Breaker.
        If Realized Vol > 2.5 * Forecasted GARCH Vol -> KILL SWITCH.
        """
        try:
            if len(returns) < 50: return # Insufficient data
            
            # 1. Forecast GARCH Variance
            garch_vars = calculate_garch_jit(np.array(returns, dtype=np.float64))
            forecast_vol = np.sqrt(garch_vars[-1])
            
            # 2. Realized Volatility (Last 10 bars)
            realized_vol = np.std(returns[-10:])
            
            # 3. Check for Shock
            if realized_vol > 2.5 * forecast_vol and realized_vol > 0.01: # Min 1% vol to trigger
                logger.critical(f"🛑 [CIRCUIT BREAKER] GARCH SHOCK on {symbol}! Realized={realized_vol:.4f} > 2.5x GARCH={forecast_vol:.4f}")
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
        if self.portfolio and hasattr(self.portfolio, 'get_statistics'):
            stats = self.portfolio.get_statistics()
            port_total = stats.get('total_trades', 0)
            port_wr = stats.get('win_rate', 0.0)
            
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
                return (real_wr * weight_real) + (self.bootstrap_win_rate * (1 - weight_real))
            return self.bootstrap_win_rate
        return self.win_count / total if total > 0 else 0.5
        
    def get_bayesian_win_rate(self) -> float:
        """Phase 6: Bayesian Posterior Win Rate (Scientific)."""
        # Use Bayesian Inference for more robust "Real" Win Rate for optimization
        return StatisticsPro.bayesian_win_rate(self.win_count, self.loss_count, prior_alpha=10, prior_beta=10)

    def _compute_kelly_math(self, p: float, b: float, apply_mult: bool = True) -> float:
        """
        [PRECISION-AXIOMA] Core math for Kelly Criterion via Numba JIT.
        Eliminates the millisecond-latency of Python Decimal overhead.
        """
        try:
            # Defensive Scaling (Risk Fortress)
            kelly_mult = 0.25 # Quarter-Kelly for Scalping volatility
            
            # Clamp between 0% and 40% exposure
            clamped = compute_kelly_fraction_jit(
                p=float(p), 
                b=float(b), 
                apply_mult=apply_mult, 
                kelly_mult=float(kelly_mult), 
                stress_score=float(self.stress_score), 
                max_exposure=0.40
            )
            
            logger.debug(f"📐 [Axioma-Kelly NANO] P:{p:.3f} B:{b:.3f} Final:{clamped:.4f}")
            return float(clamped)
            
        except Exception as e:
            logger.error(f"❌ [AXIOMA] Nano Kelly calculation failed: {e}. Defaulting to 0.0")
            return 0.0

    def calculate_kelly_fraction(self, symbol: str = "", strategy_id: str = None, rr_ratio: float = 0.75, signal_event=None) -> float:
        """
        [PHASE 13] ALPHA-SHIELD: Dynamic Kelly Sizing
        QUÉ: Calcula la fracción óptima de Kelly basada en el performance real del símbolo/estrategia.
        POR QUÉ: Maximiza el crecimiento geométrico mientras protege contra la ruina.
        """
        try:
            # 1. Gather Stats from Cache (PHASE 56: O(1) in-memory)
            trades = [t for t in self._trade_cache if (not symbol or t['symbol'] == symbol) and (not strategy_id or t.get('strategy_id') == strategy_id)]
            
            if len(trades) < 10:
                # Fallback to Bayesian Win Rate if no symbol data
                p = self.get_bayesian_win_rate()
                b = rr_ratio # Payoff ratio
            else:
                # [NANO-SPEED] Use compiled kernel for stats
                pnl_arr = np.array([t['pnl_pct'] for t in trades], dtype=np.float64)
                is_win_arr = np.array([t['is_win'] for t in trades], dtype=np.bool_)
                p, b = extract_kelly_stats_jit(pnl_arr, is_win_arr)
                
            # 2. Kelly Formula (JIT Delegated)
            kelly_frac_float = self._compute_kelly_math(p, b, apply_mult=False)
            kelly = kelly_frac_float
            
            # 3. Defensive Scaling (Risk Fortress)
            # SOVEREIGN-DEPLOY: Absolute Fractional Kelly Enforcement (f*/10)
            kelly_mult = getattr(Config.Strategies, 'ML_KELLY_FRACTION', 0.25)
            
            # Extreme Defense: If Ruin Risk (Stress Score) is low
            if self.stress_score < 90: kelly_mult = 0.25 # Quarter-Kelly
            
            # AEGIS-ULTRA: Systemic Risk Shield (Contagion)
            # If fleet correlation is high, reduce size to avoid synchronized drawdowns
            if hasattr(self, 'fleet_correlation') and self.fleet_correlation > 0.85:
                 logger.warning(f"🚨 SYSTEMIC RISK: Fleet Correlation {self.fleet_correlation:.2f}. Reducing Size by 50%.")
                 kelly_mult *= 0.5
            
            fractional_kelly = max(0.0, kelly * float(kelly_mult))
            
            # 4. Symbol Isolation & Sector Blocker
            if signal_event and hasattr(signal_event, 'symbol'):
                if not self.validate_symbol_isolation(signal_event.symbol):
                    return 0.0
                
                sector = self._get_sector(signal_event.symbol)
                current_sector_exposure = self._get_sector_exposure(sector)
                capital = self.portfolio.get_total_equity() if self.portfolio else 15.0
                if current_sector_exposure >= (capital * self.max_sector_exposure):
                    logger.warning(f"🚫 Sector limit reached: {sector}")
                    return 0.0

            # 5. Final Clamp
            return float(max(0.05, min(fractional_kelly, 0.40))) # Min 5%, Max 40% (Aggressive for $12)

        except Exception as e:
            logger.error(f"Kelly Error: {e}")
            return 0.15 # Safe Default
            
    def validate_symbol_isolation(self, symbol: str) -> bool:
        """
        [PHASE 14] Memory Isolation Check
        QUÉ: Verifica que no excedamos el presupuesto de memoria para 20 símbolos.
        POR QUÉ: Evitar fugas de memoria y degradación de performance en HFT.
        """
        active_symbols = 0
        if self.portfolio:
             active_symbols = sum(1 for pos in self.portfolio.positions.values() if pos['quantity'] != 0)
        
        # Budget: 20 Símbolos Máximo para estabilidad micro-latencia
        if active_symbols >= 20 and not (self.portfolio and symbol in self.portfolio.positions and self.portfolio.positions[symbol]['quantity'] != 0):
             logger.critical(f"🛑 [ISOLATION] Memory Budget Exceeded! Blocking {symbol}.")
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
            if symbol: self.consecutive_losses[symbol] = 0
        else:
            self.loss_count += 1
            if symbol:
                self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
                if self.consecutive_losses[symbol] >= 3:
                    logger.critical(f"🛑 [KILL-SWITCH L1] {symbol} accumulated 3 consecutive losses! 1-Hour Soft-Lock.")
                    # Soft Lock for 1 hour to prevent bleeding on a single asset
                    cooldown_manager.set_cooldown(symbol, 3600, "Kill-Switch Level 1: 3 Losses Streak")
                    self.consecutive_losses[symbol] = 0
            
        self.cvar_calc.update(pnl_pct)
        
        # Update Metal-Core Cache
        self._trade_cache.append({
            'is_win': is_win,
            'pnl_pct': pnl_pct,
            'symbol': symbol
        })
        
        # Optional: Limit cache growth to last 1000 trades for performance
        if len(self._trade_cache) > 1000:
            self._trade_cache.pop(0)
        
        # ⚡ PHASE OMNI: TICK-LEVEL KELLY RECALCULATION
        # Uses a rolling window of last 50 trades for responsive sizing
        _KELLY_WINDOW = 50
        trade_list = list(self._trade_cache)
        recent = trade_list[-_KELLY_WINDOW:]
        
        if len(recent) >= 10:  # Minimum sample size for statistical validity
            # [NANO-SPEED] Use compiled kernel for stats
            pnl_arr = np.array([t['pnl_pct'] for t in recent], dtype=np.float64)
            is_win_arr = np.array([t['is_win'] for t in recent], dtype=np.bool_)
            p, b = extract_kelly_stats_jit(pnl_arr, is_win_arr)
            
            # Decimal Kelly Math Evaluation
            raw_kelly = self._compute_kelly_math(p, b, apply_mult=False)
            
            # Half-Kelly with regime-aware scaling
            kelly_mult = 0.5
            if self.stress_score < 90:
                kelly_mult = 0.25  # Quarter-Kelly under stress
            
            tick_kelly = float(max(0.05, min(raw_kelly * kelly_mult, 0.40)))
            
            # EMA smoothing to prevent whipsaw (alpha=0.2)
            if not hasattr(self, '_tick_kelly'):
                self._tick_kelly = tick_kelly
            else:
                self._tick_kelly = 0.2 * tick_kelly + 0.8 * self._tick_kelly
            
            logger.debug(f"⚡ [Kelly/Axioma] Tick Update: p={p:.3f} b={b:.3f} raw={raw_kelly:.3f} → {self._tick_kelly:.3f}")

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
            return 0.03 # 3% base risk for micro accounts ($13 * 3% = $0.39 risk -> completely viable for scalping)
            
        initial = safe_leverage_calculator.initial_capital
        peak = safe_leverage_calculator.peak_capital
        
        # 1. Base Logic (Drawdown Protection - Tightened for HFT)
        risk_pct = 0.01  # Default 1%
        if peak > 0:
            dd = (peak - capital) / peak
            if dd > 0.025: risk_pct = 0.002 # 0.2% (Deep defense)
            elif dd > 0.015: risk_pct = 0.005 # 0.5% (Early defense)

        # 2. Profit Lock Milestones (Wealth Preservation)
        # "Si cuenta +50% sobre HWM" (interpretado como Growth sobre Initial)
        if peak >= (initial * 2.0): # +100% Growth
            risk_pct *= 0.10 # Reduce to 10% of standard (0.1% risk)
            # Logic: "Account doubled. Don't blow it."
        elif peak >= (initial * 1.5): # +50% Growth
            risk_pct *= 0.25 # Reduce to 25% of standard (0.25% risk)
            
        # 3. Protected Capital Floor (The Ratchet)
        profit = peak - initial
        if profit > 0:
            # Lock 80% of ATH profits
            protected_capital = initial + (profit * 0.80)
            
            # Calculate Max Loss Allowed for this trade
            max_loss_allowed = capital - protected_capital
            
            if max_loss_allowed <= 0:
                print(f"🛑 PROTECTED CAPITAL REACHED (${protected_capital:.2f}). Trading Halted.")
                return 0.0
            
            # Clamp risk amount
            current_risk_amt = capital * risk_pct
            if current_risk_amt > max_loss_allowed:
                print(f"🛡️ RATCHET: Clamping risk ${current_risk_amt:.2f} -> ${max_loss_allowed:.2f}")
                risk_pct = max_loss_allowed / capital
                
        return risk_pct

    def _update_stress_metrics(self):
        """Phase 56: Use in-memory cache instead of CSV for PoR."""
        import time
        now = time.time()
        if now - self.last_stress_check < self.stress_check_interval:
            return

        if not self._cache_initialized: self._initialize_cache()
        
        try:
            pnl_returns = [t['pnl_pct'] for t in self._trade_cache]
            
            if len(pnl_returns) >= 20:
                paths = StatisticsPro.generate_monte_carlo_paths(pnl_returns, n_sims=500) # Reduced sims for speed
                metrics = StatisticsPro.calculate_stress_metrics(paths)
                self.stress_score = metrics.get('stress_score', 100.0)
            
            self.last_stress_check = now
        except Exception as e:
            pass

    def _calculate_dynamic_stop_loss(self, atr_pct: float, horizon: str = 'SCALPING') -> float:
        """
        Calcula SL dinámico basado en régimen de volatilidad.
        SUPREMO-V4: ADAPTABILIDAD en mercados CHOPPY.
        """
        horizon_str = getattr(Config.Strategies, 'ACTIVE_HORIZON', '1D')
        horizon_days = int(horizon_str.replace('D', '')) if 'D' in horizon_str else 1
        h_sqrt = math.sqrt(horizon_days)

        # 1. Base Multiplier por Volatilidad
        # SUPREMO-V4: Cambio de SL fijo a dinámico (1.5x ATR)
        # POR QUÉ: Si el precio se mueve 1.5x su ATR en contra, la tesis del trade murió para micro-cuentas.
        # Un SL de 3.0x es demasiado suelto y genera pérdidas de -5%.
        mult = 1.5 # Reducido radicalmente de 3.0 para cuenta de $13
            
        # 2. Ajuste por Régimen de Mercado (ADAPTABILIDAD)
        # En CHOPPY, APRETAMOS el stop (no lo ampliamos) porque no queremos regalar capital al ruido.
        # En TRENDING, podemos ajustarlo más para proteger ganancias.
        if self.current_regime == 'CHOPPY':
            mult *= 0.90
            logger.debug(f"🛡️ [ADAPTIVE SL] CHOPPY regime detected. TIGHTENING SL multiplier to {mult:.2f} (Capital Protection)")
        elif self.current_regime == 'TRENDING':
            mult *= 0.85
            
        # 3. AEGIS-ULTRA: MAE-Based Stop Optimization
        # If we have trade history, check average MAE (Max Adverse Excursion)
        if hasattr(self, '_trade_cache') and len(self._trade_cache) > 20:
            winning_maes = [t.get('max_adverse_excursion', 0) for t in self._trade_cache if t['is_win']]
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
        if not hasattr(self, 'capital_history'):
            import collections
            import time
            self.capital_history = collections.deque(maxlen=100)
            self._last_capital_track = 0.0

        import time
        now = time.time()
        
        # Sample equity at most every 60 seconds (simulating 1-minute bars)
        if now - getattr(self, '_last_capital_track', 0) > 60:
            self.capital_history.append({'value': capital, 'ts': now})
            self._last_capital_track = now
            
        multiplier = 1.0
        
        # Evaluate drops
        if len(self.capital_history) >= 30: # at least 30 minutes of data
            point_30m_ago = self.capital_history[-min(30, len(self.capital_history))]
            drop_30m = (point_30m_ago['value'] - capital) / point_30m_ago['value']
            
            if drop_30m > 0.005:  # >0.5% drop
                logger.warning(f"📉 [VELOCITY] Fast 30m Drop: {drop_30m*100:.2f}%. Halving size.")
                multiplier *= 0.5
                
        if len(self.capital_history) >= 60: # at least 60 minutes
            point_60m_ago = self.capital_history[-60]
            drop_60m = (point_60m_ago['value'] - capital) / point_60m_ago['value']
            
            if drop_60m > 0.01:   # >1.0% drop
                logger.warning(f"🚨 [VELOCITY DEFENSE] 60m Drop: {drop_60m*100:.2f}%. Defensive Mode.")
                multiplier *= 0.5 # Aggregate 0.5 * 0.5 = 0.25 (Quarter-size)

        return multiplier

    # ============================================================
    # POSITION SIZING (FIXED)
    # ============================================================

    @trace_execution
    def size_position(self, signal_event, current_price):
        """FIXED: Position sizing for micro accounts ($12)"""
        if self.portfolio:
            capital = self.portfolio.get_total_equity()
        else:
            capital = safe_leverage_calculator.get_capital()
            
        # VIRTUAL CAPITAL CAP: If capital is huge (Testnet default), cap to $15 for sizing
        # to respect the user's $15 micro-account strategy during testing.
        # [SS-007 FIX] Removed duplicate testnet cap block
        # FORENSIC FIX #6: Only cap for TESTNET (not Demo Trading).
        # POR QUÉ: Demo Trading tiene capital virtual realista, no necesita cap.
        #   Testnet sí necesita cap porque da millones de fake balance.
        if capital > 100 and Config.BINANCE_USE_TESTNET and not getattr(Config, 'BINANCE_USE_DEMO', False):
            logger.info(f"🧪 TESTNET: Simulating $15 account (Actual: ${capital:.2f})")
            capital = 15.0

        # Phase 6: Equal Weighting for Fair Competition (Demo Only)
        if Config.BINANCE_USE_DEMO and getattr(Config.Sniper, 'PERMISSIVE_MODE', False):
            # Bypass Kelly/Growth logic to test pure signal quality
            fixed_pct = getattr(Config.Strategies, 'DEMO_EQUAL_WEIGHTING', 0.05)
            logger.debug(f"🧪 LAB MODE: Using Fixed Equal Weighting ({fixed_pct*100}%) for comparison.")
            return capital * fixed_pct

        phase = safe_leverage_calculator.get_phase(capital)
        
        # FORENSIC FIX #11: Was 0.95 (95%) — suicidal.
        # POR QUÉ: Con MAX_CONCURRENT_POSITIONS=3 y base_pct=0.95, el sistema intentaría
        #   asignar 285% del capital. Además, una sola pérdida de SL 0.15% sobre 95% del
        #   capital = 0.14% de equity, pero con leverage 10x el riesgo real escala.
        # FORENSIC-V11 FIX: Dynamic sizing based on MAX_CONCURRENT_POSITIONS
        # ANTES: 40% × 3 positions = 120% → 3rd position ALWAYS fails reserve_cash
        # AHORA: 90% / MAX_POS → con 3: 30%, con 2: 45% → total never exceeds 90%
        # PARA QUÉ: 30% per trade × 10x leverage = $39 notional (still > $5 min)
        if capital < 50:
            max_pos = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 3)
            base_pct = max(0.25, 0.90 / max_pos)  # 30% with 3 pos, 45% with 2 pos
        elif "GROWTH" in phase:
            base_pct = self.POSITION_PCT_GROWTH  # 30%
        elif capital < 1000:
            # Phase 14: Use Portfolio's global Kelly tracker
            if self.portfolio:
                wr, pr = self.portfolio.get_kelly_metrics()
                kelly_frac = self._compute_kelly_math(wr, pr)
            else:
                strat_id = getattr(signal_event, 'strategy_id', None)
                kelly_frac = self.calculate_kelly_fraction(strategy_id=strat_id)
            base_pct = max(0.20, kelly_frac)
        else:
            if self.portfolio:
                wr, pr = self.portfolio.get_kelly_metrics()
                kelly_frac = self._compute_kelly_math(wr, pr)
            else:
                strat_id = getattr(signal_event, 'strategy_id', None)
                kelly_frac = self.calculate_kelly_fraction(strategy_id=strat_id)
            base_pct = kelly_frac
            
        target_exposure = capital * base_pct
        
        # --- PHOENIX FIX: SIMPLIFIED REGIME ALLOCATION ---
        # Previous logic zeroed out ML in RANGING (60% of time) and TECHNICAL in CHOPPY.
        # For $13 micro-capital, we cannot afford to kill strategies entirely.
        # Conservative adjustment: never go below 0.6x or above 1.3x
        if hasattr(signal_event, 'strategy_id'):
            st_id = str(signal_event.strategy_id).upper()
            st_regime = "UNKNOWN"
            if hasattr(self.portfolio, 'global_regime_data') and self.portfolio.global_regime_data:
                st_regime = self.portfolio.global_regime_data.get('sentiment', 'UNKNOWN')
                
            alloc_mult = 1.0
            if 'BULL' in st_regime:
                if 'TECHNICAL' in st_id: alloc_mult = 1.2
            elif 'BEAR' in st_regime:
                if 'ML' in st_id or 'XGBOOST' in st_id: alloc_mult = 0.7
            elif 'CHOPPY' in st_regime:
                if 'TECHNICAL' in st_id: alloc_mult = 0.6  # Reduce, don't kill
                
            if alloc_mult != 1.0:
                logger.debug(f"🔄 [ADAPTIVE ALLOC] Regime: {st_regime} | Strat: {st_id} | Mult: {alloc_mult:.2f}x")
                target_exposure *= alloc_mult
                
        # --- PHASE 5: RELATIVE STRENGTH ALLOCATION (Capital Rotation) ---
        if self.portfolio and hasattr(self.portfolio, 'update_relative_strength'):
            self.portfolio.update_relative_strength() # Lazy update (cached)
            if hasattr(signal_event, 'signal_type'):
                is_long = signal_event.signal_type == SignalType.LONG
                rs_mult = self.portfolio.get_allocation_multiplier(signal_event.symbol, is_long)
                if rs_mult != 1.0:
                    logger.info(f"🔄 [REL STRENGTH] {signal_event.symbol} sizing adjusted {rs_mult:.2f}x.")
                    target_exposure *= rs_mult
        
        # ATR-based sizing (VOLATILITY ADJUSTED)
        # Size = (Capital * Risk%) / SL_Distance
        if hasattr(signal_event, 'atr') and signal_event.atr is not None and signal_event.atr > 0:
            current_risk_pct = self._get_dynamic_risk_per_trade(capital)
            risk_amount = capital * current_risk_pct
            
            # Estimate SL distance for sizing (use dynamic logic)
            atr_pct = (signal_event.atr / current_price) if current_price and current_price > 0 else 0.02
            horizon_val = getattr(signal_event, 'horizon', 'SCALPING')
            est_sl_pct = self._calculate_dynamic_stop_loss(atr_pct, horizon=horizon_val)
            
            # Formula: Risk = Size * SL_Pct  =>  Size = Risk / SL_Pct
            if est_sl_pct and est_sl_pct > 0:
                vol_adjusted_size = risk_amount / est_sl_pct
            else:
                vol_adjusted_size = risk_amount / 0.01 # Fallback 1%
            
            # VOLATILITY WEIGHTED SIZING
            # Assets like SOL/DOGE get lower size multiplier than BTC/ETH
            vol_multiplier = 1.0
            if "SOL" in signal_event.symbol or "DOGE" in signal_event.symbol:
                vol_multiplier = 0.75 # Use 25% less exposure for volatile memes
            
            logger.info(f"⚖️ Sizing: Risk={current_risk_pct*100}% (${risk_amount:.2f}) | SL={est_sl_pct*100:.2f}% | Size=${vol_adjusted_size:.2f} (VolMult: {vol_multiplier})")
            target_exposure = min(target_exposure, vol_adjusted_size) * vol_multiplier
        
        # [EXECUTION AUDIT] FIX F-010: Strength scaling for sizing
        # For micro accounts (<$50), ignore strength to prevent dropping below notional minimums.
        # For larger accounts, apply strength as conviction scaling (capped at 1.2x).
        if hasattr(signal_event, 'strength'):
            if capital >= 50:
                target_exposure *= min(signal_event.strength, 1.2)
            
        # FORENSIC FIX: Removed dangerous static $6 TARGET_EXPOSURE padding.
        # target_exposure is MARGIN, not NOTIONAL. Clamping margin to $6 on a $13 account
        # forces extreme risk. The $5 USD Binance Notional limit is properly calculated
        # post-leverage in _calculate_order_params.

        # AEGIS-ULTRA: EXPECTED VALUE VETO (Phase 14)
        # If expected value is strictly negative (Kelly <= 0), Veto trade
        if self.portfolio and capital >= 50:
            wr, pr = self.portfolio.get_kelly_metrics()
            kelly_frac = self._compute_kelly_math(wr, pr, apply_mult=False)
            if kelly_frac <= 0 and wr > 0: # Only if we have some history
                logger.warning(f"🛑 [KELLY VETO] EV is Negative. WinRate: {wr:.2f}, Payoff: {pr:.2f}, Kelly: {kelly_frac:.2f}. Blocking {signal_event.symbol}")
                return 0.0

        # AEGIS-ULTRA: CONTAGION PROTOCOL (Phase 15)
        # If Fleet Correlation > 0.85, reduce risk by 50%
        if hasattr(self.portfolio, 'global_regime_data'):
            breadth = self.portfolio.global_regime_data
            if breadth.get('contagion_risk', False):
                 target_exposure *= 0.5
                 logger.warning(f"☢️ [AEGIS] Contagion Protocol Active (Corr > 0.85). Sizing halved.")
            
        # STRESS TEST ADJUSTMENT (Phase 6)
        # If Stress Score < 95 (PoR > 5%), reduce sizing proportionally
        # Example: Score 80 -> Mult 0.8
        self._update_stress_metrics() # Lazy update check
        if self.stress_score < 95:
             stress_mult = self.stress_score / 100.0
             logger.info(f"📉 Ruin Risk Protection: Scaling size by {stress_mult:.2f}x (Score: {self.stress_score})")
             target_exposure *= stress_mult
        
        # 4. Contextual Sizing (World Awareness Adaptive Filter)
        # PROFESSOR METHOD: Reduced exposure in thin liquidity to prevent slippage.
        context = world_awareness.get_market_context()
        ls = context.get('liquidity_score', 0.8)
        
        ls_mult = 1.0
        if ls <= 0.45: ls_mult = 0.5   # 50% red. in Dead Zone
        elif ls <= 0.65: ls_mult = 0.75 # 25% red. in Low sessions
            
        if ls_mult < 1.0:
            logger.info(f"🌍 Session Risk Adapter: Scaling size by {ls_mult:.2f}x (LS: {ls:.2f})")
            target_exposure *= ls_mult
        
        # FIXED: Update capital tracking before checking CVaR
        self._update_capital_tracking(capital)
        
        # PHASE 12: Asymmetric Sizing based on Fall Velocity
        velocity_mult = self._track_drawdown_velocity(capital)
        if velocity_mult < 1.0:
            target_exposure *= velocity_mult
        
        # CVaR reduction (FIXED: Use peak_capital for accurate drawdown)
        current_dd = 1 - (capital / self.peak_capital) if capital < self.peak_capital else 0
        
        # FORENSIC-V9-FIX: CVaR + Recovery are now NON-MULTIPLICATIVE
        # QUÉ: Antes CVaR (×0.5) * Recovery (×0.5) = ×0.25 → mataba sizing en micro-cuentas.
        # POR QUÉ: $13 × 0.40 × 0.25 = $1.30 margen → $10.4 notional (borderline viable).
        # PARA QUÉ: Usar max(CVaR, Recovery) para una sola reducción de 50%, no 75%.
        # CÓMO: Calculamos ambas penalidades pero solo aplicamos la más severa.
        cvar_reduction = 1.0
        recovery_reduction = 1.0
        
        if self.cvar_calc.should_reduce_risk(current_dd):
            cvar_reduction = 0.5
            logger.warning(f"⚠️ CVaR: Risk reduction flagged (DD: {current_dd*100:.1f}%)")
        
        # --- PHASE 14: DYNAMIC RECOVERY STATE ---
        self._update_resolution_state(current_dd)
        if self.resolution_state == ResolutionState.RECOVERY:
            recovery_reduction = 0.5
            logger.warning(f"🛡️ [RECOVERY MODE] Drawdown ({current_dd*100:.1f}%) > threshold.")
        elif self.resolution_state == ResolutionState.GROWTH:
             recovery_reduction = 1.2  # Boost if profitable
             logger.info(f"🚀 [GROWTH MODE] Account flying high. Boost enabled.")
        
        # Apply the SINGLE worst reduction (not both)
        worst_reduction = min(cvar_reduction, recovery_reduction)
        if worst_reduction != 1.0:
            target_exposure *= worst_reduction
            logger.info(f"⚖️ [V9-SIZING] Applied single worst reduction: {worst_reduction:.2f}x (CVaR={cvar_reduction}, Recovery={recovery_reduction})")
        
        # --- PHASE 14: ML CONFIDENCE SCALING ---
        if hasattr(signal_event, 'strength'):
            strength = signal_event.strength
            if strength >= 0.75:
                # High confidence boost (Max 1.5x)
                # Linear scale: 0.75->1.0x, 1.0->1.5x
                boost = 1.0 + ((strength - 0.75) * 2.0)
                boost = min(boost, 1.5)
                target_exposure *= boost
                logger.info(f"🧠 ML Confidence Boost: {boost:.2f}x (Strength: {strength:.2f})")
            elif strength < 0.6 and capital > 100: # Only penalize if not micro
                # Low confidence penalty
                target_exposure *= 0.5
                logger.info(f"🧠 Low Confidence Penalty: 0.5x (Strength: {strength:.2f})")

        # FORENSIC-V9-FIX: MINIMUM SIZING FLOOR FOR MICRO-ACCOUNTS
        # QUÉ: Después de ~15 multiplicadores en cascada, target_exposure puede
        #   caer a $0.50 (insuficiente para cualquier trade viable).
        # POR QUÉ: La cascada multiplicativa de 15 factores × <1.0 cada uno
        #   colapsa exponencialmente: 0.8^15 = 0.035, convirtiendo $5.20 en $0.18.
        # PARA QUÉ: Garantizar que el sizing NUNCA caiga por debajo del mínimo
        #   necesario para superar el notional de $5 USD de Binance.
        # CÓMO: Floor = ($5 / leverage_estimado) × 1.10 (margen de seguridad 10%).
        if capital < 50:
            leverage_estimate = 8  # Micro-account minimum leverage
            min_viable_margin = (5.0 / leverage_estimate) * 1.10  # ~$0.6875
            if target_exposure < min_viable_margin:
                logger.warning(
                    f"🔒 [V9-FLOOR] Sizing cascade collapsed target to ${target_exposure:.2f}. "
                    f"Restoring to minimum viable: ${min_viable_margin:.2f}"
                )
                target_exposure = min_viable_margin

        return target_exposure
    
    def _update_resolution_state(self, current_dd: float):
        """Phase 14: State Machine for Risk Appetite"""
        if current_dd > self.recovery_threshold:
            self.resolution_state = ResolutionState.RECOVERY
        elif current_dd < (self.recovery_threshold * 0.5) and self.resolution_state == ResolutionState.RECOVERY:
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
        if not self._cache_initialized: self._initialize_cache()
        
        try:
            sym_trades = [t for t in self._trade_cache if t['symbol'] == symbol]
            if len(sym_trades) < 10: return True # Learning mode
                
            wins = sum(1 for t in sym_trades if t['is_win'])
            total = len(sym_trades)
            avg_win = np.mean([t['pnl_pct'] for t in sym_trades if t['is_win']]) if wins > 0 else 0
            avg_loss = np.mean([abs(t['pnl_pct']) for t in sym_trades if not t['is_win']]) if (total - wins) > 0 else 0
            
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
        🛡️ SUPREMO-V3: ATOMIC ORDER GENERATION PIPELINE
        QUÉ: Transforma señales en órdenes válidas tras pasar 7 filtros de seguridad.
        POR QUÉ: Garantiza que ninguna orden "tóxica" llegue al exchange.
        """
        # 1. EMERGENCY BYPASS (Rule 2.1) - EXIT Signals ignore everything
        if self._validate_emergency_bypass(signal_event):
            horizon = getattr(signal_event, 'horizon', 'SCALPING')
            logger.info(f"🚨 [BYPASS] Exit signal for {signal_event.symbol} bypassing safety gates.")
            
            # Fetch position to know exactly what to close
            pos = self.portfolio.get_horizon_position(signal_event.symbol, horizon) if self.portfolio else None
            
            if not pos or abs(pos.get('quantity', 0)) < 1e-8:
                logger.warning(f"⚠️ Exit signal for {signal_event.symbol} ignored: No open position in {horizon}")
                return None
            
            qty = pos['quantity']
            direction = OrderSide.BUY if qty < 0 else OrderSide.SELL
            
            # ═══════════════════════════════════════════════════════════════
            # BBO ARCHITECTURE: "MAKER PROFIT, TAKER PANIC"
            # QUÉ: Exits normales (TP, trailing stop) → LIMIT BBO (Maker 0.02%)
            #   Emergencias (Kill Switch, priority=0) → MARKET (Taker 0.0375%)
            # POR QUÉ: Taker fees desangran $13 micro-cuentas (~0.075%/trip)
            # PARA QUÉ: Ahorrar ~47% en fees = ~4% más capital en 15 días
            # CÓMO: Si priority > 0 (exit normal), usar LIMIT con metadata
            #   para que executor ponga precio en BBO y OrderManager chase.
            #   Si priority == 0 (kill switch), MARKET nuclear.
            # ═══════════════════════════════════════════════════════════════
            exit_priority = getattr(signal_event, 'priority', 1)
            strategy_id = getattr(signal_event, 'strategy_id', 'EXIT')
            is_kill_switch = (exit_priority == 0 or strategy_id == 'EMERGENCY_EXIT' 
                              or strategy_id == 'KILL_SWITCH')
            
            use_limit_exits = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'USE_LIMIT_BBO_EXITS', True)
            
            if is_kill_switch or not use_limit_exits:
                # 🔴 TAKER PANIC: Emergency exit → MARKET nuclear
                exit_order_type = OrderType.MARKET
                exit_metadata = {'exit_mode': 'TAKER_PANIC'}
                logger.warning(f"🔴 [BBO] EMERGENCY EXIT {signal_event.symbol}: MARKET (Taker Panic)")
            else:
                # 🟢 MAKER PROFIT: Normal exit → LIMIT BBO (Post-Only)
                exit_order_type = OrderType.LIMIT
                use_gtx = getattr(Config.Execution, 'POST_ONLY_GTX', True)
                exit_ttl = getattr(Config.Execution, 'CHASE_TIMEOUT_SECONDS', 5)
                exit_metadata = {
                    'timeInForce': 'GTX' if use_gtx else 'GTC',
                    'is_bbo_exit': True,
                    'is_exit': True,
                    'exit_mode': 'MAKER_PROFIT',
                }
                logger.info(f"🟢 [BBO] LIMIT EXIT {signal_event.symbol}: Maker Profit mode (GTX)")
            
            return OrderEvent(
                symbol=signal_event.symbol,
                order_type=exit_order_type,
                quantity=abs(qty),
                direction=direction,
                price=current_price,
                horizon=horizon,
                priority=exit_priority,
                is_exit=True,
                is_close=True,
                ttl=getattr(Config.Execution, 'CHASE_TIMEOUT_SECONDS', 5) if exit_order_type == OrderType.LIMIT else None,
                ml_confidence=getattr(signal_event, 'ml_confidence', None),
                predicted_duration=getattr(signal_event, 'predicted_duration', None),
                metadata=exit_metadata,
            )

        # 2. ATOMIC VALIDATIONS (Sequencial & Fast)
        if not self._validate_kill_switch(): 
            print("[RISK] Rejected by Kill Switch")
            return None
        if not self._validate_frequency_limits(signal_event.symbol, signal_event.signal_type): 
            print("[RISK] Rejected by Frequency Limits")
            return None
        if not self._validate_regime_veto(signal_event.symbol, signal_event.signal_type): 
            print("[RISK] Rejected by Regime Veto")
            return None
        
        # Spot Mode Safety: SHORT is ONLY for Futures
        if not getattr(Config, 'BINANCE_USE_FUTURES', False) and signal_event.signal_type == SignalType.SHORT:
            logger.warning(f"🛡️ [SpotSafety] SHORT rejected for {signal_event.symbol} (Futures Mode is OFF).")
            print("[RISK] Rejected by SpotSafety")
            return None
            
        horizon = getattr(signal_event, 'horizon', 'SCALPING')
        if not self._validate_directional_safety(signal_event.symbol, signal_event.signal_type, horizon): 
            print("[RISK] Rejected by Directional Safety")
            return None
        if not self._validate_margin_ratio(): 
            print("[RISK] Rejected by Margin Ratio")
            return None
        
        # 2.5 FAT FINGER PROTECTION (Dept C Audit Requirement)
        if not self._validate_fat_finger(current_price, signal_event.symbol): 
            print("[RISK] Rejected by Fat Finger")
            return None
        
        # 2.6 LIQUIDITY AWARENESS (Sovereign-Deploy Slippage)
        if not self._validate_slippage(signal_event.symbol, current_price): 
            print("[RISK] Rejected by Slippage")
            return None
        
        # 3. MAX POSITIONS CHECK (Virtual Ledger Aware)
        symbol = signal_event.symbol
        horizon = getattr(signal_event, 'horizon', 'SCALPING')
        if self.portfolio:
            # SUPREMO-V4: CANNIBALIZATION AUDIT
            v_key_opp = f"{symbol}_{'SWING' if horizon == 'SCALPING' else 'SCALPING'}"
            opp_pos = self.portfolio.virtual_ledger.get(v_key_opp)
            if opp_pos and opp_pos.get('quantity', 0) != 0:
                is_opp_long = opp_pos['quantity'] > 0
                is_new_short = signal_event.signal_type == SignalType.SHORT
                is_new_long = signal_event.signal_type == SignalType.LONG
                
                if (is_opp_long and is_new_short) or (not is_opp_long and is_new_long):
                    logger.info(f"🔄 [CANNIBAL-GUARD] Opposite position in {v_key_opp} detected. Allowing signal as Net-Exposure Reducer.")
            
            open_positions = sum(1 for pos in self.portfolio.virtual_ledger.values() if pos.get('quantity', 0) != 0)
            if open_positions >= Config.MAX_CONCURRENT_POSITIONS and signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
                if not self.portfolio.has_position_for_horizon(symbol, horizon):
                    print(f"[RISK] Rejected by Max Positions (Open={open_positions}, Max={Config.MAX_CONCURRENT_POSITIONS})")
                    return None

        # 4. ORDER CALCULATION (Isolated Logic)
        try:
            params = self._calculate_order_params(signal_event, current_price)
            if not params: 
                print("[RISK] Rejected by _calculate_order_params")
                return None
            
            # 5. FINAL MARGIN RESERVATION
            if self.portfolio and not self.portfolio.reserve_cash(params['dollar_size'], horizon=horizon):
                logger.warning(f"⚠️ Reserve failed for {symbol} on {horizon}")
                print("[RISK] Rejected by reserve_cash")
                return None

            # 6. EXECUTION & LOGGING
            cooldown_manager.record_trade(symbol, strategy_id="RISK_MANAGER")
            self.global_trade_count += 1
            
            # ═══════════════════════════════════════════════════════════════
            # BBO ARCHITECTURE: SMART ENTRY ORDER SELECTION (LIMIT vs MARKET)
            # QUÉ: Entries usan LIMIT BBO por defecto (Maker 0.02%).
            # POR QUÉ: MARKET entries pagan Taker 0.0375% — casi el doble.
            # PARA QUÉ: Reducir el costo de apertura en ~47%.
            # CÓMO: Solo priority==0 (kill-switch-level urgency) usa MARKET.
            #   strength > 0.92 mantiene LIMIT pero con pricing agresivo.
            # CUÁNDO: En cada entrada nueva.
            # DÓNDE: risk/risk_manager.py :: generate_order()
            # QUIÉN: Risk Manager + Executor
            # ═══════════════════════════════════════════════════════════════
            strength = getattr(signal_event, 'strength', 0)
            priority = getattr(signal_event, 'priority', 1)
            use_limit_entries = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'USE_LIMIT_BBO_ENTRIES', True)
            
            if priority == 0:
                # Kill-switch-level urgency ONLY → MARKET
                order_type = OrderType.MARKET
                entry_metadata = {'strength': strength, 'entry_mode': 'TAKER_PANIC'}
                logger.info(f"⚡ [BBO] MARKET ENTRY {symbol}: Kill-switch priority.")
            elif use_limit_entries:
                # 🟢 MAKER PROFIT: LIMIT BBO with Post-Only
                order_type = OrderType.LIMIT
                use_gtx = getattr(Config.Execution, 'POST_ONLY_GTX', True)
                entry_ttl = getattr(Config.Execution, 'ENTRY_TTL_SECONDS', 30)
                entry_metadata = {
                    'strength': strength,
                    'entry_mode': 'MAKER_PROFIT',
                    'timeInForce': 'GTX' if use_gtx else 'GTC',
                }
                logger.info(f"🟢 [BBO] LIMIT ENTRY {symbol}: Maker Profit mode (strength={strength:.2f})")
            else:
                # Fallback: original behavior
                order_type = OrderType.LIMIT
                entry_metadata = {'strength': strength, 'entry_mode': 'LEGACY'}
                if strength > 0.92:
                    order_type = OrderType.LIMIT
                    entry_metadata['entry_mode'] = 'BBO_AGGRESSIVE'
                    entry_metadata['timeInForce'] = 'GTC' # No GTX so it can cross spread if needed
                    logger.info(f"⚡ [SMART-ORDER] High conviction signal ({strength*100:.1f}%) for {symbol}. Using AGGRESSIVE LIMIT (BBO).")
            
            # SUPREMO-V4: SHADOW MODE (MICRO-ACCOUNT PROTECTION)
            # If capital is critical, simulate Swing trades to preserve margin for Scalping
            equity = self.portfolio.get_total_equity() if self.portfolio else 13.0
            is_shadow_final = False
            
            entry_metadata['dollar_size'] = params['dollar_size']

            # FORENSIC FIX: Lowered shadow threshold from 13.0 to 8.0 to re-enable SWING.
            if equity < 8.0 and horizon == 'SWING':
                is_shadow_final = True
                logger.info(f"🧬 [SHADOW-MODE] Capital ${equity:.2f} is critical. Simulating SWING trade.")

            return OrderEvent(
                symbol=symbol,
                order_type=order_type,
                quantity=params['quantity'],
                direction=params['direction'],
                strategy_id=getattr(signal_event, 'strategy_id', None),
                sl_pct=params['sl_pct'],
                tp_pct=params['tp_pct'],
                price=current_price,
                ttl=getattr(Config.Execution, 'ENTRY_TTL_SECONDS', 30) if order_type == OrderType.LIMIT else None,
                horizon=horizon,
                priority=priority,
                is_shadow=is_shadow_final,
                ml_confidence=getattr(signal_event, 'ml_confidence', None),
                predicted_duration=getattr(signal_event, 'predicted_duration', None),
                metadata=entry_metadata
            )
        except Exception as e:
            logger.error(f"Order Generation Failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def _generate_exit_order(self, signal_event, current_price):
        """
        BBO ARCHITECTURE: Exit Order Generator
        QUÉ: Genera orden de cierre con tipo LIMIT BBO o MARKET según urgencia.
        POR QUÉ: Exits normales pueden esperar BBO fill → ahorro 47% en fees.
        PARA QUÉ: Maximizar retención de capital en micro-cuenta.
        CÓMO: strategy_id 'EMERGENCY_EXIT'/'KILL_SWITCH' → MARKET. Otros → LIMIT BBO.
        """
        if not self.portfolio: return None
            
        horizon = getattr(signal_event, 'horizon', 'SCALPING')
        pos = self.portfolio.get_horizon_position(signal_event.symbol, horizon)
        
        if not pos: 
            return None
            
        qty = pos['quantity']
        if qty == 0: return None
        
        # BBO Decision: Emergency or Normal exit?
        strategy_id = getattr(signal_event, 'strategy_id', 'EXIT')
        exit_priority = getattr(signal_event, 'priority', 1)
        is_emergency = (strategy_id in ('EMERGENCY_EXIT', 'KILL_SWITCH') or exit_priority == 0)
        use_limit_exits = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'USE_LIMIT_BBO_EXITS', True)
        
        if is_emergency or not use_limit_exits:
            exit_type = OrderType.MARKET
            exit_metadata = {'exit_mode': 'TAKER_PANIC', 'is_exit': True}
            logger.warning(f"🔴 [BBO] EMERGENCY EXIT {signal_event.symbol}/{horizon}: MARKET")
        else:
            exit_type = OrderType.LIMIT
            use_gtx = getattr(Config.Execution, 'POST_ONLY_GTX', True)
            exit_metadata = {
                'timeInForce': 'GTX' if use_gtx else 'GTC',
                'is_bbo_exit': True,
                'is_exit': True,
                'exit_mode': 'MAKER_PROFIT',
            }
            logger.info(f"🟢 [BBO] LIMIT EXIT {signal_event.symbol}/{horizon}: Maker Profit")
        
        return OrderEvent(
            symbol=signal_event.symbol,
            order_type=exit_type,
            quantity=abs(qty),
            direction=OrderSide.SELL if qty > 0 else OrderSide.BUY,
            strategy_id=strategy_id,
            price=current_price,
            horizon=horizon,
            priority=exit_priority,
            is_exit=True,
            is_close=True,
            ttl=getattr(Config.Execution, 'CHASE_TIMEOUT_SECONDS', 5) if exit_type == OrderType.LIMIT else None,
            metadata=exit_metadata,
        )

    def _calculate_order_params(self, signal_event, current_price):
        """Aislamiento de la lógica de sizing y apalancamiento."""
        margin_size = self.size_position(signal_event, current_price)
        
        raw_atr = getattr(signal_event, 'atr', None)
        atr_val = raw_atr if raw_atr is not None else (current_price * 0.02 if current_price else 0.0)
        safe_calc = safe_leverage_calculator.calculate_safe_leverage(atr_val, current_price)
        leverage = safe_calc['leverage']
        
        if not safe_calc['is_safe']: return None
        
        # Leverage adjustments (Dynamic Adaptation)
        regime = self.current_regime
        # Get advice from Intelligence Layer (if valid)
        try:
             from core.market_regime import MarketRegimeDetector
             # Use the detector logic directly or from a text map
             # For speed, we use a local interpretation of the REGIME_MAP
             from config import Config
             regime_map = getattr(Config.Sniper, 'REGIME_MAP', {})
             params = regime_map.get(regime, {})
             regime_leverage_limit = params.get('leverage', 1)
             
             # CLAMP: Leverage cannot exceed Regime Limit
             leverage = min(leverage, regime_leverage_limit)
             
             # Also respect Config Max
             leverage = min(leverage, getattr(Config.Sniper, 'MAX_LEVERAGE', 10))
             
             # MICRO-ACCOUNT BOOST (Only if Regime allows it)
             capital = self.portfolio.get_total_equity() if self.portfolio else 15.0
             if capital < 20 and regime_leverage_limit >= 5:
                  leverage = max(leverage, 8)
                  
        except Exception:
             # FORENSIC-V9-FIX: Changed fallback from 1x to 8x
             # QUÉ: leverage=1 con $13 producía notional=$5.20 (borderline).
             # POR QUÉ: Combinado con multiplicadores de sizing, el notional
             #   caía por debajo del mínimo de $5 → orden rechazada silenciosamente.
             # PARA QUÉ: Garantizar notional viable incluso cuando MarketRegime falla.
             leverage = 8 # Micro-account safe fallback
        
        # FORENSIC FIX #7: MICRO-ACCOUNT LEVERAGE FLOOR
        # POR QUÉ: La REGIME_MAP puede clampar leverage a 1-3x, pero con $13
        #   de margin y 3x leverage, notional = $39. El mínimo de Binance es $5
        #   pero con 3 concurrent positions necesitamos notional suficiente.
        # CÓMO: Para cuentas < $50, forzamos leverage mínimo 8x sin importar régimen.
        #   Esto garantiza $13×0.40×8 = $41.6 notional mínimo por trade.
        capital_check = self.portfolio.get_total_equity() if self.portfolio else 15.0
        if capital_check < 50 and leverage < 8:
            logger.info(f"⚡ [MICRO-FLOOR] Leverage boosted {leverage}x → 8x (micro-account protection)")
            leverage = 8
        
        notional = margin_size * leverage
        
        # 🚀 PHOENIX MICRO-ACCOUNT GUARD
        if notional < 5.10:
            required_margin = (5.10 / leverage)
            horizon = getattr(signal_event, 'horizon', 'SCALPING')
            available_cash = self.portfolio.get_available_cash(horizon=horizon) if self.portfolio else 13.0
            
            if required_margin > available_cash:
                logger.warning(f"⚠️ [MICRO-GUARD] Insufficient capital for {signal_event.symbol}. Need ${required_margin:.2f}, Avail: ${available_cash:.2f}")
                return None
                
            capital = self.portfolio.get_total_equity() if self.portfolio else 13.0
            margin_pct = required_margin / capital if capital > 0 else 0
            if margin_pct > 0.35 and capital < 50:
                logger.info(f"⚡ [MICRO-GUARD] Upsizing margin to {margin_pct*100:.1f}%. Safe due to SL.")
                
            margin_size = required_margin
            notional = margin_size * leverage
            
        # FORENSIC FIX #3: Removed dead fee check (was: fees > notional * 0.015 * 0.45)
        # POR QUÉ: Con Binance fees de 0.0375%, este check NUNCA se activaba.
        #   El check real está en L1309 (Fee-Aware Block) que SÍ funciona.
        # El check duplicado fue eliminado para evitar confusión.
        fees = self.fee_calc.calculate_round_trip_fee(notional, order_type='LIMIT')
            
        # ATR Targets (Fallback)
        atr_pct = atr_val / current_price if current_price > 0 else 0.01
        
        # FORENSIC FIX #4: STANDARDIZED TP/SL CONTRACT
        # All strategies now send TP/SL as decimal fractions (e.g., 0.004 = 0.4%)
        # The old auto-detect heuristic (> 0.5 → /100) was BROKEN:
        #   - technical.py sent 0.4 (intended 0.4%) → heuristic said "< 0.5, use as-is" → 40%
        #   - sniper sent 3.0 (intended 3%) → heuristic said "> 0.5, /100" → 0.03 = 3% ✓ (by luck)
        # Now: direct passthrough, no guessing.
        event_sl = getattr(signal_event, 'sl_pct', None)
        event_tp = getattr(signal_event, 'tp_pct', None)
        if event_sl is not None and event_sl > 0:
            sl_pct = event_sl  # Already decimal fraction (e.g., 0.0015 = 0.15%)
        else:
            sl_pct = self._calculate_dynamic_stop_loss(atr_pct)
            
        if event_tp is not None and event_tp > 0:
            tp_pct = event_tp  # Already decimal fraction (e.g., 0.004 = 0.4%)
        else:
            tp_pct = max(0.015, sl_pct * 2.0)

        # ================================================================
        # FORENSIC REMEDIATION: SYMMETRIC SHORT SL/TP (Fixed Logic)
        # QUÉ: Antes se DIVIDÍA el SL por SHORT_SL_MULTIPLIER, haciéndolo MÁS
        #   estrecho (0.6% / 1.2 = 0.5%). Shorts NECESITAN más espacio porque
        #   squeezes son más violentos que dips.
        # POR QUÉ: Short squeezes pueden mover +1-2% en segundos. Un SL de 0.5%
        #   se come el ruido y dispara Hard SL instantáneamente.
        # PARA QUÉ: Ampliar SL para shorts (0.6% × 1.2 = 0.72%) y reducir TP
        #   para tomar profit más rápido (shorts revierten más rápido).
        # ================================================================
        if signal_event.signal_type == SignalType.SHORT:
            sl_mult = getattr(Config.Strategies, 'SHORT_SL_MULTIPLIER', 1.0)
            tp_mult = getattr(Config.Strategies, 'SHORT_TP_MULTIPLIER', 1.0)
            sl_pct = sl_pct * sl_mult  # WIDER SL for shorts (was: / sl_mult → TIGHTER = bad)
            tp_pct = tp_pct * tp_mult
            logger.debug(f"📉 [SYMMETRIC SHORT] Adjusting for {signal_event.symbol}: SL_mult={sl_mult:.2f}x, TP_mult={tp_mult:.2f}x. New SL={sl_pct*100:.2f}%, TP={tp_pct*100:.2f}%")
        # ================================================================

        # 🛡️ AXIOMATIC FIX: Viability Floor Clamp
        # QUÉ: Anula silenciosamente y empuja hacia arriba cualquier TP inferior a 1.0%
        # POR QUÉ: Costos de Micro-Cuenta (fees+volver breakeven) requieren un TP ≥ 1.0%
        if tp_pct < 0.010:
            logger.info(f"⚡ [AXIOMATIC-CLAMP] Target TP {tp_pct*100:.2f}% forced to Minimum Viability (1.00%)")
            tp_pct = 0.010

        # ═══════════════════════════════════════════════════════════════════
        # REMEDIACIÓN QUIRÚRGICA: MIN_EDGE_NET (Fee + Slippage Viability)
        # QUÉ: Rechaza trades donde la expectativa neta es negativa.
        # POR QUÉ: El check anterior solo verificaba fees × 2.0, pero
        #   NO incluía slippage estimado (~0.03% round-trip). Con $13 de
        #   capital, incluso 0.03% de slippage puede comer la ganancia.
        # PARA QUÉ: GARANTIZAR que TODO trade tiene edge NETO positivo
        #   después de fees + slippage + buffer de seguridad.
        # CÓMO: expected_net = profit_bruto - fees - slippage.
        #   Si expected_net < min_edge_absolute → rechazar.
        # CUÁNDO: Última línea de defensa antes de emitir la orden.
        # DÓNDE: risk/risk_manager.py :: _calculate_order_params()
        # QUIÉN: Risk Manager (este módulo)
        # ═══════════════════════════════════════════════════════════════════
        fees = self.fee_calc.calculate_round_trip_fee(notional, order_type='LIMIT')
        estimated_slippage = notional * 0.0003  # 0.03% round-trip average
        total_costs = fees + estimated_slippage
        expected_profit = notional * tp_pct
        expected_net = expected_profit - total_costs

        # Min edge: profit must cover 2.5x total costs for viability
        min_edge_multiplier = 2.5
        if expected_net < (total_costs * (min_edge_multiplier - 1.0)):
            logger.warning(
                f"📉 [MIN-EDGE-NET] {signal_event.symbol} REJECTED: "
                f"Net=${expected_net:.3f} < MinEdge=${total_costs * (min_edge_multiplier - 1.0):.3f} "
                f"(Profit=${expected_profit:.3f}, Fees=${fees:.4f}, "
                f"Slip=${estimated_slippage:.4f}, TP={tp_pct*100:.2f}%)"
            )
            return None

        
        return {
            'quantity': notional / current_price,
            'direction': OrderSide.BUY if signal_event.signal_type == SignalType.LONG else OrderSide.SELL,
            'leverage': leverage,
            # 🔥 FORENSIC-V12 FIX #3: Usar leverage LOCAL (no Config.BINANCE_LEVERAGE)
            # POR QUÉ: El leverage puede ser dinámico (safe_leverage_calculator,
            #   REGIME_MAP, MICRO-FLOOR). Usar Config era hardcoded → margin leak.
            # PARA QUÉ: dollar_size == margin REAL que Portfolio.pending_cash trackea.
            'dollar_size': notional / leverage if getattr(Config, 'BINANCE_USE_FUTURES', False) else notional,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct
        }

    # ============================================================
    # PHASE 5: INTELLIGENT REVERSE FACADE
    # ============================================================
    
    def analyze_flip_viability(self, symbol, current_pnl_pct, next_signal_strength, atr_pct) -> dict:
        """
        PROFESSOR METHOD:
        QUÉ: Análisis de viabilidad técnico-económica para una reversión.
        POR QUÉ: Flipping tiene costes dobles (comisión salida + comisión entrada + slippage x2).
        CÓMO: Comparamos el Valor Esperado (EV) de la nueva señal vs el hundimiento de costes.
        """
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # 1. Check Daily Limit
        symbol_flips = self.daily_flips.get(symbol, {"date": now_date, "count": 0})
        if symbol_flips["date"] != now_date:
            symbol_flips = {"date": now_date, "count": 0}
            
        if symbol_flips["count"] >= getattr(Config, 'FLIP_MAX_DAILY_COUNT', 3):
            return {"is_viable": False, "reason": f"Daily flip limit reached ({symbol_flips['count']})"}
            
        # 2. Check Cooldown
        last_flip = self.last_flip_times.get(symbol, 0)
        cooldown = getattr(Config, 'FLIP_COOLDOWN_SECONDS', 300)
        if (time.time() - last_flip) < cooldown:
            return {"is_viable": False, "reason": f"Flipping Cooldown active ({int(cooldown - (time.time() - last_flip))}s)"}

        # 3. Volatility Filter
        min_atr = getattr(Config, 'FLIP_MIN_ATR_PCT', 0.005)
        if atr_pct < min_atr:
            return {"is_viable": False, "reason": f"Volatility too low for flip: {atr_pct*100:.2f}% < {min_atr*100:.2f}%"}

        # 4. Cost-Benefit Analysis
        # Cost = Exit Fee (0.05%) + Entry Fee (0.05%) + Slippage Exit (0.05%) + Slippage Entry (0.05%) = ~0.2%
        est_cost = getattr(Config, 'FLIP_COST_THRESHOLD', 0.002)
        
        # Expected Benefit = Expected Move (based on ATR) * Strategy Confidence
        # Scalping target is usually ~1.5 - 2.0x ATR
        potential_move = atr_pct * 1.5 
        expected_benefit = potential_move * next_signal_strength
        
        # Minimum R:R for the Flip (Expected profit must cover at least 2x the cost)
        min_rr = getattr(Config, 'FLIP_MIN_POTENTIAL_RR', 2.0)
        
        if expected_benefit < (est_cost * min_rr):
            return {
                "is_viable": False, 
                "reason": f"Cost hurdle too high (EV: {expected_benefit*100:.2f}% vs Threshold: {est_cost*min_rr*100:.2f}%)"
            }

        return {"is_viable": True, "reason": "Viability check passed"}

    def _record_flip(self, symbol):
        """Update flip counters"""
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if symbol not in self.daily_flips or self.daily_flips[symbol]["date"] != now_date:
            self.daily_flips[symbol] = {"date": now_date, "count": 1}
        else:
            self.daily_flips[symbol]["count"] += 1
            
        self.last_flip_times[symbol] = time.time()

    # ============================================================
    # CHECK STOPS - COMPLETE ORIGINAL
    # ============================================================

    def check_stops(self, portfolio, data_provider, symbol_filter=None):
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
        stop_signals = []
        now = datetime.now(timezone.utc)
        
        # 🛡️ PHOENIX V3: Iteramos sobre el Libro Mayor Virtual para asegurar aislamiento Scalping vs Swing
        for v_key, pos in portfolio.virtual_ledger.items():
            qty = pos.get('quantity', 0.0)
            if abs(qty) < 1e-8: continue
            
            # v_key is like 'BTC/USDT_SCALPING'
            parts = v_key.rsplit('_', 1)
            symbol = parts[0]
            pos_horizon = parts[1] if len(parts) > 1 else pos.get('horizon', 'SCALPING')
            
            # FORENSIC-V12 FIX #2: Skip positions not matching the filter
            if symbol_filter and symbol != symbol_filter:
                continue
                
            current_price = pos.get('current_price')
            entry_price = pos.get('avg_price')
            if not current_price or not entry_price: continue
            
            # ================================================================
            # FORENSIC REMEDIATION: Horizon-aware SL/TP fallbacks
            # QUÉ: Los fallbacks originales (0.003 SL / 0.008 TP) eran LETALES
            #   para BTC con ATR normal de 0.5-1.5% en 5m.
            # POR QUÉ: Con 0.3% SL y 10x leverage, cualquier movimiento normal
            #   de BTC (0.2-0.5%) disparaba Hard SL instantáneamente (-2% a -5%).
            # PARA QUÉ: SL debe ser ≥ ATR medio para sobrevivir ruido normal.
            # ================================================================
            default_sl = 0.006 if pos_horizon == 'SCALPING' else 0.015
            default_tp = 0.012 if pos_horizon == 'SCALPING' else 0.035
            sl_pct = pos.get('sl_pct', default_sl) or default_sl
            tp_pct = pos.get('tp_pct', default_tp) or default_tp
            hwm = pos.get('high_water_mark', entry_price)
            lwm = pos.get('low_water_mark', entry_price)
            
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if qty > 0 else \
                                 ((entry_price - current_price) / entry_price) * 100
            # 🕰️ [FINOPS TIME-STOP] Protección de cuentas Micro ($13) vs Funding Fee en Swing Shorts/Longs
            if pos_horizon in ['SWING', 'MACRO'] and portfolio.get_total_equity() < 50.0:
                if 'entry_time' in pos:
                    entry_time_val = pos['entry_time']
                    if hasattr(entry_time_val, 'timestamp'):
                        entry_time_val = entry_time_val.timestamp()
                    hours_held = (time.time() - entry_time_val) / 3600
                    if hours_held > 7.5 and unrealized_pnl_pct < 0.5: # Si no ganamos al menos +0.5% en 7.5 hrs, abortar antes del funding
                        logger.warning(f"🛑 [FINOPS TIME-STOP] {symbol} {pos_horizon} max holding time (7.5h) reached. Exiting to prevent Funding Fee bleed.")
                        stop_signals.append(SignalEvent(strategy_id="TIME_STOP", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                        continue

            # LONG POSITION
            if qty > 0:
                # 1. Momentum Exit (Proactive)
                if self._check_momentum_exit(symbol, 'LONG', data_provider):
                    print(f"🪂 {pos_horizon} MOMENTUM EXIT {symbol}! (Proactive)")
                    stop_signals.append(SignalEvent(strategy_id="MOMENT_MGR", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, 0.0)
                    continue

                # FORENSIC FIX #9: EXPLICIT TAKE PROFIT (was completely missing)
                # Before: Trades at 100% TP were NEVER closed — only trailing at 25-85%
                # Now: Close at full TP target, then let trailing handle partial profits
                if current_price >= (entry_price * (1 + tp_pct)):
                    tp_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    print(f"🎯 [LONG {pos_horizon}] TAKE PROFIT {symbol}! +{tp_pnl_pct:.2f}% (Target: {tp_pct*100:.2f}%)")
                    stop_signals.append(SignalEvent(strategy_id="TAKE_PROFIT", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, tp_pnl_pct)
                    continue

                # 2 & 3. 3-STAGE ADAPTIVE TRAILING + TURBO-BREAKEVEN
                # FORENSIC-V11: Use BBO Maker fee for round-trip (90%+ of orders are LIMIT)
                # Before: TAKER fee (0.0375%) × 2 = 0.075% → threshold 0.1125%
                # After: MAKER fee (0.02%) × 2 = 0.04% → threshold 0.06%
                # This RAISES the turbo threshold because MAKER fee is lower:
                # net profit after fees is HIGHER, so we can afford to wait longer.
                _maker_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
                _taker_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)
                fee_buffer = (_maker_fee + _taker_fee)  # Entry=Maker, Exit=varies
                peak_pnl = ((hwm - entry_price) / entry_price) * 100
                tp_target_pct = tp_pct * 100 if tp_pct > 0 else 1.0  # Safe fallback
                
                # ⚡ Turbo-Breakeven (Stage 0): Immediate capital protection once fee gap is broken
                # En force un escape de seguridad absoluto ante la menor insinuación de volatilidad adversa
                # Se activa con spread + comisiones
                if peak_pnl >= (fee_buffer * 100 * 1.5): # 1.5x Fees es el umbral para activar
                    # We lock in entry_price + fee_buffer + slippage + micro-profit
                    turbo_be_price = entry_price * (1 + fee_buffer + 0.0006) # 0.06% Total FinOps Net-Zero
                    if current_price < turbo_be_price: # Price crashed back after hitting PEAK
                        print(f"⚡ [LONG {pos_horizon}] TURBO-BREAKEVEN {symbol}! Peak +{peak_pnl:.2f}% gave us edge. Bailing at {current_price:.4f}")
                        stop_signals.append(SignalEvent(strategy_id="TURBO_BE", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                        self.record_trade_result(True, unrealized_pnl_pct)
                        continue
                
                progress = peak_pnl / tp_target_pct if tp_target_pct > 0 else 0
                
                trail_price = None
                trail_name = None
                
                # Fetch dynamically passed momentum threshold for this trade
                self._last_momentum_accel = pos.get('metadata', {}).get('momentum_exit_accel', -0.012) if isinstance(pos.get('metadata'), dict) else -0.012
                
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
                    print(f"🛡️/💰 [LONG {pos_horizon}] {trail_name} {symbol}! Triggered at {current_price:.4f} (Peak: +{peak_pnl:.2f}%)")
                    stop_signals.append(SignalEvent(strategy_id=trail_name, symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, unrealized_pnl_pct)
                    continue

                # 4. Initial Hard Stop Loss (Protective)
                if current_price < (entry_price * (1 - sl_pct)):
                    print(f"🛑 HARD SL [{pos_horizon}] {symbol}! {unrealized_pnl_pct:.2f}%")
                    stop_signals.append(SignalEvent(strategy_id="HARD_SL", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(False, unrealized_pnl_pct)
                    continue

            # SHORT POSITION
            elif qty < 0:
                # 1. Momentum Exit
                if self._check_momentum_exit(symbol, 'SHORT', data_provider):
                    print(f"🪂 {pos_horizon} SHORT MOMENTUM EXIT {symbol}! (Proactive)")
                    stop_signals.append(SignalEvent(strategy_id="MOMENT_MGR", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, 0.0)
                    continue

                # FORENSIC FIX #9: EXPLICIT TAKE PROFIT FOR SHORTS
                if current_price <= (entry_price * (1 - tp_pct)):
                    tp_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    print(f"🎯 [SHORT {pos_horizon}] TAKE PROFIT {symbol}! +{tp_pnl_pct:.2f}% (Target: {tp_pct*100:.2f}%)")
                    stop_signals.append(SignalEvent(strategy_id="TAKE_PROFIT", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, tp_pnl_pct)
                    continue

                # 2 & 3. 3-STAGE ADAPTIVE TRAILING + TURBO-BREAKEVEN
                # FORENSIC-V11: BBO Maker fee for round-trip (same fix as LONG side)
                _maker_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
                _taker_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)
                fee_buffer = (_maker_fee + _taker_fee)  # Entry=Maker, Exit=varies
                peak_pnl = ((entry_price - lwm) / entry_price) * 100
                tp_target_pct = tp_pct * 100 if tp_pct > 0 else 1.0  # Safe fallback
                
                # ⚡ Turbo-Breakeven (Stage 0): Immediate capital protection
                if peak_pnl >= (fee_buffer * 100 * 1.5): # 1.5x Fees
                    # We lock in entry_price - fee_buffer - slippage - micro-profit
                    turbo_be_price = entry_price * (1 - fee_buffer - 0.0006) # 0.06% Total FinOps Net-Zero
                    if current_price > turbo_be_price: # Price bounced back up
                        print(f"⚡ [SHORT {pos_horizon}] TURBO-BREAKEVEN {symbol}! Peak +{peak_pnl:.2f}% gave us edge. Bailing at {current_price:.4f}")
                        stop_signals.append(SignalEvent(strategy_id="TURBO_BE", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                        self.record_trade_result(True, unrealized_pnl_pct)
                        continue
                        
                progress = peak_pnl / tp_target_pct if tp_target_pct > 0 else 0
                
                trail_price = None
                trail_name = None
                
                self._last_momentum_accel = pos.get('metadata', {}).get('momentum_exit_accel', -0.012) if isinstance(pos.get('metadata'), dict) else -0.012
                
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
                    print(f"🛡️/💰 [SHORT {pos_horizon}] {trail_name} {symbol}! Triggered at {current_price:.4f} (Peak: +{peak_pnl:.2f}%)")
                    stop_signals.append(SignalEvent(strategy_id=trail_name, symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(True, unrealized_pnl_pct)
                    continue

                # 4. Initial Hard Stop
                if current_price > (entry_price * (1 + sl_pct)):
                    print(f"🛑 SHORT HARD SL [{pos_horizon}] {symbol}! {unrealized_pnl_pct:.2f}%")
                    stop_signals.append(SignalEvent(strategy_id="HARD_SL", symbol=symbol, datetime=now, signal_type=SignalType.EXIT, strength=1.0, horizon=pos_horizon))
                    self.record_trade_result(False, unrealized_pnl_pct)
                    continue
        
        return stop_signals


    # ============================================================
    # KILL SWITCH FACADE
    # ============================================================
    
    # Using the L596 update_equity instead.
    def activate_kill_switch(self, reason: str):
        if self.kill_switch:
            self.kill_switch.record_loss()
    def record_api_error(self):
        if self.kill_switch:
            self.kill_switch.record_api_error()
    def reset_api_errors(self):
        if self.kill_switch:
            self.kill_switch.reset_api_errors()
    
    # ============================================================
    # SNIPER STRATEGY METHODS (ORIGINAL)
    # ============================================================
    
    def calculate_dynamic_leverage(self, atr: float, price: float) -> int:
        print(f"Legacy calculate_dynamic_leverage called. Delegating...")
        result = safe_leverage_calculator.calculate_safe_leverage(atr, price)
        return result['leverage']
    
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
        if volatility > 0.05: # High Vol
            self.max_leverage = 3
            logger.warning(f"❄️ GARCH High Vol Cluster ({volatility:.4f}) -> Leverage CAPPED to 3x")
        elif volatility > 0.025: # Elevated Vol
            self.max_leverage = 7
        else: # Low/Stable Vol
            self.max_leverage = 12
            
        # Regime Specific Adjustments
        if regime == 'TRENDING_UP':
             self.max_leverage = min(self.max_leverage, 15) # Boost for BTC runs
        elif regime == 'CHOPPY':
             self.max_leverage = min(self.max_leverage, 5)  # Defensive
    
    def calculate_liquidation_price(self, entry_price: float, leverage: int, 
                                     direction: str, margin_type: str = 'ISOLATED') -> float:
        if leverage <= 0:
            return 0.0
        mmr = 0.004
        if direction == 'LONG':
            liq_price = entry_price * (1 - (1 / leverage) + mmr)
        else:
            liq_price = entry_price * (1 + (1 / leverage) - mmr)
        return liq_price
    
    def calculate_distance_to_liquidation(self, entry_price: float, current_price: float,
                                           leverage: int, direction: str) -> dict:
        liq_price = self.calculate_liquidation_price(entry_price, leverage, direction)
        if direction == 'LONG':
            distance = (current_price - liq_price) / current_price * 100
        else:
            distance = (liq_price - current_price) / current_price * 100
        return {
            'liq_price': liq_price,
            'distance_pct': distance,
            'is_danger': distance < 2.0
        }
    
    def calculate_sniper_position_size(self, capital: float, leverage: int, 
                                        entry_price: float) -> dict:
        notional = capital * leverage
        quantity = notional / entry_price if entry_price > 0 else 0
        margin_required = notional / leverage
        return {
            'notional': notional,
            'quantity': quantity,
            'margin_required': margin_required,
            'leverage': leverage
        }
    
    def check_portfolio_var(self, new_trade_value: float) -> bool:
        """
        [PHASE 10] Dynamic Hedging / VaR Check
        Calculates simple Parametric VaR (95%) for the portfolio.
        Returns False if adding 'new_trade_value' exceeds Max VaR allowed.
        """
        if not self.portfolio:
            return True
            
        # 1. Get total portfolio value
        total_equity = self.portfolio.get_total_equity()
        max_var_limit = total_equity * 0.05 # Max 5% VaR allowed
        
        # 2. Estimate Current VaR
        # Simplified: Using fixed volatility assumption (2% daily) if GARCH not available per symbol here
        # In full implementation, we'd use Correlation Matrix from Phase 6
        
        current_exposure = 0.0
        for s, pos in self.portfolio.positions.items():
            current_exposure += abs(pos['quantity'] * pos['current_price'])
            
        future_exposure = current_exposure + new_trade_value
        
        # Simple VaR = Exposure * Volatility * Z(95%)
        # Z(95%) ~= 1.65
        # Assuming avg daily vol of 3% for crypto portfolio
        daily_vol = 0.03
        
        estimated_var = future_exposure * daily_vol * 1.65
        
        if estimated_var > max_var_limit:
            logger.warning(f"🛡️ VaR REJECTION: Est VaR ${estimated_var:.2f} > Limit ${max_var_limit:.2f}")
            return False
            
        return True

    def validate_sniper_order(self, symbol: str, quantity: float, 
                               entry_price: float, leverage: int) -> dict:
        notional = quantity * entry_price
        margin_required = notional / leverage
        # PHASE 1: Execution Audit ($13 Micro-Account Hardening)
        # We increase the hard MIN_NOTIONAL from Binance's 5.0 to 6.0 to prevent 
        # rejected orders due to Taker fees or sub-cent slippage pushing it below limits.
        MIN_NOTIONAL = 6.0
        MIN_MARGIN = 1.0
        
        if notional < MIN_NOTIONAL:
            return {
                'is_valid': False,
                'reason': f'Notional ${notional:.2f} < MIN ${MIN_NOTIONAL}',
                'adjusted_qty': MIN_NOTIONAL / entry_price
            }
        if margin_required < MIN_MARGIN:
            return {
                'is_valid': False,
                'reason': f'Margin ${margin_required:.2f} < MIN ${MIN_MARGIN}',
                'adjusted_qty': (MIN_MARGIN * leverage) / entry_price
            }
        return {
            'is_valid': True,
            'reason': 'OK',
            'adjusted_qty': quantity
        }