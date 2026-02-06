"""
Risk Manager FINAL CORREGIDO: 
- Original check_stops (COMPLETO)
- Original Sniper Methods (TODOS)
- Scientific: Kelly, CVaR, Fees (FIXED)
- Growth Phases (CALIBRADO para $12)
- Leverage calculation (CORREGIDO)
- Enhanced debugging
"""

from core.events import OrderEvent, SignalEvent
from core.enums import OrderSide, SignalType, OrderType
from core.resolution_state import ResolutionState
from core.world_awareness import world_awareness
from config import Config
from .kill_switch import KillSwitch
from utils.debug_tracer import trace_execution
from datetime import timedelta, datetime, timezone
import numpy as np
from collections import deque
from utils.cooldown_manager import cooldown_manager
from utils.safe_leverage import safe_leverage_calculator
from utils.cooldown_manager import cooldown_manager
from utils.safe_leverage import safe_leverage_calculator
from utils.cooldown_manager import cooldown_manager
from utils.safe_leverage import safe_leverage_calculator
from utils.logger import logger
from utils.analytics import AnalyticsEngine
from core.data_handler import get_data_handler
from utils.statistics_pro import StatisticsPro



# ============================================================
# SCIENTIFIC RISK TOOLS (FIXED)
# ============================================================

class FeeCalculator:
    """Cálculo preciso de fees - CORREGIDO"""
    TAKER_FEE_BNB = Config.BINANCE_TAKER_FEE_BNB  # Unmanaged from Config (FIXED)
    
    @staticmethod
    def calculate_round_trip_fee(notional_value: float) -> float:
        return notional_value * FeeCalculator.TAKER_FEE_BNB * 2


class CVaRCalculator:
    """Conditional Value at Risk - CALIBRADO para leverage"""
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.loss_history = deque(maxlen=100)
        
    def update(self, pnl_pct: float):
        if pnl_pct < 0:
            self.loss_history.append(abs(pnl_pct))
    
    def calculate_cvar(self) -> float:
        if len(self.loss_history) < 10:
            return 0.05  # 5% default (REALISTA con 10x lev)
        
        losses = sorted(self.loss_history, reverse=True)
        var_index = max(1, int(len(losses) * (1 - self.confidence_level)))
        worst_losses = losses[:var_index]
        return np.mean(worst_losses) if worst_losses else 0.05

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
        self.kill_switch = KillSwitch()
        
        # Kelly Stats (FIXED bootstrap)
        self.win_count = 0
        self.loss_count = 0
        self.bootstrap_win_rate = 0.52  # REALISTA para scalping
        self.bootstrap_trades = 20
        
        # Growth Phases (CALIBRADO)
        self.LEVERAGE_GROWTH = 10
        self.POSITION_PCT_GROWTH = 0.30  # 30% en growth
        
        # Phase 5: Flipping State
        self.daily_flips = {}  # {symbol: {date: "YYYY-MM-DD", count: N}}
        # Phase 5: Flipping State
        self.daily_flips = {}  # {symbol: {date: "YYYY-MM-DD", count: N}}
        self.last_flip_times = {} # {symbol: timestamp}
        
        # Phase 6: Stress Testing
        self.stress_score = 100.0 # Default perfect score (0% Ruin Risk)
        self.last_stress_check = 0
        self.stress_check_interval = 3600 # Check every hour

        # Meta-Brain Integration (Phase 7)
        self.strategy_selector = None # Set by Engine
        
        # Execution Caps
        self.MAX_TRADES_TOTAL = 100
        self.global_regime = 'UNKNOWN' # BTC Leader (Phase 8)
        
        # Phase 14: Dynamic Capital Allocation
        self.resolution_state = ResolutionState.STABLE
        self.recovery_threshold = 0.05 # 5% Drawdown triggers recovery
        self.growth_threshold = 0.10   # 10% Profit triggers growth
        
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

    # ============================================================
    # GROWTH PHASE METHODS (FIXED)
    # ============================================================
    
    def get_current_phase(self, capital: float) -> str:
        """Delegate to SafeLeverageCalculator"""
        return safe_leverage_calculator.get_phase(capital)

    def get_win_rate(self) -> float:
        total = self.win_count + self.loss_count
        if total < self.bootstrap_trades:
            # Weighted average: más peso a datos reales conforme crecen
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

    def calculate_kelly_fraction(self, strategy_id: str = None, rr_ratio: float = 0.75) -> float:
        """Phase 6: Continuous Kelly with Optimization + Phase 14: Strategy Specific"""
        
        # Phase 14: Use strategy-specific metrics if available
        p = 0.0
        if strategy_id and self.portfolio:
             metrics = self.portfolio.get_strategy_metrics(strategy_id)
             if metrics['total_trades'] >= 10:
                 p = metrics['win_rate']
        
        # Fallback to global Bayesian WR
        if p == 0.0:
            p = self.get_bayesian_win_rate()
        
        # Use StatisticsPro for precise continuous Kelly
        kelly = StatisticsPro.kelly_criterion_continuous(p, rr_ratio)
        fractional_kelly = kelly * 0.4
        
        # FIXED: Diferentes límites según capital
        capital = self.portfolio.get_total_equity() if self.portfolio else safe_leverage_calculator.get_capital()
        if capital < 20:
            return max(0.25, min(fractional_kelly, 0.40))  # 25-40% growth
        else:
            return max(0.15, min(fractional_kelly, 0.35))  # 15-35% normal

    def record_trade_result(self, is_win: bool, pnl_pct: float = 0):
        if is_win:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.cvar_calc.update(pnl_pct)

    def update_equity(self, equity: float):
        """
        External update from Main Loop to sync Kill Switch & Safe Leverage.
        """
        # 1. Update Kill Switch (Critical Safety)
        if self.kill_switch:
            self.kill_switch.update_equity(equity)
            
        # 2. Update Safe Leverage Calculator (Growth Phase Tracking)
        safe_leverage_calculator.update_capital(equity)

    def _update_capital_tracking(self, equity: float):
        """Internal helper to update Kill Switch during sizing checks"""
        self.update_equity(equity)

    def _get_dynamic_risk_per_trade(self, capital: float) -> float:
        """
        Calcula riesgo por trade basado en Profit Lock y Drawdown.
        PROFESSOR METHOD:
        1. Base Risk: 1% (o menos si hay DD).
        2. Profit Scaling: Si ganamos mucho (+50%, +100%), reducimos riesgo para "asegurar".
        3. Protected Floor: Nunca arriesgar el capital "bloqueado" (80% de ganancias).
        """
        initial = safe_leverage_calculator.initial_capital
        peak = safe_leverage_calculator.peak_capital
        
        # 1. Base Logic (Drawdown Protection)
        risk_pct = 0.01  # Default 1%
        if peak > 0:
            dd = (peak - capital) / peak
            if dd > 0.10: risk_pct = 0.005 # 0.5%
            elif dd > 0.05: risk_pct = 0.0075 # 0.75%

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

        return risk_pct

    def _update_stress_metrics(self):
        """Phase 6: Update Risk of Ruin (PoR) metrics periodically."""
        import time
        now = time.time()
        if now - self.last_stress_check < self.stress_check_interval:
            return

        try:
            dh = get_data_handler()
            # Determine path (futures/spot) - simplistic approach
            csv_path = "dashboard/data/futures/trades.csv" 
            trades = get_data_handler().load_trades_df(csv_path)
            
            pnl_returns = []
            if not trades.empty:
                for _, t in trades.iterrows():
                    if t['entry_price'] > 0 and t['quantity'] > 0:
                        cost = t['entry_price'] * t['quantity']
                        ret = t['net_pnl'] / cost
                        pnl_returns.append(ret)
            
            if len(pnl_returns) >= 20:
                paths = StatisticsPro.generate_monte_carlo_paths(pnl_returns, n_sims=1000)
                metrics = StatisticsPro.calculate_stress_metrics(paths)
                self.stress_score = metrics.get('stress_score', 100.0)
                por = metrics.get('por', 0.0)
                
                if por > 5.0:
                    logger.warning(f"⚠️ HIGH RUIN RISK: PoR={por:.1f}% | Sizing will be reduced.")
            
            self.last_stress_check = now
        except Exception as e:
            logger.error(f"Stress Check Failed: {e}")

    def _calculate_dynamic_stop_loss(self, atr_pct: float) -> float:
        """
        Calcula SL dinámico basado en régimen de volatilidad.
        - Low Vol (<0.5%): SL = 3x ATR (Dar espacio)
        - High Vol (>1.0%): SL = 2x ATR (Cortar rápido)
        """
        if atr_pct < 0.005:  # Low Vol
            mult = 3.0
        elif atr_pct > 0.01: # High Vol
            mult = 2.0
        else:                # Normal
            mult = 2.5
            
        sl_raw = atr_pct * mult
        
        # Hard limits
        return max(0.003, min(sl_raw, 0.015)) # Min 0.3%, Max 1.5%
    
    def _update_capital_tracking(self, current_equity: float):
        """
        Update peak capital for accurate drawdown calculation.
        Delegated to SafeLeverageCalculator.
        """
        safe_leverage_calculator.update_capital(current_equity)
        # Update local references if needed (deprecated)
        self.peak_capital = safe_leverage_calculator.peak_capital

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
        if capital > 15 and Config.BINANCE_USE_TESTNET:
            logger.info(f"🧪 TESTNET: Simulating $15 account (Actual: ${capital:.2f})")
            capital = 15.0

        if capital > 15 and Config.BINANCE_USE_TESTNET:
            logger.info(f"🧪 TESTNET: Simulating $15 account (Actual: ${capital:.2f})")
            capital = 15.0

        # Phase 6: Equal Weighting for Fair Competition (Demo Only)
        if Config.BINANCE_USE_DEMO and getattr(Config.Strategies, 'PERMISSIVE_MODE', False):
            # Bypass Kelly/Growth logic to test pure signal quality
            fixed_pct = getattr(Config.Strategies, 'DEMO_EQUAL_WEIGHTING', 0.05)
            logger.info(f"🧪 LAB MODE: Using Fixed Equal Weighting ({fixed_pct*100}%) for comparison.")
            return capital * fixed_pct

        phase = safe_leverage_calculator.get_phase(capital)
        
        # MICRO ACCOUNT FIX: Aggressive sizing for small capital
        # MICRO ACCOUNT FIX: Aggressive sizing for small capital
        if capital < 50:
            base_pct = Config.POSITION_SIZE_MICRO_ACCOUNT  # 40% (Defined in Config)
        elif "GROWTH" in phase:
            base_pct = self.POSITION_PCT_GROWTH  # 30%
        elif capital < 1000:
            # Phase 14: Pass strategy_id for specific Kelly
            strat_id = getattr(signal_event, 'strategy_id', None)
            kelly = self.calculate_kelly_fraction(strategy_id=strat_id)
            base_pct = max(0.20, kelly)
        else:
            strat_id = getattr(signal_event, 'strategy_id', None)
            base_pct = self.calculate_kelly_fraction(strategy_id=strat_id)
            
        target_exposure = capital * base_pct
        
        # ATR-based sizing (VOLATILITY ADJUSTED)
        # Size = (Capital * Risk%) / SL_Distance
        if hasattr(signal_event, 'atr') and signal_event.atr is not None and signal_event.atr > 0:
            current_risk_pct = self._get_dynamic_risk_per_trade(capital)
            risk_amount = capital * current_risk_pct
            
            # Estimate SL distance for sizing (use dynamic logic)
            atr_pct = (signal_event.atr / current_price) if current_price and current_price > 0 else 0.02
            est_sl_pct = self._calculate_dynamic_stop_loss(atr_pct)
            
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
        
        # MICRO ACCOUNT FIX: Reduce signal strength impact for small accounts
        if hasattr(signal_event, 'strength'):
            if capital < 50:
                target_exposure *= 1.0 # Ignore strength for micro
            else:
                target_exposure *= min(signal_event.strength, 1.2)
            
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
        
        # CVaR reduction (FIXED: Use peak_capital for accurate drawdown)
        current_dd = 1 - (capital / self.peak_capital) if capital < self.peak_capital else 0
        if self.cvar_calc.should_reduce_risk(current_dd):
            target_exposure *= 0.5
            logger.warning(f"⚠️ CVaR: Reducing size 50% (DD: {current_dd*100:.1f}%)")
        
        # --- PHASE 14: DYNAMIC RECOVERY STATE ---
        self._update_resolution_state(current_dd)
        if self.resolution_state == ResolutionState.RECOVERY:
            # Defensive Mode: Cut risk by 50% until we recover half the DD
            target_exposure *= 0.5
            logger.warning(f"🛡️ [RECOVERY MODE] Drawdown ({current_dd*100:.1f}%) > 5%. Sizing halved.")
        elif self.resolution_state == ResolutionState.GROWTH:
             # Aggressive Mode: 1.2x boost if strictly profitable
             target_exposure *= 1.2
             logger.info(f"🚀 [GROWTH MODE] Account flying high. Boost enabled.")
        
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

        # BINANCE MINIMUM: Ensure position meets $5 minimum for futures
        MIN_MARGIN = 5.0
        if target_exposure < MIN_MARGIN and capital >= MIN_MARGIN:
            target_exposure = MIN_MARGIN
            logger.info(f"📈 Boosted to Binance min: ${target_exposure:.2f}")
            
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
        """
        Final safety check: Block trades if historical Expectancy < 0.
        Rule: "Do not play games you are statistically likely to lose."
        """
        try:
            dh = get_data_handler()
            csv_path = "dashboard/data/trades.csv" 
            trades = dh.load_trades_df(csv_path)
            
            if trades.empty:
                return True
                
            sym_trades = trades[trades['symbol'] == symbol]
            # Need distinct trades, not just rows (if split)
            # Assuming row per trade for simple count
            if len(sym_trades) < 10:
                return True # Learning mode
                
            stats = AnalyticsEngine.calculate_expectancy(sym_trades)
            e = stats.get('expectancy', 0.0)
            
            if e <= 0:
                logger.warning(f"🛑 [RiskMgr] Gatekeeper Block: {symbol} has NEGATIVE Expectancy (${e:.4f}).")
                return False
                
            return True
        except Exception as e:
            logger.error(f"⚠️ Risk Expectancy Check failed: {e}")
            return True # Fail open

    # ============================================================
    # ORDER GENERATION (FIXED)
    # ============================================================

    @trace_execution
    def generate_order(self, signal_event, current_price):
        """ORIGINAL with FIXED leverage logic"""
        
        # -1. Kill Switch
        if not self.kill_switch.check_status():
            logger.warning(f"💀 Kill Switch Active: {self.kill_switch.activation_reason}")
            return None
            
        # -0.7 MARGIN RATIO SAFETY (Phase 6 Final)
        # Block new positions if Margin Ratio > 50% (75% for micro) to protect session capital.
        if self.portfolio:
             status = get_data_handler().load_cached_status()
             if status:
                 maint_margin = status.get('maint_margin', 0)
                 margin_balance = status.get('margin_balance', 0)
                 
                 if margin_balance > 0:
                     margin_ratio = (maint_margin / margin_balance) * 100
                     equity = self.portfolio.get_total_equity()
                     # RELAXED FOR MICRO: If equity < $50, allow up to 75% margin ratio
                     limit = 75 if equity < 50 else 50
                     
                     if margin_ratio > limit:
                         logger.warning(f"🛡️ [RiskMgr] MARGIN SAFETY BLOCK: Ratio {margin_ratio:.1f}% > {limit}%. Entry Rejected.")
                         return None
        
        # 0. Daily Frequency Limits (User Request: 15/symbol)
        if signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.daily_trade_logs:
                self.daily_trade_logs[today] = {}
            
            symbol_count = self.daily_trade_logs[today].get(signal_event.symbol, 0)
            total_count = sum(self.daily_trade_logs[today].values())
            
            if symbol_count >= self.MAX_TRADES_PER_SYMBOL:
                logger.info(f"🚫 [{signal_event.symbol}] Limit: Max daily trades ({self.MAX_TRADES_PER_SYMBOL}) reached.")
                return None
            
            if total_count >= self.MAX_TRADES_TOTAL:
                logger.info(f"🚫 [SYSTEM] GLOBAL Limit: Max daily trades ({self.MAX_TRADES_TOTAL}) reached.")
                return None
            
        # -0.5 Proactive Expectancy Gate (Phase 5)
        if signal_event.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.REVERSE]:
            if not self._check_expectancy_viability(signal_event.symbol):
                return None
        
        # -0.4 Meta-Brain Veto (Phase 7)
        if self.strategy_selector and signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
            strat_id = getattr(signal_event, 'strategy_id', 'Unknown').upper()
            if not self.strategy_selector.should_allow_trade(strat_id):
                return None

        # -0.3 GLOBAL REGIME VETO (Phase 8 - Institutional Rule)
        # QUÉ: Veto Global basado en el Líder (BTC).
        # POR QUÉ: Evitar "longs" en Altcoins cuando BTC está en tendencia bajista clara.
        if self.global_regime == 'TRENDING_BEAR':
            if signal_event.symbol != 'BTC/USDT' and signal_event.signal_type == SignalType.LONG:
                logger.warning(f"🛡️ [RiskMgr] INSTITUTIONAL VETO: Blocking LONG {signal_event.symbol} because BTC is Bearish.")
                return None


        # 0. Max positions
        if self.portfolio:
            open_positions = sum(1 for pos in self.portfolio.positions.values() if pos['quantity'] != 0)
            
            # SMART FLIP: Check if we already have a position in this symbol
            # If so, allow the trade (it's a flip, reduction, or pyramiding) even if max positions reached
            is_existing_position = False
            if signal_event.symbol in self.portfolio.positions:
                qty = self.portfolio.positions[signal_event.symbol]['quantity']
                if abs(qty) > 0:
                    is_existing_position = True
            
            # 0.1 Dynamic Position Limits (ADAPTATIVE)
            capital = self.portfolio.get_total_equity()
            dynamic_max = Config.MAX_CONCURRENT_POSITIONS # Use explicit limit for multi-symbol
            
            # DIRECTIONAL PROTECTION: Block duplicate entries in same direction
            if is_existing_position:
                qty = self.portfolio.positions[signal_event.symbol]['quantity']
                # If we are LONG and signal is LONG -> Block Duplicate
                if qty > 0 and signal_event.signal_type == SignalType.LONG:
                    logger.info(f"🛡️ [{signal_event.symbol}] Directional Block: Already LONG. Duplicate entry rejected.")
                    return None
                # If we are SHORT and signal is SHORT -> Block Duplicate
                if qty < 0 and signal_event.signal_type == SignalType.SHORT:
                    logger.info(f"🛡️ [{signal_event.symbol}] Directional Block: Already SHORT. Duplicate entry rejected.")
                    return None
                # If it's a FLIP (Long -> Short or Short -> Long), we allow it
            
            # Block only if it's a NEW position and we are at capacity
            if not is_existing_position and open_positions >= dynamic_max and signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
                logger.info(f"⚖️ [{signal_event.symbol}] Limit: Max positions ({dynamic_max}) reached. Entry Blocked.")
                return None
        
        
        # 0.5 Cooldowns (Centralized)
        if signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
            # DOGE SPECIAL RULE: 60m cooldown if last trade was a loss
            cooldown_seconds = 60 if signal_event.symbol != "DOGEUSDT" else 3600
            
            can_trade, reason = cooldown_manager.can_trade(signal_event.symbol, strategy_id="RISK_MANAGER")
            if not can_trade:
                logger.info(f"❄️ [{signal_event.symbol}] Cooldown active: {reason}")
                return None
        
        # 0.6 MARGIN VALIDATION (Rule 2.2)
        # PROFESSOR METHOD:
        # QUÉ: Validación preventiva de colateral.
        # POR QUÉ: Binance rechaza órdenes si el margen (notional/leverage) supera el balance libre + mantenimiento.
        # CÓMO: Calculamos coste estimado y comparamos con available_cash de Portfolio.
        if self.portfolio and signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
            available = self.portfolio.get_available_cash()
            
            # Estimación de coste de margen para la orden
            # Notional = Sizing base (ajustado en generate_order)
            margin_size = self.size_position(signal_event, current_price)
            
            if margin_size > available:
                # FALLBACK: Intentar reducir al mínimo si el capital es insuficiente
                logger.warning(f"⚠️ [{signal_event.symbol}] Insufficient Margin: Need ${margin_size:.2f}, Have ${available:.2f}")
                
                # Minimum viable margin fallback (Binance requires ~$5 notional)
                # If leverage is 10x, margin needed is ~$0.50
                min_required = 1.0  # Buffer de seguridad
                if available >= min_required:
                    logger.info(f"🔄 Using Fallback margin sizing: ${min_required:.2f}")
                    margin_size = min_required
                else:
                    logger.error(f"❌ [{signal_event.symbol}] Critical: Not enough margin even for fallback. Rejecting.")
                    return None

        
        # PYRAMIDING (FIXED: Don't modify frozen SignalEvent)
        pyramid_multiplier = 1.0  # Default: no adjustment
        if self.portfolio and signal_event.signal_type == SignalType.LONG:
            existing = self.portfolio.positions.get(signal_event.symbol, {})
            existing_qty = existing.get('quantity', 0)
            
            if existing_qty > 0:
                entry_price = existing.get('avg_price', 0)
                if entry_price > 0:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    if profit_pct >= 0.8:
                        # Use multiplier instead of modifying frozen event
                        pyramid_multiplier = 0.5
                        logger.info(f"📈 PYRAMIDING: +{profit_pct:.1f}%")
                    else:
                        print(f"⚠️ Position exists: +{profit_pct:.1f}%")
                        return None
                else:
                    return None
        
        # === EXIT (ORIGINAL) ===
        if signal_event.signal_type == SignalType.EXIT:
            if self.portfolio and signal_event.symbol in self.portfolio.positions:
                existing_qty = self.portfolio.positions[signal_event.symbol]['quantity']
                if existing_qty == 0:
                    return None
                quantity = abs(existing_qty)
                direction = OrderSide.SELL if existing_qty > 0 else OrderSide.BUY
                dollar_size = quantity * current_price
            else:
                return None
        
        # === LONG (FIXED LEVERAGE) ===
        elif signal_event.signal_type == SignalType.LONG:
            margin_size = self.size_position(signal_event, current_price)
            
            # Calculate leverage (SAFE LEVERAGE)
            # FIXED: Robust None guard for ATR (Rule 1.1)
            atr_val = getattr(signal_event, 'atr', None)
            if atr_val is None:
                atr_val = (current_price * 0.02) if current_price else 0.0
                
            safe_calc = safe_leverage_calculator.calculate_safe_leverage(atr_val, current_price)
            leverage = safe_calc['leverage']
            
            # FIXED: Leverage validation y boost inteligente
            if not safe_calc['is_safe']:
                logger.warning(f"⚠️ Unsafe leverage: {safe_calc['reason']}")
                return None
            
            # Additional safety cap
            if leverage > Config.Sniper.MAX_LEVERAGE:
                leverage = Config.Sniper.MAX_LEVERAGE
                logger.info(f"⚡ Leverage capped: {leverage}x")
            
            # FIXED: Boost más inteligente
            capital = self.portfolio.get_total_equity() if self.portfolio else self.current_capital
            if capital < 20 and leverage < 8:
                leverage = 8  # Mínimo 8x para cuentas micro
                logger.info(f"🚀 Growth boost: {leverage}x")
            
            if margin_size is None or leverage is None:
                return None
                
            notional_value = margin_size * leverage
            
            logger.info(f"🎯 LONG: margin=${margin_size:.2f}, lev={leverage}x, notional=${notional_value:.2f}")
            
            # Binance minimum
            MIN_NOTIONAL = 5.0
            if notional_value < MIN_NOTIONAL:
                required_margin = (MIN_NOTIONAL / leverage) * 1.05
                available = self.portfolio.get_available_cash() if self.portfolio else margin_size
                if required_margin > (available or 0):
                    logger.warning(f"⚠️ Insufficient: Need ${required_margin:.2f}")
                    return None
                margin_size = required_margin
                notional_value = margin_size * leverage
                logger.info(f"⚠️ Boosted to min: ${notional_value:.2f}")
            
            # FIXED: Fee validation más permisiva
            expected_profit = notional_value * 0.015
            fees = self.fee_calc.calculate_round_trip_fee(notional_value)
            if fees > (expected_profit * 0.45):  # 45% max (era 40%)
                logger.warning(f"📉 Fees too high: ${fees:.4f} vs profit ${expected_profit:.4f}")
                return None
            
            dollar_size = margin_size
            if not current_price or current_price <= 0:
                return None
            quantity = notional_value / current_price
            
            if quantity <= 0:
                return None
            direction = OrderSide.BUY
        
        # === SHORT (SAME LOGIC) ===
        elif signal_event.signal_type == SignalType.SHORT:
            if not Config.BINANCE_USE_FUTURES:
                print(f"⚠️ SHORT rejected: Spot mode")
                return None

            margin_size = self.size_position(signal_event, current_price)
            # FIXED: Robust None guard for ATR (Rule 1.1)
            atr_val = getattr(signal_event, 'atr', None)
            if atr_val is None:
                atr_val = (current_price * 0.02) if current_price else 0.0
                
            safe_calc = safe_leverage_calculator.calculate_safe_leverage(atr_val, current_price)
            leverage = safe_calc['leverage']
            
            if not safe_calc['is_safe']:
                return None
            
            if leverage > Config.Sniper.MAX_LEVERAGE:
                leverage = Config.Sniper.MAX_LEVERAGE
            
            capital = self.portfolio.get_total_equity() if self.portfolio else self.current_capital
            if capital < 20 and leverage < 8:
                leverage = 8
            
            if margin_size is None or leverage is None:
                return None
                
            notional_value = margin_size * leverage
            
            print(f"🎯 SHORT: margin=${margin_size:.2f}, lev={leverage}x, notional=${notional_value:.2f}")
            
            MIN_NOTIONAL = 5.0
            if notional_value < MIN_NOTIONAL:
                required_margin = (MIN_NOTIONAL / leverage) * 1.05
                available = self.portfolio.get_available_cash() if self.portfolio else margin_size
                if required_margin > (available or 0):
                    return None
                margin_size = required_margin
                notional_value = margin_size * leverage
            
            expected_profit = notional_value * 0.015
            fees = self.fee_calc.calculate_round_trip_fee(notional_value)
            if fees > (expected_profit * 0.45):
                return None
            
            dollar_size = margin_size
            if not current_price or current_price <= 0:
                return None
            quantity = notional_value / current_price
            
            if quantity <= 0:
                return None
            direction = OrderSide.SELL
        
        else:
            print(f"⚠️ Unknown signal: {signal_event.signal_type}")
            return None
        
        # === REVERSE (NEW Phase 5) ===
        if signal_event.signal_type == SignalType.REVERSE:
            # 1. Check if we have an existing position to flip
            if not self.portfolio or signal_event.symbol not in self.portfolio.positions:
                logger.warning(f"🔄 [{signal_event.symbol}] Reverse signal discarded: No open position found.")
                return None
            
            existing_pos = self.portfolio.positions[signal_event.symbol]
            existing_qty = existing_pos.get('quantity', 0)
            
            if abs(existing_qty) == 0:
                logger.warning(f"🔄 [{signal_event.symbol}] Reverse signal discarded: Position already CLOSED.")
                return None
            
            # 2. Analyze Viability
            current_pnl = 0.0
            if 'avg_price' in existing_pos and 'current_price' in existing_pos:
                ep = existing_pos['avg_price']
                cp = existing_pos['current_price']
                if ep > 0:
                    current_pnl = (cp - ep) / ep if existing_qty > 0 else (ep - cp) / ep
            
            atr_pct = (signal_event.atr / current_price) if signal_event.atr and current_price > 0 else 0.01
            
            viability = self.analyze_flip_viability(
                symbol=signal_event.symbol,
                current_pnl_pct=current_pnl,
                next_signal_strength=signal_event.strength,
                atr_pct=atr_pct
            )
            
            if not viability['is_viable']:
                logger.info(f"🔄 [{signal_event.symbol}] Flip Rejected: {viability['reason']}")
                return None
            
            # 3. Size the NEW position (Same logic as LONG/SHORT)
            # Flip is Close + Open. We need the target quantity for the NEW direction.
            # Strategy ID tells us the NEW direction logic
            is_currently_long = existing_qty > 0
            new_direction = OrderSide.SELL if is_currently_long else OrderSide.BUY
            
            margin_size = self.size_position(signal_event, current_price)
            safe_calc = safe_leverage_calculator.calculate_safe_leverage(signal_event.atr or 0, current_price)
            leverage = safe_calc['leverage']
            
            notional_value = margin_size * leverage
            new_qty = notional_value / current_price if current_price > 0 else 0
            
            # 4. Atomic Execution Info: Total quantity = Close Old + Open New
            # But the order executor/engine will handle the sequence. 
            # We return a REVERSE OrderEvent or a standard OrderEvent marked as flip.
            # Let's use a standard OrderEvent but the quantity must be LARGE enough to FLIP.
            total_flip_qty = abs(existing_qty) + new_qty
            
            direction = new_direction
            dollar_size = margin_size # Margin for the NEW leg
            quantity = total_flip_qty
            
            logger.info(f"🔄 [{signal_event.symbol}] FLIPPING {existing_qty:.4f} -> {new_qty:.4f} (Total Order: {quantity:.4f} {direction.name})")
            
            # Record the flip
            self._record_flip(signal_event.symbol)
        
        # Cash reservation (ORIGINAL)
        if self.portfolio and signal_event.signal_type in [SignalType.LONG, SignalType.SHORT]:
            available = self.portfolio.get_available_cash()
            
            if dollar_size < 5.0:
                logger.warning(f"⚠️ Size too small: ${dollar_size:.2f}")
                return None
                
            if available < dollar_size:
                logger.warning(f"⚠️ Insufficient: ${available:.2f}")
                return None
            
            if not self.portfolio.reserve_cash(dollar_size):
                logger.warning(f"⚠️ Reserve failed: ${dollar_size:.2f}")
                return None
            
            # Cooldown (ORIGINAL)
            if self.current_regime == 'TRENDING':
                cooldown_duration = 2
            elif self.current_regime == 'CHOPPY':
                cooldown_duration = 15
            else:
                cooldown_duration = 5
        else:
            cooldown_duration = 1
        
        
        # CRITICAL: Force cooldown update immediately
        # Use explicit record_trade to mark symbol as "busy"
        cooldown_manager.record_trade(signal_event.symbol, strategy_id="RISK_MANAGER")
        logger.info(f"❄️ Cooldown recorded for {signal_event.symbol}")
        
        phase = safe_leverage_calculator.get_phase(self.portfolio.get_total_equity() if self.portfolio else safe_leverage_calculator.get_capital())
        logger.info(f"✅ {phase}: {direction.name} {quantity:.6f} {signal_event.symbol} (${dollar_size:.2f})")
        
        strategy_id = getattr(signal_event, 'strategy_id', None)
        
        # --- NEW: Dynamic ATR-based Exit Levels (Microscalping) ---
        atr_val = getattr(signal_event, 'atr', None)
        if atr_val is None:
            atr_val = (current_price * 0.01) if current_price else 0.0
            
        if current_price and current_price > 0:
            atr_pct = atr_val / current_price
        else:
            atr_pct = 0.01 # Fallback 1%
        
        # Multipliers based on regime (OVERRIDES)
        if self.current_regime == 'TRENDING':
            tp_mult = 3.0 # Let winners run
        elif self.current_regime == 'CHOPPY':
            tp_mult = 1.5 # Quick scalp
        else:
            tp_mult = 2.0
            
        # Dynamic Stop Calculation
        sl_pct = self._calculate_dynamic_stop_loss(atr_pct)
        
        # TP derived from SL (min 1.5 R:R)
        tp_pct = max(0.005, sl_pct * 1.5)
        
        # Adjust TP for Regime
        if self.current_regime == 'TRENDING':
             tp_pct = max(tp_pct, atr_pct * 4.0)
        
        # Log target info
        logger.info(f"📐 Target Sync: TP={tp_pct*100:.2f}%, SL={sl_pct*100:.2f}% | Regime: {self.current_regime}")
        
        # MICRO-SCALPING OPTIMIZATION: Use LIMIT Orders (Maker)
        # Entry requires patience.
        # Price: slightly inside spread to ensure Maker?
        # Ideally: Current Price. 'GTX' will reject if Taker.
        # We handle this in Executor.
        
        # Increment Daily Trades Counter
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_trade_logs: self.daily_trade_logs[today] = {}
        self.daily_trade_logs[today][signal_event.symbol] = self.daily_trade_logs[today].get(signal_event.symbol, 0) + 1
        
        return OrderEvent(
            symbol=signal_event.symbol, 
            order_type=OrderType.LIMIT, # FORCE LIMIT
            quantity=quantity, 
            direction=direction, 
            strategy_id=strategy_id,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            price=current_price, # Need price for Limit order
            ttl=getattr(signal_event, 'ttl', None)
        )

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

    def check_stops(self, portfolio, data_provider):
        """ORIGINAL COMPLETE check_stops - NO CHANGES"""
        stop_signals = []
        
        for symbol, pos in portfolio.positions.items():
            qty = pos['quantity']
            if qty == 0:
                continue
                
            current_price = pos.get('current_price')
            entry_price = pos.get('avg_price')
            
            if current_price is None or entry_price is None:
                continue
            if current_price == 0 or entry_price == 0:
                continue
            
            # LONG POSITION
            if qty > 0:
                hwm = pos.get('high_water_mark', entry_price)
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                peak_pnl_pct = ((hwm - entry_price) / entry_price) * 100
                
                if peak_pnl_pct >= 3.0:
                    gain_from_entry = hwm - entry_price
                    trail_distance = gain_from_entry * 0.1
                    stop_price = hwm - trail_distance
                    
                    if current_price < stop_price:
                        print(f"💰 TP3 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                        
                elif peak_pnl_pct >= 2.0:
                    gain_from_entry = hwm - entry_price
                    trail_distance = gain_from_entry * 0.25
                    stop_price = hwm - trail_distance
                    min_stop = entry_price * 1.015
                    stop_price = max(stop_price, min_stop)
                    
                    if current_price < stop_price:
                        print(f"💰 TP2 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                        
                elif peak_pnl_pct >= 1.0:
                    gain_from_entry = hwm - entry_price
                    trail_distance = gain_from_entry * 0.5
                    stop_price = hwm - trail_distance
                    breakeven_stop = entry_price * 1.003
                    stop_price = max(stop_price, breakeven_stop)
                    
                    if current_price < stop_price:
                        print(f"💰 TP1 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)

                elif unrealized_pnl_pct >= 0.6:
                    stop_price = entry_price * 1.004
                    
                    if current_price < stop_price:
                        print(f"⚡ Micro-Scalp {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                        
                else:
                    stop_distance = pos.get('stop_distance', current_price * 0.02)
                    stop_price = entry_price - stop_distance
                    
                    if current_price < stop_price:
                        print(f"🛑 Stop Loss {symbol}! {unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("RISK_MGR", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(False, unrealized_pnl_pct)
            
            # SHORT POSITION
            elif qty < 0:
                lwm = pos.get('low_water_mark', entry_price)
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                peak_pnl_pct = ((entry_price - lwm) / entry_price) * 100
                
                if peak_pnl_pct >= 3.0:
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.1
                    stop_price = lwm + trail_distance
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP3 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                
                elif peak_pnl_pct >= 2.0:
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.25
                    stop_price = lwm + trail_distance
                    min_stop = entry_price * 0.985
                    stop_price = min(stop_price, min_stop)
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP2 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                
                elif peak_pnl_pct >= 1.0:
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.5
                    stop_price = lwm + trail_distance
                    breakeven_stop = entry_price * 0.997
                    stop_price = min(stop_price, breakeven_stop)
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP1 {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                
                elif unrealized_pnl_pct >= 0.6:
                    stop_price = entry_price * 0.996
                    
                    if current_price > stop_price:
                        print(f"⚡ SHORT Micro-Scalp {symbol}! +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(True, unrealized_pnl_pct)
                
                else:
                    stop_distance = pos.get('stop_distance', current_price * 0.02)
                    stop_price = entry_price + stop_distance
                    
                    if current_price > stop_price:
                        print(f"🛑 SHORT Stop Loss {symbol}! {unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("RISK_MGR", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0)
                        stop_signals.append(sig)
                        self.record_trade_result(False, unrealized_pnl_pct)
                
        return stop_signals

    # ============================================================
    # KILL SWITCH FACADE
    # ============================================================
    
    def update_equity(self, equity):
        """Update equity in kill switch AND capital tracking."""
        # Update peak tracking first
        self._update_capital_tracking(equity)
        
        # Then update kill switch
        if self.kill_switch:
            self.kill_switch.update_equity(equity)
    def record_loss(self):
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
    
    def validate_sniper_order(self, symbol: str, quantity: float, 
                               entry_price: float, leverage: int) -> dict:
        notional = quantity * entry_price
        margin_required = notional / leverage
        MIN_NOTIONAL = 5.0
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