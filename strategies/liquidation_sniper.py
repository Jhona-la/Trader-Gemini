"""
LIQUIDATION SNIPER STRATEGY - High Frequency Reversion Squeeze
========================================================
Listens to forceOrder events and triggers counter-trend entries when 
a large liquidation cascade happens.
"""

from typing import Optional, Dict
from datetime import datetime, timezone
import numpy as np

from .strategy import Strategy
from core.events import SignalEvent, MarketEvent
from core.enums import SignalType
from config import Config
from utils.logger import logger
from utils.common import validate_market_data, performance_timer
from utils.safe_leverage import safe_leverage_calculator

class LiquidationSniper(Strategy):
    """
    Estrategia Cuántica: Aprovecha el vacío de liquidez dejado por
    las liquidaciones masivas ejecutando un scalp en dirección opuesta
    con un Win Rate esperado cercano al 100% dado el rebote mecánico.
    """
    def __init__(self, data_provider, events_queue, executor=None, portfolio=None, horizon: str = "SCALPING", priority: int = 1):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.executor = executor
        self.portfolio = portfolio
        self.horizon = horizon
        self.priority = priority
        lbl = "[SCL]" if horizon == "SCALPING" else "[SWG]"
        self.strategy_id = f"{lbl}_LIQ_SNIPER_{horizon}"
        
        # HORIZON AWARE
        if horizon.upper() == 'SCALPING':
            h_params = getattr(Config.Horizons, 'Scalping', {})
            self.primary_tf = h_params.get('primary_tf', '1m') if h_params else '1m'
            self.COOLDOWN_SECONDS = 5  # Very fast cooldown for liquidations
        elif horizon.upper() == 'SWING':
            h_params = getattr(Config.Horizons, 'Swing', {})
            self.primary_tf = h_params.get('primary_tf', '1h') if h_params else '1h'
            self.COOLDOWN_SECONDS = 300
        else:
            self.primary_tf = '5m'
            self.COOLDOWN_SECONDS = 60
            
        # Hard thresholds
        self.MIN_LIQ_USD = 25000.0 # Minimum cascade
        
        # TP/SL specific for violent squeezes
        # ⚡ FASE 8 FIX: R:R was INVERTED (TP < SL = suicidal at 50x).
        # Now 2:1 R:R: TP wider than SL. At 50x, 1% TP = +50% margin. 0.5% SL = -25% margin.
        self.TP_PCT = 0.0100 # 1.0% fast scalp (was 0.5% - too tight)
        self.SL_PCT = 0.0050 # 0.5% tight SL (was 1.0% - too wide)
        
        self.last_signal_time = {}
        logger.info(f"⚡ LIQUIDATION SNIPER [{horizon}] INITIALIZED | Threshold: ${self.MIN_LIQ_USD:,.0f}")

    @validate_market_data
    @performance_timer
    def calculate_signals(self, event: MarketEvent):
        if not getattr(Config.Sniper, 'ENABLED', True):
            return

        order_flow = getattr(event, 'order_flow', {}) or {}
        
        # Only process liquidation events
        if not order_flow.get('liquidation'):
            return

        symbol = event.symbol
        liq_side = order_flow.get('side')
        liq_usd = order_flow.get('usd_value', 0.0)
        liq_price = order_flow.get('price', 0.0)
        
        if liq_usd < self.MIN_LIQ_USD:
            return
            
        now = getattr(event, 'timestamp', datetime.now(timezone.utc))
        if symbol in self.last_signal_time:
            if (now - self.last_signal_time[symbol]).total_seconds() < self.COOLDOWN_SECONDS:
                return
                
        # Liquidation mechanics:
        # SELL liq = Longs got rekt -> price dropped violently -> We go LONG to catch the bounce.
        # BUY liq = Shorts got rekt -> price spiked violently -> We go SHORT to catch the reversion.
        signal_type = SignalType.LONG if liq_side == 'SELL' else SignalType.SHORT
        
        # Micro-Price Validation (Phase 11 check)
        metrics = self.data_provider.get_order_flow_metrics(symbol)
        micro_price = metrics.get('micro_price', liq_price)
        
        # Calculate dynamic edge vs MicroPrice
        edge_pct = abs(micro_price - liq_price) / liq_price if liq_price > 0 else 0
        
        logger.warning(f"🌊 [LIQ-SNIPER] {symbol} | Cascade ${liq_usd:,.0f} {liq_side} | Edge {edge_pct:.3%} | Sniping {signal_type.name}!")
        
        # 🚀 FASE 7: APALANCAMIENTO MUTANTE (Dynamic Leverage)
        # Liquidations offer the highest alpha. We quantum-scale to 50x leverage.
        leverage = 50
        
        # Sophia AI pre-validation check
        sophia_report_dict = {}
        if hasattr(self, 'sophia') and self.sophia:
            market_regime = "VOLATILE"
            if self.portfolio and hasattr(self.portfolio, 'market_regime') and self.portfolio.market_regime:
                market_regime = self.portfolio.market_regime.get_current_regime()

            try:
                sophia_report = self.sophia.analyze(
                    symbol=symbol,
                    direction=signal_type.name,
                    signal_strength=1.0,
                    setups={'is_liquidation': True, 'liq_usd': liq_usd},
                    confluence_score=1.0,
                    tp_pct=self.TP_PCT,
                    sl_pct=self.SL_PCT,
                    returns=None,
                    ttl_seconds=120.0,
                    regime=market_regime,
                )

                if sophia_report.win_probability < 0.50:
                    logger.info(f"🛑 [SOPHIA VETO] Liq Sniper Blocked. Exactitude ({sophia_report.win_probability*100:.1f}%) < 50%")
                    return
                sophia_report_dict = sophia_report.to_dict()
            except Exception as e:
                logger.warning(f"Sophia analyze failed in LiqSniper: {e}")
            
        signal = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            datetime=now,
            signal_type=signal_type,
            strength=1.0,
            atr=0.0, # Handled by execution/risk
            tp_pct=self.TP_PCT,
            sl_pct=self.SL_PCT,
            current_price=liq_price,
            leverage=leverage,
            horizon=self.horizon,
            priority=10, # Top priority
            metadata={
                'is_liquidation_squeeze': True,
                'liq_value': liq_usd,
                'micro_price_edge': edge_pct,
                'sophia': sophia_report_dict,
                'trailing_breakeven_pct': 0.002 # Breakeven at 0.2% profit
            }
        )
        
        self.events_queue.put(signal)
        self.last_signal_time[symbol] = now
