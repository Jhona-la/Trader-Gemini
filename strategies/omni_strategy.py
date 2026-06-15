import numpy as np
import time
from core.events import SignalEvent
from core.enums import SignalType
from config import Config
from strategies.strategy import Strategy
from utils.logger import logger
from datetime import datetime, timezone
from utils.math_kernel import (
    calculate_rsi_jit, calculate_bollinger_robust_jit, calculate_atr_jit
)
# Intento cargar ML Oracle
try:
    from sophia.intelligence import MultiHorizonOracle
except ImportError:
    MultiHorizonOracle = None

class OmniStrategy(Strategy):
    """
    Estrategia Maestra de Producción (FASE 32).
    Refleja el 'Binomio Perfecto' del Motor Cuántico.
    Fusiona Technical, ML, Phalanx y StatArb usando los pesos del ADN Genético.
    """
    def __init__(self, data_provider, events_queue, genotype=None, horizon="SCALPING", priority=1):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.horizon = horizon
        self.strategy_id = f"[OMNI]_{horizon}"
        self.genotype = genotype
        self.priority = priority
        
        # DNA / Config Weights (Fase 31/32)
        omni_config = getattr(Config, 'OmniScore', None)
        if omni_config is None:
            # Fallback for mock environments
            omni_config = type('OmniScore', (), {})
            
        self.w_tech = getattr(omni_config, 'w_technical', 1.0)
        self.w_ml = getattr(omni_config, 'w_ml', 1.0)
        self.w_phalanx = getattr(omni_config, 'w_phalanx', 0.5)
        self.w_statarb = getattr(omni_config, 'w_statarb', 0.5)
        self.master_threshold = getattr(omni_config, 'master_threshold', 1.5)
        self.ml_bull_th = getattr(omni_config, 'ml_threshold_bull', 0.55)
        self.ml_bear_th = getattr(omni_config, 'ml_threshold_bear', 0.55)
        self.fee_mult = getattr(omni_config, 'consensus_fee_mult', 2.0)
        
        # Parameters
        if horizon == "SCALPING":
            h_params = getattr(Config.Horizons, 'Scalping', {})
        elif horizon == "SWING":
            h_params = getattr(Config.Horizons, 'Swing', {})
        else:
            h_params = {}
            
        self.rsi_period = h_params.get('rsi_period', 14)
        self.rsi_buy = h_params.get('rsi_buy', 30)
        self.rsi_sell = h_params.get('rsi_sell', 70)
        self.bb_period = h_params.get('bb_period', 20)
        self.bb_std = h_params.get('bb_std', 2.0)
        
        self.oracle = MultiHorizonOracle() if MultiHorizonOracle else None

    def calculate_signals(self, event):
        """Calcula el Omni-Score en vivo sobre el nuevo MarketEvent."""
        if not event.symbol:
            return
            
        symbol = event.symbol
        try:
            # 1. Obtener Data
            # Map horizon to timeframe for data retrieval
            _tf_map = {"SCALPING": "1m", "SWING": "5m"}
            tf = _tf_map.get(self.horizon, "1m")
            bars = self.data_provider.get_latest_bars(symbol, n=100, timeframe=tf)
            if bars is None or len(bars) < 50:
                return
            
            # Compatible con BacktestDataProvider (numpy structured array) Y producción (pandas DataFrame)
            _get = lambda arr, key: arr[key].values if hasattr(arr[key], 'values') else np.asarray(arr[key])
            close = _get(bars, 'close').astype(np.float32)
            high = _get(bars, 'high').astype(np.float32)
            low = _get(bars, 'low').astype(np.float32)
            open_p = _get(bars, 'open').astype(np.float32)
            
            current_price = close[-1]
            
            # 2. INDICADORES TÉCNICOS
            rsi_arr = calculate_rsi_jit(close, self.rsi_period)
            bbu_arr, bbm_arr, bbl_arr = calculate_bollinger_robust_jit(close, self.bb_period, self.bb_std)
            atr_arr = calculate_atr_jit(high, low, close, 14)
            
            rsi = rsi_arr[-1]
            bbl = bbl_arr[-1]
            bbu = bbu_arr[-1]
            atr_pct = atr_arr[-1] / current_price
            
            fee_threshold = (Config.BINANCE_MAKER_FEE_BNB + Config.BINANCE_TAKER_FEE_BNB) * self.fee_mult
            valid_volatility = (atr_pct >= fee_threshold)
            
            if not valid_volatility:
                return # Veto absoluto por volatilidad
            
            tech_long = 1.0 if (rsi < self.rsi_buy and current_price <= bbl) else 0.0
            tech_short = 1.0 if (rsi > self.rsi_sell and current_price >= bbu) else 0.0
            
            # 3. INTELIGENCIA ARTIFICIAL (ML)
            ml_bull, ml_bear = 0.0, 0.0
            if self.oracle:
                try:
                    pred = self.oracle.predict_live(symbol, self.horizon, bars)
                    if pred:
                        ml_bull = pred.get('P_BULL', 0.0)
                        ml_bear = pred.get('P_BEAR', 0.0)
                except Exception as e:
                    pass
            
            ml_long_sig = 1.0 if (ml_bull >= self.ml_bull_th) else 0.0
            ml_short_sig = 1.0 if (ml_bear >= self.ml_bear_th) else 0.0
            
            # 4. PROXIES SECUNDARIOS (Live)
            # StatArb Z-Score
            sma_50 = np.mean(close[-50:])
            std_50 = np.std(close[-50:])
            z_score = (current_price - sma_50) / (std_50 + 1e-9)
            statarb_long = 1.0 if z_score <= -2.5 else 0.0
            statarb_short = 1.0 if z_score >= 2.5 else 0.0
            
            # Phalanx
            mean_atr = np.mean(atr_arr[-20:])
            phalanx_long = 1.0 if (atr_pct > (mean_atr/current_price)*1.5 and close[-1] > open_p[-1]) else 0.0
            phalanx_short = 1.0 if (atr_pct > (mean_atr/current_price)*1.5 and close[-1] < open_p[-1]) else 0.0
            
            # 5. OMNI-SCORE FUSION
            score_long = (tech_long * self.w_tech) + (ml_long_sig * self.w_ml) + (phalanx_long * self.w_phalanx) + (statarb_long * self.w_statarb)
            score_short = (tech_short * self.w_tech) + (ml_short_sig * self.w_ml) + (phalanx_short * self.w_phalanx) + (statarb_short * self.w_statarb)
            
            # 6. GATILLO DE SEÑAL MAESTRA
            now_utc = datetime.now(timezone.utc)
            norm_score = lambda s: s / max(0.1, (self.w_tech + self.w_ml + self.w_phalanx + self.w_statarb))
            
            # Inject dynamic TP/SL
            h_params = getattr(Config.Horizons, 'Scalping' if self.horizon == 'SCALPING' else 'Swing', {})
            tp_pct = h_params.get('tp_pct', 0.02)
            sl_pct = h_params.get('sl_pct', 0.01)

            if score_long >= self.master_threshold:
                sig = SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strategy_id=self.strategy_id,
                    datetime=now_utc,
                    strength=min(score_long, 3.0),
                    ml_confidence=norm_score(score_long),
                    current_price=current_price,
                    horizon=self.horizon,
                    metadata={'setup_type': 'MOMENTUM', 'omni_score': score_long, 'tp_pct': tp_pct, 'sl_pct': sl_pct}
                )
                self.events_queue.put(sig)
                logger.info(f"🟢 [OMNI-SCORE] {symbol} {self.horizon} LONG | Score: {score_long:.2f} >= {self.master_threshold} (T:{tech_long} M:{ml_long_sig} P:{phalanx_long} S:{statarb_long})")
                
            elif score_short >= self.master_threshold:
                sig = SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strategy_id=self.strategy_id,
                    datetime=now_utc,
                    strength=min(score_short, 3.0),
                    ml_confidence=norm_score(score_short),
                    current_price=current_price,
                    horizon=self.horizon,
                    metadata={'setup_type': 'MOMENTUM', 'omni_score': score_short, 'tp_pct': tp_pct, 'sl_pct': sl_pct}
                )
                self.events_queue.put(sig)
                logger.info(f"🔴 [OMNI-SCORE] {symbol} {self.horizon} SHORT | Score: {score_short:.2f} >= {self.master_threshold} (T:{tech_short} M:{ml_short_sig} P:{phalanx_short} S:{statarb_short})")
                
        except Exception as e:
            logger.error(f"[OmniStrategy] Error in calculate_signals para {symbol}: {e}")
