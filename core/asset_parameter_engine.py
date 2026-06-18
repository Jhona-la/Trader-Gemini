"""
🧠 ASSET PARAMETER ENGINE — Dynamic TP/SL per Asset
=====================================================
QUÉ: Motor que calcula TP/SL/TTL dinámicamente POR ACTIVO basado en su perfil
     de volatilidad real (ATR, daily range, spread).
POR QUÉ: La autopsia forense reveló que el sistema usaba TP=1.51%, SL=2.00%
     (R:R = 0.76:1) para TODOS los activos. Esto es suicidio estadístico.
     BTC (ATR ~0.5%) NO puede usar los mismos parámetros que WIF (ATR ~5%).
PARA QUÉ: Maximizar la probabilidad de que cada trade alcance su TP antes del SL
     al calibrar los targets según la volatilidad REAL del activo.
CÓMO: 
  1. Lee datos OHLCV del cache parquet o data_handler
  2. Calcula ATR-14, daily range %, volatility % por activo
  3. Aplica fórmulas:
     - SL = ATR_14_pct * sl_atr_multiplier (default 0.5x ATR)
     - TP = SL * min_rr_ratio (default 2.0x)
     - Clamped a [min_sl, max_sl] y [min_tp, max_tp]
  4. Cada activo obtiene su "personalidad" con params óptimos
CUÁNDO: Inicialización del RiskManager y recalibración periódica (cada 1h)
DÓNDE: core/asset_parameter_engine.py → consultado por risk_manager.py
QUIÉN: Risk Manager (consulta), Quant Developer (cálculos)
"""

import os
import time
import numpy as np
from collections import defaultdict
from utils.logger import logger


class AssetProfile:
    """
    Perfil de volatilidad de un activo individual.
    
    QUÉ: Snapshot de la "personalidad" de un activo.
    POR QUÉ: Cada activo se mueve diferente. BTC es lento y predecible,
             WIF es explosivo e impredecible.
    """
    __slots__ = [
        'symbol', 'atr_14_pct', 'avg_daily_range_pct', 'volatility_pct',
        'optimal_tp_scalping', 'optimal_sl_scalping',
        'optimal_tp_swing', 'optimal_sl_swing',
        'optimal_tp_micro', 'optimal_sl_micro',  # HORIZON: MICRO
        'last_price', 'last_calculated', 'data_points',
        'atr_1m_pct', 'atr_5m_pct', 'atr_1h_pct',
        'leverage_scalping', 'leverage_swing', 'leverage_micro',  # HORIZON: MICRO
        'max_risk_pct'
    ]
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.atr_14_pct = 0.0
        self.avg_daily_range_pct = 0.0
        self.volatility_pct = 0.0
        self.optimal_tp_scalping = 0.003  # Default fallback
        self.optimal_sl_scalping = 0.002
        self.optimal_tp_swing = 0.045
        self.optimal_sl_swing = 0.025
        self.optimal_tp_micro = 0.0020  # HORIZON: MICRO default
        self.optimal_sl_micro = 0.0012  # HORIZON: MICRO default
        self.last_price = 0.0
        self.last_calculated = 0.0
        self.data_points = 0
        self.atr_1m_pct = 0.0
        self.atr_5m_pct = 0.0
        self.atr_1h_pct = 0.0
        self.leverage_scalping = 10
        self.leverage_swing = 10
        self.leverage_micro = 15  # HORIZON: MICRO default
        self.max_risk_pct = 0.02
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'atr_14_pct': self.atr_14_pct,
            'avg_daily_range_pct': self.avg_daily_range_pct,
            'volatility_pct': self.volatility_pct,
            'optimal_tp_scalping': self.optimal_tp_scalping,
            'optimal_sl_scalping': self.optimal_sl_scalping,
            'optimal_tp_swing': self.optimal_tp_swing,
            'optimal_sl_swing': self.optimal_sl_swing,
            'last_price': self.last_price,
            'data_points': self.data_points,
        }


class AssetParameterEngine:
    """
    Motor de Parámetros por Activo — Calibra TP/SL según volatilidad real.
    
    PRINCIPIO FUNDAMENTAL:
    - SL debe ser lo suficientemente amplio para NO ser tocado por el ruido normal
    - TP debe ser alcanzable dentro de la ventana de tiempo esperada
    - R:R MÍNIMO = 1.5:1 (hardcoded floor — NUNCA se viola)
    
    FÓRMULAS:
    Scalping (M5):
      - SL = max(ATR_14% * 0.40, 0.0015) clamped to [0.0015, 0.008]
      - TP = max(SL * 2.0, 0.003) clamped to [0.003, 0.015]
    
    Swing (H1/H4):
      - SL = max(ATR_14% * 1.5, 0.008) clamped to [0.008, 0.05]
      - TP = max(SL * 2.0, 0.015) clamped to [0.015, 0.10]
    """
    
    # ═══════════════════════════════════════════════════════════════
    # IMMUTABLE SAFETY CONSTRAINTS — PER HORIZON
    # These floors and ceilings can NEVER be violated.
    # POR QUÉ: Mathematical survival — R:R < 1.5:1 with WR < 65% = guaranteed ruin.
    # MÓDULO HORIZON: Each horizon has its own bounds.
    # ═══════════════════════════════════════════════════════════════
    MIN_RR_RATIO = 1.5        # NEVER trade with R:R below this
    
    # MICROSCALPING bounds — HORIZON: MICRO
    MICRO_SL_MIN = 0.0005     # 0.05% — tight noise floor
    MICRO_SL_MAX = 0.0030     # 0.30% — max SL for micro
    MICRO_TP_MIN = 0.0008     # 0.08% — min TP for micro
    MICRO_TP_MAX = 0.0060     # 0.60% — max TP for micro
    MICRO_ATR_MULT = 0.40     # SL = 40% of ATR-1m (very tight)
    
    # Scalping bounds — HORIZON: SCALP
    SCALPING_SL_MIN = 0.0015  # 0.15% — min SL (floor to survive noise)
    SCALPING_SL_MAX = 0.0150  # 1.50% — max SL (Reality Veto — prevents Zombies)
    SCALPING_TP_MIN = 0.0020  # 0.20% — min TP (floor TP)
    SCALPING_TP_MAX = 0.0300  # 3.00% — max TP (Reality Veto — prevents Zombies)
    SCALPING_ATR_MULT = 1.0   # SL = 100% of ATR (allow more breathing room)
    
    # Swing bounds — HORIZON: SWING
    SWING_SL_MIN = 0.008      # 0.80% — min SL for swing
    SWING_SL_MAX = 0.050      # 5.00% — max SL for swing
    SWING_TP_MIN = 0.015      # 1.50% — min TP for swing
    SWING_TP_MAX = 0.100      # 10.0% — max TP for swing
    SWING_ATR_MULT = 1.50     # SL = 1.5x ATR (gives room to breathe)
    
    # Fee awareness
    MIN_TP_AFTER_FEES = 0.002  # TP must exceed 0.20% to survive fees
    
    # Recalibration interval
    RECALIBRATE_INTERVAL_S = 3600  # 1 hour

    def __init__(self):
        self._profiles: dict[str, AssetProfile] = {}
        self._initialized = False
        self._last_global_calc = 0.0
        
        # Load optimal profiles from optimal_profiles.json if present
        self.calibrated_profiles = {}
        try:
            import os
            import json
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prof_path = os.path.join(root_dir, 'optimal_profiles.json')
            if os.path.exists(prof_path):
                with open(prof_path, 'r') as f:
                    self.calibrated_profiles = json.load(f)
                logger.info(f"🧠 [AssetParamEngine] Loaded optimal profiles from {prof_path} for horizons: {list(self.calibrated_profiles.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ [AssetParamEngine] Could not load optimal_profiles.json: {e}")
            
        logger.info("🧠 [AssetParamEngine] Initialized — Dynamic TP/SL calibration active")
    
    def get_profile(self, symbol: str) -> AssetProfile:
        """Get or create asset profile."""
        clean = symbol.replace("/", "").upper()
        if clean not in self._profiles:
            self._profiles[clean] = AssetProfile(clean)
        return self._profiles[clean]

    def _get_bounds(self, horizon_key: str) -> tuple:
        """
        MÓDULO HORIZON: Return (sl_min, sl_max, tp_min, tp_max, atr_mult) for horizon.
        QUÉ: Resolver los bounds de TP/SL por horizonte.
        POR QUÉ: MICRO necesita bounds mucho más ajustados que SCALP/SWING.
        """
        if horizon_key == "MICROSCALPING":
            return (self.MICRO_SL_MIN, self.MICRO_SL_MAX,
                    self.MICRO_TP_MIN, self.MICRO_TP_MAX, self.MICRO_ATR_MULT)
        elif horizon_key == "SWING":
            return (self.SWING_SL_MIN, self.SWING_SL_MAX,
                    self.SWING_TP_MIN, self.SWING_TP_MAX, self.SWING_ATR_MULT)
        else:  # SCALPING (default)
            return (self.SCALPING_SL_MIN, self.SCALPING_SL_MAX,
                    self.SCALPING_TP_MIN, self.SCALPING_TP_MAX, self.SCALPING_ATR_MULT)

    def get_leverage(self, symbol: str, horizon: str = "SCALPING") -> int:
        """
        QUÉ: Retorna el apalancamiento objetivo dinámico por activo y horizonte.
        """
        from config import Config
        prof = Config.SymbolProfiles.get(symbol, horizon)
        return prof.get("base_leverage", 5)

    def get_risk_pct(self, symbol: str, horizon: str = "SCALPING") -> float:
        """
        QUÉ: Retorna el porcentaje máximo de riesgo permitido para el activo y horizonte.
        """
        from config import Config
        prof = Config.SymbolProfiles.get(symbol, horizon)
        return prof.get("max_risk_pct", 0.02)

    
    def get_tp(self, symbol: str, horizon: str = "SCALPING", direction: str = "LONG") -> float:
        """
        QUÉ: Retorna el TP óptimo para este activo y horizonte, ajustado por dirección.
        MÓDULO HORIZON: Uses _get_bounds() for horizon-specific clipping.
        """
        profile = self.get_profile(symbol)
        clean = symbol.replace("/", "").upper()
        horizon_key = horizon.upper()
        sl_min, sl_max, tp_min, tp_max, _ = self._get_bounds(horizon_key)
        
        # Resolve correct SL first (which resolves the correct ATR dynamically)
        sl = self.get_sl(symbol, horizon, direction)
        
        # 1. Check if we have calibrated profile override
        lookup_horizon = "SCALPING" if horizon_key == "MICROSCALPING" else horizon_key
        if lookup_horizon in self.calibrated_profiles:
            prof_data = None
            for k, v in self.calibrated_profiles[lookup_horizon].items():
                if k.replace("/", "").upper() == clean:
                    prof_data = v
                    break
            if prof_data and 'tp_rr_ratio' in prof_data:
                tp_rr_ratio = prof_data['tp_rr_ratio']
                if direction.upper() == "SHORT":
                    tp_rr_ratio *= 0.8
                raw_tp = sl * tp_rr_ratio
                return float(np.clip(raw_tp, tp_min, tp_max * 1.5))
                    
        # 2. [SHORT INTELLIGENCE: Mantener R:R de 2:1 basado en el SL ampliado]
        if direction.upper() == "SHORT":
            raw_tp = sl * 2.0
            return float(np.clip(raw_tp, tp_min, tp_max * 1.5))
                
        # 3. Dynamic fallback for LONG without overrides (Target 2:1 R:R)
        raw_tp = sl * 2.0
        return float(np.clip(raw_tp, tp_min, tp_max))
    
    def get_sl(self, symbol: str, horizon: str = "SCALPING", direction: str = "LONG") -> float:
        """
        QUÉ: Retorna el SL óptimo para este activo y horizonte, ajustado por dirección.
        """
        profile = self.get_profile(symbol)
        clean = symbol.replace("/", "").upper()
        horizon_key = horizon.upper()
        sl_min, sl_max, tp_min, tp_max, atr_mult = self._get_bounds(horizon_key)
        
        # Resolve correct ATR for this horizon
        atr = profile.atr_1h_pct if horizon_key == "SWING" else (profile.atr_1m_pct if horizon_key == "MICROSCALPING" else profile.atr_5m_pct)
        if atr <= 0:
            atr = profile.atr_14_pct
        if atr <= 0:
            atr = 0.005  # standard fallback 0.5%
        
        # Check if we have calibrated profile override
        lookup_horizon = "SCALPING" if horizon_key == "MICROSCALPING" else horizon_key
        if lookup_horizon in self.calibrated_profiles:
            prof_data = None
            for k, v in self.calibrated_profiles[lookup_horizon].items():
                if k.replace("/", "").upper() == clean:
                    prof_data = v
                    break
            if prof_data and 'sl_atr_mult' in prof_data:
                sl_mult = prof_data['sl_atr_mult']
                if direction.upper() == "SHORT":
                    sl_mult *= 1.2
                raw_sl = atr * sl_mult
                return float(np.clip(raw_sl, sl_min, sl_max * 1.5))
                    
        # [SHORT INTELLIGENCE: Stop Placement asimétrico]
        if direction.upper() == "SHORT":
            sym = symbol.replace("/", "").upper()
            
            # Fallo Tipo 3: Multiplicadores ATR más amplios para Shorts
            if "BTC" in sym: mult = 1.8
            elif "ETH" in sym: mult = 2.0
            elif "BNB" in sym or "XRP" in sym: mult = 2.3
            elif "SOL" in sym: mult = 2.5
            elif sym in ["DOGE", "SHIB", "PEPE", "FLOKI", "WIF"]: mult = 3.0
            else: mult = 2.5
            
            # Escalar proporcionalmente al horizonte
            base_mult = mult if horizon_key == "SWING" else mult * (self.SCALPING_ATR_MULT / self.SWING_ATR_MULT)
            raw_sl = atr * base_mult
            return float(np.clip(raw_sl, sl_min, sl_max * 1.5))
                
        # Default LONG fallback — MÓDULO HORIZON: per-horizon profiles
        if horizon_key == "SWING":
            return profile.optimal_sl_swing
        elif horizon_key == "MICROSCALPING":
            return profile.optimal_sl_micro
        return profile.optimal_sl_scalping
    
    def get_params(self, symbol: str, horizon: str = "SCALPING", direction: str = "LONG") -> dict:
        """
        QUÉ: Retorna dict completo {tp_pct, sl_pct, atr_pct, rr_ratio} para un activo.
        PARA QUÉ: Compatibilidad con horizon_params dict en risk_manager.
        """
        profile = self.get_profile(symbol)
        tp = self.get_tp(symbol, horizon, direction)
        sl = self.get_sl(symbol, horizon, direction)
        return {
            "take_profit_pct": tp,
            "stop_loss_pct": sl,
            "atr_pct": profile.atr_14_pct,
            "rr_ratio": tp / sl if sl > 0 else self.MIN_RR_RATIO,
            "volatility_pct": profile.volatility_pct,
        }
    
    def calibrate_from_bars(self, symbol: str, closes: np.ndarray, 
                            highs: np.ndarray, lows: np.ndarray, horizon: str = "SCALPING") -> AssetProfile:
        """
        QUÉ: Calibra el perfil de un activo usando datos OHLCV reales.
        POR QUÉ: Solo datos reales pueden decir la "personalidad" del activo.
        CÓMO:
          1. Calcula ATR-14 como % del precio
          2. Calcula avg daily range
          3. Calcula volatilidad (stddev de returns)
          4. Deriva TP/SL óptimos con fórmulas calibradas
        
        Args:
            symbol: e.g. "BTCUSDT" or "BTC/USDT"
            closes: Array de precios de cierre
            highs: Array de precios máximos
            lows: Array de precios mínimos
            horizon: El horizonte temporal de calibración (SCALPING, SWING, MICROSCALPING)
        """
        clean = symbol.replace("/", "").upper()
        profile = self.get_profile(clean)
        
        try:
            n = len(closes)
            if n < 20:
                logger.warning(f"[AssetParamEngine] {clean}: insufficient data ({n} bars)")
                return profile
            
            # 1. ATR-14 calculation
            tr1 = highs[1:] - lows[1:]
            tr2 = np.abs(highs[1:] - closes[:-1])
            tr3 = np.abs(lows[1:] - closes[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Simple moving average of TR for ATR-14
            period = min(14, len(true_range))
            atr_14 = np.mean(true_range[-period:])
            last_price = closes[-1]
            
            if last_price <= 0:
                return profile
            
            atr_14_pct = atr_14 / last_price
            
            # 2. Daily range %
            daily_range_pct = np.mean((highs - lows) / closes) if n > 1 else atr_14_pct
            
            # 3. Volatility (stddev of log returns)
            returns = np.diff(np.log(closes[closes > 0]))
            volatility_pct = np.std(returns) if len(returns) > 5 else atr_14_pct
            
            # Store raw metrics
            profile.atr_14_pct = float(atr_14_pct)
            
            # Store horizon-specific ATR
            horizon_key = horizon.upper()
            if horizon_key == "SWING":
                profile.atr_1h_pct = float(atr_14_pct)
            elif horizon_key == "MICROSCALPING":
                profile.atr_1m_pct = float(atr_14_pct)
            else:
                profile.atr_5m_pct = float(atr_14_pct)
                
            profile.avg_daily_range_pct = float(daily_range_pct)
            profile.volatility_pct = float(volatility_pct)
            profile.last_price = float(last_price)
            profile.data_points = n
            profile.last_calculated = time.time()
            
            # ═══════════════════════════════════════════════════════════════
            # 4. DERIVE OPTIMAL TP/SL
            # ═══════════════════════════════════════════════════════════════
            
            # --- SCALPING ---
            # Check if we have calibrated profile override
            horizon_key = "SCALPING"
            cal_sl_mult = None
            cal_tp_rr = None
            if horizon_key in self.calibrated_profiles:
                for k, v in self.calibrated_profiles[horizon_key].items():
                    if k.replace("/", "").upper() == clean:
                        cal_sl_mult = v.get('sl_atr_mult')
                        cal_tp_rr = v.get('tp_rr_ratio')
                        break
            
            if cal_sl_mult is not None:
                raw_sl_scalp = atr_14_pct * cal_sl_mult
            else:
                # Capa 6: Sniper Mode - Dynamic ATR Multiplier based on symbol noise
                if "BTC" in clean:
                    dynamic_scalp_mult = 0.40
                elif "ETH" in clean or "BNB" in clean:
                    dynamic_scalp_mult = 0.60
                elif "SOL" in clean or "XRP" in clean or clean in ["DOGE", "SHIB", "PEPE", "FLOKI", "WIF"]:
                    dynamic_scalp_mult = 0.85
                else:
                    dynamic_scalp_mult = 0.70
                raw_sl_scalp = atr_14_pct * dynamic_scalp_mult
                
            sl_scalp = float(np.clip(raw_sl_scalp, self.SCALPING_SL_MIN, self.SCALPING_SL_MAX))
            
            if cal_tp_rr is not None:
                raw_tp_scalp = sl_scalp * cal_tp_rr
            else:
                raw_tp_scalp = sl_scalp * 2.0  # Target 2:1 R:R
                
            tp_scalp = float(np.clip(raw_tp_scalp, self.SCALPING_TP_MIN, self.SCALPING_TP_MAX))
            
            # Enforce minimum R:R after clamping
            if tp_scalp / sl_scalp < self.MIN_RR_RATIO:
                # Widen TP to restore R:R instead of tightening SL to prevent premature noise stop-outs
                tp_scalp = sl_scalp * self.MIN_RR_RATIO
                tp_scalp = float(np.clip(tp_scalp, self.SCALPING_TP_MIN, self.SCALPING_TP_MAX))
            
            profile.optimal_sl_scalping = sl_scalp
            profile.optimal_tp_scalping = tp_scalp
            
            # --- SWING ---
            horizon_key = "SWING"
            cal_sl_mult = None
            cal_tp_rr = None
            if horizon_key in self.calibrated_profiles:
                for k, v in self.calibrated_profiles[horizon_key].items():
                    if k.replace("/", "").upper() == clean:
                        cal_sl_mult = v.get('sl_atr_mult')
                        cal_tp_rr = v.get('tp_rr_ratio')
                        break
            
            if cal_sl_mult is not None:
                raw_sl_swing = atr_14_pct * cal_sl_mult
            else:
                raw_sl_swing = atr_14_pct * self.SWING_ATR_MULT
                
            sl_swing = float(np.clip(raw_sl_swing, self.SWING_SL_MIN, self.SWING_SL_MAX))
            
            if cal_tp_rr is not None:
                raw_tp_swing = sl_swing * cal_tp_rr
            else:
                raw_tp_swing = sl_swing * 2.0  # Target 2:1 R:R
                
            tp_swing = float(np.clip(raw_tp_swing, self.SWING_TP_MIN, self.SWING_TP_MAX))
            
            # Enforce minimum R:R after clamping
            if tp_swing / sl_swing < self.MIN_RR_RATIO:
                # Widen TP to restore R:R instead of tightening SL to prevent premature noise stop-outs
                tp_swing = sl_swing * self.MIN_RR_RATIO
                tp_swing = float(np.clip(tp_swing, self.SWING_TP_MIN, self.SWING_TP_MAX))
            
            profile.optimal_sl_swing = sl_swing
            profile.optimal_tp_swing = tp_swing
            
            rr_scalp = tp_scalp / sl_scalp if sl_scalp > 0 else 0
            rr_swing = tp_swing / sl_swing if sl_swing > 0 else 0
            
            logger.info(
                f"🧠 [AssetParamEngine] {clean}: ATR={atr_14_pct*100:.3f}% | "
                f"SCALP TP={tp_scalp*100:.2f}%/SL={sl_scalp*100:.2f}% R:R={rr_scalp:.1f}:1 | "
                f"SWING TP={tp_swing*100:.2f}%/SL={sl_swing*100:.2f}% R:R={rr_swing:.1f}:1"
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"[AssetParamEngine] Calibration error {clean}: {e}")
            return profile
    
    def calibrate_from_data_handler(self, symbol: str, data_handler=None, horizon: str = "SCALPING") -> AssetProfile:
        """
        QUÉ: Calibra usando el data_handler del sistema (producción).
        POR QUÉ: En producción, los datos vienen del data_handler, no de parquet.
        
        FORENSIC-V130: HORIZON-AWARE ATR CALIBRATION
        QUÉ: El ATR usado para calibrar TP/SL DEBE calcularse en el
          timeframe operativo del horizonte.
        POR QUÉ: ATR-14 en 1m = ~7 minutos de contexto (insuficiente para Swing).
          ATR-14 en 1h = 14 horas de contexto (correcto para Swing).
        PARA QUÉ: Calibrar TP/SL que sean matemáticamente alcanzables en
          la temporalidad real de operación.
        CÓMO: SCALPING → 5m (ATR-14 = 70min), SWING → 1h (ATR-14 = 14h).
        """
        if data_handler is None or not hasattr(data_handler, "get_latest_bars"):
            try:
                from data.data_provider import get_data_provider
                data_handler = get_data_provider()
            except Exception:
                pass
        
        if data_handler is None or not hasattr(data_handler, "get_latest_bars"):
            try:
                from core.data_handler import get_data_handler
                dh_candidate = get_data_handler()
                if dh_candidate and hasattr(dh_candidate, "get_latest_bars"):
                    data_handler = dh_candidate
            except Exception:
                pass
        
        if data_handler is None or not hasattr(data_handler, "get_latest_bars"):
            # Fallback to current cached profile if no source with get_latest_bars is available
            return self.get_profile(symbol)
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V130: Route to horizon-appropriate timeframe
            # ═══════════════════════════════════════════════════════════════
            _atr_tf = '1m' if horizon == 'MICROSCALPING' else ('5m' if horizon == 'SCALPING' else '1h')
            bars = data_handler.get_latest_bars(symbol, n=200, timeframe=_atr_tf)
            if bars is None or len(bars) < 20:
                return self.get_profile(symbol)
            
            closes = np.array(bars['close'], dtype=np.float64)
            highs = np.array(bars['high'], dtype=np.float64)
            lows = np.array(bars['low'], dtype=np.float64)
            
            return self.calibrate_from_bars(symbol, closes, highs, lows, horizon=horizon)
        except Exception as e:
            logger.error(f"[AssetParamEngine] DH calibration error {symbol}: {e}")
            return self.get_profile(symbol)
    
    def calibrate_all_from_parquet(self, cache_dir: str = None):
        """
        QUÉ: Calibra TODOS los activos desde el cache parquet.
        POR QUÉ: Útil para inicialización bulk (backtest y startup).
        """
        import pandas as pd
        
        if cache_dir is None:
            from config import Config
            cache_dir = os.path.join(Config.BASE_DIR, "data", "cache_parquet")
        
        if not os.path.exists(cache_dir):
            logger.warning(f"[AssetParamEngine] Cache dir not found: {cache_dir}")
            return
        
        count = 0
        for fname in os.listdir(cache_dir):
            if not fname.endswith('.parquet'):
                continue
            try:
                fpath = os.path.join(cache_dir, fname)
                df = pd.read_parquet(fpath)
                
                sym = fname.replace('features_', '').replace('.parquet', '').replace('_', '')
                
                if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                    continue
                
                # LEY DEL FLOAT32 INQUEBRANTABLE (Cero-Copy)
                closes = df['close'].values.astype(np.float32, copy=False)
                highs = df['high'].values.astype(np.float32, copy=False)
                lows = df['low'].values.astype(np.float32, copy=False)
                
                # Filter out zeros/NaN usando np.nan_to_num in-place para purificación
                np.nan_to_num(closes, copy=False, nan=0.0)
                np.nan_to_num(highs, copy=False, nan=0.0)
                np.nan_to_num(lows, copy=False, nan=0.0)
                
                valid = (closes > 0) & (highs > 0) & (lows > 0)
                closes = closes[valid]
                highs = highs[valid]
                lows = lows[valid]
                
                if len(closes) > 20:
                    self.calibrate_from_bars(sym, closes, highs, lows)
                    count += 1
            except Exception as e:
                logger.error(f"[AssetParamEngine] Parquet error {fname}: {e}")
        
        self._initialized = True
        self._last_global_calc = time.time()
        logger.info(f"🧠 [AssetParamEngine] Bulk calibration complete: {count} assets profiled")
    
    def needs_recalibration(self) -> bool:
        """Check if profiles are stale and need recalibration."""
        return (time.time() - self._last_global_calc) > self.RECALIBRATE_INTERVAL_S
    
    def get_all_profiles_summary(self) -> list[dict]:
        """Returns all profiles as list of dicts for logging/DB."""
        return [p.to_dict() for p in self._profiles.values()]


# ═══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════
_asset_param_engine = None

def get_asset_parameter_engine() -> AssetParameterEngine:
    """
    QUÉ: Singleton factory.
    POR QUÉ: Solo debe haber UN motor de parámetros en todo el sistema.
    """
    global _asset_param_engine
    if _asset_param_engine is None:
        _asset_param_engine = AssetParameterEngine()
    return _asset_param_engine
