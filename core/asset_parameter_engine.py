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
        'last_price', 'last_calculated', 'data_points'
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
        self.last_price = 0.0
        self.last_calculated = 0.0
        self.data_points = 0
    
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
    # IMMUTABLE SAFETY CONSTRAINTS
    # These floors and ceilings can NEVER be violated.
    # POR QUÉ: Mathematical survival — R:R < 1.5:1 with WR < 65% = guaranteed ruin.
    # ═══════════════════════════════════════════════════════════════
    MIN_RR_RATIO = 1.5        # NEVER trade with R:R below this
    
    # Scalping bounds
    SCALPING_SL_MIN = 0.0015  # 0.15% — min SL (below this = noise death)
    SCALPING_SL_MAX = 0.008   # 0.80% — max SL (above this = too much risk)
    SCALPING_TP_MIN = 0.003   # 0.30% — min TP (below this = fee death)
    SCALPING_TP_MAX = 0.015   # 1.50% — max TP (above this = unreachable in M5)
    SCALPING_ATR_MULT = 0.40  # SL = 40% of ATR (tight but above noise)
    
    # Swing bounds
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
        logger.info("🧠 [AssetParamEngine] Initialized — Dynamic TP/SL calibration active")
    
    def get_profile(self, symbol: str) -> AssetProfile:
        """Get or create asset profile."""
        clean = symbol.replace("/", "").upper()
        if clean not in self._profiles:
            self._profiles[clean] = AssetProfile(clean)
        return self._profiles[clean]
    
    def get_tp(self, symbol: str, horizon: str = "SCALPING") -> float:
        """
        QUÉ: Retorna el TP óptimo para este activo y horizonte.
        PARA QUÉ: El RiskManager usa este valor en vez de un hardcode estático.
        """
        profile = self.get_profile(symbol)
        if horizon == "SWING":
            return profile.optimal_tp_swing
        return profile.optimal_tp_scalping
    
    def get_sl(self, symbol: str, horizon: str = "SCALPING") -> float:
        """
        QUÉ: Retorna el SL óptimo para este activo y horizonte.
        """
        profile = self.get_profile(symbol)
        if horizon == "SWING":
            return profile.optimal_sl_swing
        return profile.optimal_sl_scalping
    
    def get_params(self, symbol: str, horizon: str = "SCALPING") -> dict:
        """
        QUÉ: Retorna dict completo {tp_pct, sl_pct, atr_pct, rr_ratio} para un activo.
        PARA QUÉ: Compatibilidad con horizon_params dict en risk_manager.
        """
        profile = self.get_profile(symbol)
        tp = self.get_tp(symbol, horizon)
        sl = self.get_sl(symbol, horizon)
        return {
            "take_profit_pct": tp,
            "stop_loss_pct": sl,
            "atr_pct": profile.atr_14_pct,
            "rr_ratio": tp / sl if sl > 0 else self.MIN_RR_RATIO,
            "volatility_pct": profile.volatility_pct,
        }
    
    def calibrate_from_bars(self, symbol: str, closes: np.ndarray, 
                            highs: np.ndarray, lows: np.ndarray) -> AssetProfile:
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
            profile.avg_daily_range_pct = float(daily_range_pct)
            profile.volatility_pct = float(volatility_pct)
            profile.last_price = float(last_price)
            profile.data_points = n
            profile.last_calculated = time.time()
            
            # ═══════════════════════════════════════════════════════════════
            # 4. DERIVE OPTIMAL TP/SL
            # ═══════════════════════════════════════════════════════════════
            
            # --- SCALPING ---
            raw_sl_scalp = atr_14_pct * self.SCALPING_ATR_MULT
            sl_scalp = float(np.clip(raw_sl_scalp, self.SCALPING_SL_MIN, self.SCALPING_SL_MAX))
            
            # TP = SL * R:R ratio (minimum 1.5x, target 2.0x)
            raw_tp_scalp = sl_scalp * 2.0  # Target 2:1 R:R
            tp_scalp = float(np.clip(raw_tp_scalp, self.SCALPING_TP_MIN, self.SCALPING_TP_MAX))
            
            # Enforce minimum R:R after clamping
            if tp_scalp / sl_scalp < self.MIN_RR_RATIO:
                # Tighten SL to restore R:R
                sl_scalp = tp_scalp / self.MIN_RR_RATIO
                sl_scalp = float(np.clip(sl_scalp, self.SCALPING_SL_MIN, self.SCALPING_SL_MAX))
            
            profile.optimal_sl_scalping = sl_scalp
            profile.optimal_tp_scalping = tp_scalp
            
            # --- SWING ---
            raw_sl_swing = atr_14_pct * self.SWING_ATR_MULT
            sl_swing = float(np.clip(raw_sl_swing, self.SWING_SL_MIN, self.SWING_SL_MAX))
            
            raw_tp_swing = sl_swing * 2.0  # Target 2:1 R:R
            tp_swing = float(np.clip(raw_tp_swing, self.SWING_TP_MIN, self.SWING_TP_MAX))
            
            # Enforce minimum R:R after clamping
            if tp_swing / sl_swing < self.MIN_RR_RATIO:
                sl_swing = tp_swing / self.MIN_RR_RATIO
                sl_swing = float(np.clip(sl_swing, self.SWING_SL_MIN, self.SWING_SL_MAX))
            
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
    
    def calibrate_from_data_handler(self, symbol: str, data_handler=None) -> AssetProfile:
        """
        QUÉ: Calibra usando el data_handler del sistema (producción).
        POR QUÉ: En producción, los datos vienen del data_handler, no de parquet.
        """
        if data_handler is None:
            try:
                from core.data_handler import get_data_handler
                data_handler = get_data_handler()
            except ImportError:
                return self.get_profile(symbol)
        
        try:
            bars = data_handler.get_latest_bars(symbol, n=200)
            if bars is None or len(bars) < 20:
                return self.get_profile(symbol)
            
            closes = np.array(bars['close'], dtype=np.float64)
            highs = np.array(bars['high'], dtype=np.float64)
            lows = np.array(bars['low'], dtype=np.float64)
            
            return self.calibrate_from_bars(symbol, closes, highs, lows)
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
                
                closes = df['close'].values.astype(np.float64)
                highs = df['high'].values.astype(np.float64)
                lows = df['low'].values.astype(np.float64)
                
                # Filter out zeros/NaN
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
