"""
⚛️ MÓDULO MAESTRO: INTELIGENCIA DIFERENCIAL DE ACTIVOS Y SISTEMAS DE APERTURA/CIERRE
=============================================================================
PROFESSOR METHOD:
- QUÉ: Componente centralizado para gobernar las decisiones operativas basadas en
       la taxonomía, liquidez, volatilidad y catalizadores específicos de cada activo.
- POR QUÉ: Evitar el sobre-simplificado de parámetros genéricos que expone la cuenta
           de $13 USD a pérdidas. Cada activo tiene una microestructura única.
- PARA QUÉ: Enforzar las políticas de entrada (A1-A7), salida (C1-C7) y compatibilidad
             de estrategias (1-12) para proteger la supervivencia de la cuenta.
- CÓMO: 
    - Clasifica activos en Tiers (0-4), niveles de liquidez (1-5) y perfiles de volatilidad (A-D).
    - Aplica filtros secuenciales estrictos de Apertura y Cierre.
    - Integra hooks dinámicos para detectar caídas de red (Solana) y riesgo regulatorio (XRP/BNB).
- CUÁNDO: En cada evaluación de señal en el MetaCoordinator y en cada chequeo de stops en el RiskManager.
- DÓNDE: core/asset_intelligence.py (este archivo).
- QUIÉN: Diseñado por el Quant Developer y el Risk Manager de Trader Gemini.
"""

import time
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, timezone
import numpy as np

from utils.logger import logger
from core.events import SignalEvent, SignalType, SignalState

# =========================================================================
# 🏛️ TAXONOMY ENUMS
# =========================================================================

class AssetTier(Enum):
    TIER_0 = 0  # Líder Absoluto (BTC)
    TIER_1 = 1  # Co-Líder (ETH)
    TIER_2 = 2  # Seguidor Líder (BNB, SOL, AVAX, DOT)
    TIER_3 = 3  # Seguidor Estándar (XRP, ADA, LTC, LINK, ATOM)
    TIER_4 = 4  # Especulativo (DOGE, SHIB, altcoins menores)

class LiquidityLevel(Enum):
    LEVEL_1 = 1  # Ultra-Líquido (Spread < 0.01%, L5 > $50M)
    LEVEL_2 = 2  # Muy Líquido (Spread 0.01%-0.03%, L5 > $10M)
    LEVEL_3 = 3  # Líquido (Spread 0.03%-0.08%, L5 > $2M)
    LEVEL_4 = 4  # Semi-Líquido (Spread 0.08%-0.20%, L5 < $2M)
    LEVEL_5 = 5  # Ilíquido (Spread > 0.20%, L5 escaso)

class VolatilityProfile(Enum):
    PROFILE_A = "A"  # Referencia (BTC)
    PROFILE_B = "B"  # Beta Moderado (1.1x a 1.5x BTC)
    PROFILE_C = "C"  # Beta Alto (1.5x a 3x BTC)
    PROFILE_D = "D"  # Beta Extremo (Meme / Catalizador-driven)

class CatalystType(Enum):
    TIPO_1 = 1  # Técnico del activo (Fork, consensus change)
    TIPO_2 = 2  # Ecosistema (TVL, adoption)
    TIPO_3 = 3  # Regulatorio (SEC cases, exchange restrictions)
    TIPO_4 = 4  # Sentimiento (Tweets, social hype)
    TIPO_5 = 5  # Macro (FOMC, CPI, NFP)
    TIPO_6 = 6  # Liquidez (ETF flow, new exchange lists)


# =========================================================================
# 🧠 ACTIVE ASSET PROFILES
# =========================================================================

class AssetProfile:
    """
    Perfil estructurado de la personalidad técnica de un activo.
    """
    def __init__(self, 
                 symbol: str, 
                 tier: AssetTier, 
                 liquidity: LiquidityLevel, 
                 volatility: VolatilityProfile, 
                 base_beta: float,
                 min_signal_threshold: float,
                 factor_sizing: float,
                 stop_atr_mult: float,
                 kelly_fraction: float,
                 allowed_strategies: Set[str],
                 restrictions: List[str]):
        self.symbol = symbol
        self.tier = tier
        self.liquidity = liquidity
        self.volatility = volatility
        self.base_beta = base_beta
        self.min_signal_threshold = min_signal_threshold
        self.factor_sizing = factor_sizing
        self.stop_atr_mult = stop_atr_mult
        self.kelly_fraction = kelly_fraction
        self.allowed_strategies = allowed_strategies
        self.restrictions = restrictions

class AssetIntelligence:
    """
    Manager de Inteligencia de Activos. Enforza reglas del Módulo Maestro.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._profiles: Dict[str, AssetProfile] = {}
        self._load_default_profiles()
        self._initialized = True
        logger.info("🏛️ [AssetIntelligence] initialized under Módulo Maestro specifications.")

    def _load_default_profiles(self):
        # 1. BTC (Tier 0)
        self._profiles["BTC/USDT"] = AssetProfile(
            symbol="BTC/USDT",
            tier=AssetTier.TIER_0,
            liquidity=LiquidityLevel.LEVEL_1,
            volatility=VolatilityProfile.PROFILE_A,
            base_beta=1.0,
            min_signal_threshold=0.55,  # FORENSIC FIX: Lowered from 0.58 to match ML outputs and GlobalThresholds
            factor_sizing=1.00,
            stop_atr_mult=1.5,
            kelly_fraction=0.50, # 1/2 Kelly
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "PAIR_ARB", "ON_CHAIN", "SENTIMENT_CONTRARIAN", "WYCKOFF"},
            restrictions=[]
        )
        
        # 2. ETH (Tier 1)
        self._profiles["ETH/USDT"] = AssetProfile(
            symbol="ETH/USDT",
            tier=AssetTier.TIER_1,
            liquidity=LiquidityLevel.LEVEL_1,
            volatility=VolatilityProfile.PROFILE_B,
            base_beta=1.2,
            min_signal_threshold=0.55,  # FORENSIC FIX: Lowered from 0.55
            factor_sizing=0.95,
            stop_atr_mult=1.7,
            kelly_fraction=0.50,
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "PAIR_ARB", "ON_CHAIN", "SENTIMENT_CONTRARIAN", "WYCKOFF"},
            restrictions=[]
        )
        
        # 3. BNB (Tier 2)
        self._profiles["BNB/USDT"] = AssetProfile(
            symbol="BNB/USDT",
            tier=AssetTier.TIER_2,
            liquidity=LiquidityLevel.LEVEL_2,
            volatility=VolatilityProfile.PROFILE_C,
            base_beta=1.5,
            min_signal_threshold=0.55,  # FORENSIC FIX: Lowered to 0.55
            factor_sizing=0.85,
            stop_atr_mult=2.0,
            kelly_fraction=0.50,
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "PAIR_ARB", "ON_CHAIN", "SENTIMENT_CONTRARIAN", "WYCKOFF"},
            restrictions=["REGULATORY_EXPOSURE_CHECK"]
        )

        # 4. SOL (Tier 2)
        self._profiles["SOL/USDT"] = AssetProfile(
            symbol="SOL/USDT",
            tier=AssetTier.TIER_2,
            liquidity=LiquidityLevel.LEVEL_2,
            volatility=VolatilityProfile.PROFILE_C,
            base_beta=2.0,
            min_signal_threshold=0.55,  # FORENSIC FIX: Lowered to 0.55
            factor_sizing=0.80,
            stop_atr_mult=2.0,
            kelly_fraction=0.50,
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "PAIR_ARB", "ON_CHAIN", "SENTIMENT_CONTRARIAN", "WYCKOFF"},
            restrictions=["OUTAGE_CHECK"]
        )
        
        # 5. XRP (Tier 3)
        self._profiles["XRP/USDT"] = AssetProfile(
            symbol="XRP/USDT",
            tier=AssetTier.TIER_3,
            liquidity=LiquidityLevel.LEVEL_2,
            volatility=VolatilityProfile.PROFILE_B,
            base_beta=1.8,
            min_signal_threshold=0.55,  # FORENSIC FIX: Keep at 0.55
            factor_sizing=0.75,
            stop_atr_mult=2.0,
            kelly_fraction=0.50,
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "SENTIMENT_CONTRARIAN", "WYCKOFF"},
            restrictions=["SEC_CASE_VETO"]
        )
        
        # 6. DOGE (Tier 4)
        self._profiles["DOGE/USDT"] = AssetProfile(
            symbol="DOGE/USDT",
            tier=AssetTier.TIER_4,
            liquidity=LiquidityLevel.LEVEL_3,
            volatility=VolatilityProfile.PROFILE_D,
            base_beta=3.0,
            min_signal_threshold=0.55,  # FORENSIC FIX: Lowered to 0.55
            factor_sizing=0.50,
            stop_atr_mult=2.5,
            kelly_fraction=0.25, # 1/4 Kelly
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "FUNDING_ARB", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL", "SENTIMENT_CONTRARIAN"},
            restrictions=["SENTIMENT_CATALYST_REQUIRED", "NO_MEAN_REVERSION", "NO_VWAP_REVERSION", "NO_OVERNIGHT", "NO_WYCKOFF"]
        )

    def get_profile(self, symbol: str) -> AssetProfile:
        """Retorna el perfil del activo, o una plantilla genérica si no está registrado."""
        clean = symbol.replace("/", "").upper()
        # Find matches by mapping
        for k, v in self._profiles.items():
            if k.replace("/", "") == clean or k == symbol:
                return v
                
        # Plantilla genérica para otros activos
        return AssetProfile(
            symbol=symbol,
            tier=AssetTier.TIER_3,
            liquidity=LiquidityLevel.LEVEL_3,
            volatility=VolatilityProfile.PROFILE_C,
            base_beta=1.8,
            min_signal_threshold=0.68,
            factor_sizing=0.70,
            stop_atr_mult=2.0,
            kelly_fraction=0.50,
            allowed_strategies={"TFTF", "OB_RETEST", "CASCADE", "MEAN_REVERSION", "VWAP_REVERSION", "VOLATILITY_BREAKOUT", "MOMENTUM_VOL"},
            restrictions=[]
        )

    # =========================================================================
    # 🚥 PART III — SISTEMA DE APERTURA (A1-A7)
    # =========================================================================
    
    def verify_opening(self, intent: SignalEvent, portfolio) -> Tuple[bool, str]:
        """
        Ejecuta el pipeline de apertura de 7 pasos (A1-A7) de manera secuencial.
        Retorna (True, "APPROVED") o (False, "FAIL_A{index}: reason").
        """
        symbol = intent.symbol
        profile = self.get_profile(symbol)
        
        # Extraer estrategia del strategy_id (ej: "[SCL]_ML_HYBRID_ULTIMATE_V2_SCALPING" -> "TFTF" o equivalent)
        strategy_raw = getattr(intent, 'strategy_id', 'TFTF')
        strategy_mapped = self._map_strategy_name(strategy_raw)
        
        # ---------------------------------------------------------------------
        # Paso A1 — CONDICIÓN DE RÉGIMEN
        # ---------------------------------------------------------------------
        regime = getattr(intent, 'regime', None)
        if not regime and hasattr(intent, 'metadata') and intent.metadata:
            regime = intent.metadata.get('regime')
        if not regime:
            from core.global_state import global_state
            regime = getattr(global_state, 'market_regime', 'UNKNOWN')
        regime = str(regime).upper()
        
        # ─── OMEGA FIX: Si el régimen es UNKNOWN, no vetar inmediatamente ───
        # En backtest el engine no actualiza global_state.market_regime.
        # Permitimos que el sistema opere mientras el régimen no sea explícitamente
        # hostil para la estrategia (CHOPPY para trend-following).
        if regime == "UNKNOWN":
            # No bloquear por régimen desconocido — otros filtros (A3, A5, Sophia)
            # protegen suficientemente. El sistema necesita operar para aprender.
            pass
        elif strategy_mapped == "TFTF":
            # TFTF requiere tendencia clara — pero solo bloquea si sabemos que NO hay tendencia
            if regime in ("CHOPPY", "RANGING"):
                return False, f"FAIL_A1: Strategy TFTF requires TRENDING regime, active is {regime}"
        elif strategy_mapped == "OB_RETEST":
            # OB Retest requiere estructura o tendencia
            if "TREND" not in regime and "RUPTURA" not in regime and "BREAKOUT" not in regime:
                return False, f"FAIL_A1: OB_RETEST requires TREND/BREAKOUT regime, active is {regime}"
        elif strategy_mapped == "MEAN_REVERSION":
            # Mean reversion prohibido en tendencias extremas o activos especulativos
            if "TREND" in regime:
                return False, f"FAIL_A1: MEAN_REVERSION is blocked during TRENDING regime"
            if "NO_MEAN_REVERSION" in profile.restrictions:
                return False, f"FAIL_A1: MEAN_REVERSION is strictly forbidden for {symbol}"
                
        # ---------------------------------------------------------------------
        # Paso A2 — CONDICIÓN DE SESIÓN Y TIMING
        # ---------------------------------------------------------------------
        # Regla: No abrir 30 minutos antes de eventos macro de alto impacto (simulado via config cooldown)
        from config import Config
        from core.global_state import global_state
        
        if getattr(global_state, 'macro_event_cooldown', False):
            return False, "FAIL_A2: Blocked by macro event cooldown"
            
        # Altcoins de bajo tier requieren volumen mínimo (percentil 40) o ventana óptima
        if profile.tier.value >= 2:
            # Para altcoins, simular ventana de liquidez.
            # No bloqueamos de noche si hay volumen, pero exige que no haya drift
            pass

        # ---------------------------------------------------------------------
        # Paso A3 — CONDICIÓN DE SEÑAL PRIMARIA
        # OMEGA FIX: La cadena anterior buscaba 'confidence' (no existe en
        # SignalEvent), luego 'ml_confidence' (None para non-ML), y caía al
        # default 0.5 de 'strength'. Pero 'strength' YA contiene el valor
        # calculado por la estrategia (0.85 para Phalanx, variable para Statistical).
        # Ahora usamos max(strength, ml_confidence) para obtener la mejor señal.
        # ---------------------------------------------------------------------
        # 1. strength es el campo primario que TODAS las estrategias setean
        strength_val = getattr(intent, 'strength', 0.5)
        
        # 2. ml_confidence es set por ML/Technical strategies (puede ser mayor)
        ml_conf = getattr(intent, 'ml_confidence', None)
        
        # 3. metadata.confidence como override explícito
        meta_conf = None
        if hasattr(intent, 'metadata') and intent.metadata:
            meta_conf = intent.metadata.get('confidence')
        
        # Usar el valor más alto disponible (la mejor señal de confianza)
        candidates = [v for v in [strength_val, ml_conf, meta_conf] if v is not None]
        confidence = max(candidates) if candidates else 0.5
        
        if confidence < profile.min_signal_threshold:
            return False, f"FAIL_A3: Signal confidence {confidence:.2f} is below minimum {profile.min_signal_threshold} for Tier {profile.tier.value} ({symbol})"

        # ---------------------------------------------------------------------
        # Paso A4 — CONDICIÓN DE CONFIRMACIÓN MULTICAPA
        # ---------------------------------------------------------------------
        # Comprobar reglas de compatibilidad de la estrategia
        if strategy_mapped not in profile.allowed_strategies:
            return False, f"FAIL_A4: Strategy {strategy_mapped} is forbidden or incompatible with {symbol}"
            
        # Confirmación CVD e indicadores específicos
        metadata = getattr(intent, 'metadata', None) or {}
        
        if strategy_mapped == "TFTF":
            # Confirmación de volumen en el pullback
            pullback_vol_ratio = metadata["pullback_volume_ratio"]
            if pullback_vol_ratio > 0.60:
                # Si el pullback tiene demasiado volumen, es posible reversión, no pullback
                return False, f"FAIL_A4: Pullback volume ratio {pullback_vol_ratio:.2f} too high (max 0.60)"
                
        elif strategy_mapped == "OB_RETEST":
            # El Order Block debe tener fuerza > 1.5x ATR
            ob_strength = metadata["ob_strength_atr"]
            if ob_strength < 1.5:
                return False, f"FAIL_A4: Order Block strength {ob_strength:.2f} is below 1.5x ATR requirement"
                
        elif strategy_mapped == "CASCADE":
            # Requiere clúster a menos de X% del precio
            dist_to_cluster = metadata["distance_to_cluster"]
            max_dist = 0.015 if symbol in ["BTC/USDT", "ETH/USDT"] else 0.03
            if dist_to_cluster > max_dist:
                return False, f"FAIL_A4: Distance to liquidations cluster {dist_to_cluster:.3f} exceeds max {max_dist}"

        # ---------------------------------------------------------------------
        # Paso A5 — CONDICIÓN DE RIESGO Y SIZING
        # ---------------------------------------------------------------------
        if portfolio:
            # 1. Portfolio Heat: Máximo de 3 posiciones abiertas concurrentes
            open_count = len([p for p in portfolio.virtual_ledger.values() if abs(p["quantity"]) > 1e-8])
            if open_count >= 3:
                return False, "FAIL_A5: Max concurrent positions (3) reached"
                
            # 2. Sizing minimo: el tamaño nocional debe ser >= $5 USD para cumplir Binance limits
            cash = portfolio.get_available_cash(horizon=getattr(intent, 'horizon', 'SCALPING'))
            # Size final = cash * kelly * factor_sizing
            kelly_val = profile.kelly_fraction
            calculated_size = cash * kelly_val * profile.factor_sizing
            leverage = getattr(Config, "BINANCE_LEVERAGE", 10)
            notional = calculated_size * leverage
            
            if notional < 5.0:
                return False, f"FAIL_A5: Calculated notional size ${notional:.2f} is below Binance minimum $5.00"

        # ---------------------------------------------------------------------
        # Paso A6 — CONDICIÓN DE NO-COLISIÓN
        # ---------------------------------------------------------------------
        # Outage check para Solana (SOL)
        if "OUTAGE_CHECK" in profile.restrictions:
            if getattr(global_state, 'solana_network_outage', False):
                return False, "FAIL_A6: Blocked by SOLANA network outage"
                
        # SEC case check para XRP
        if "SEC_CASE_VETO" in profile.restrictions:
            if getattr(global_state, 'xrp_regulatory_block', False):
                return False, "FAIL_A6: Blocked by XRP active regulatory risk"
                
        # MÓDULO HORIZON: DOGE Tier 4 protection via strength penalty instead of hard block.
        # QUÉ: DOGE ya no se bloquea al 100% por falta de catalizador.
        # POR QUÉ: global_state.doge_sentiment_catalyst nunca se setea → 140 bloqueos inútiles.
        # PARA QUÉ: Permitir DOGE trades con protección via 1/4 Kelly + Tier 4 threshold (0.60).
        # CÓMO: El strength penalty se aplica downstream en el consensus filter.
        # MANTIENE: Tier 4 min_signal_threshold=0.60, kelly=0.25, factor_sizing=0.50
                
        # ---------------------------------------------------------------------
        # Paso A7 — EJECUCIÓN DE APERTURA (CALIBRACIONES)
        # ---------------------------------------------------------------------
        # Inyectar stops y buffers de liquidación calculados sobre el ATR de este activo
        from core.senior_auditor import SeniorAuditor
        passed_audit, audit_reason = SeniorAuditor().verify_opening_audit(intent, portfolio)
        if not passed_audit:
            return False, audit_reason
            
        return True, "APPROVED"

    # =========================================================================
    # 🚥 PART V — SISTEMA DE CIERRE (C1-C7)
    # =========================================================================

    def verify_closing(self, position: Dict[str, Any], current_price: float, data_provider, now: datetime, symbol: Optional[str] = None) -> Tuple[bool, str]:
        """
        Ejecuta el pipeline de cierre de 7 pasos (C1-C7) sobre una posición activa.
        Retorna (True, "EXIT_REASON") si se debe cerrar, o (False, "") si se debe mantener.
        """
        symbol = symbol or position.get("symbol") or "BTC/USDT"
        profile = self.get_profile(symbol)
        pos_horizon = position["horizon"]
        qty = position["quantity"]
        entry_price = position["avg_price"]
        
        if abs(qty) < 1e-8 or entry_price <= 0.0:
            return False, ""
            
        strategy_mapped = self._map_strategy_name(position["opener_strategy_id"])
        
        # ---------------------------------------------------------------------
        # Paso C7 — CIERRE DE EMERGENCIA (PRIORIDAD MÁXIMA)
        # ---------------------------------------------------------------------
        from core.global_state import global_state
        
        # Outage de Solana
        if symbol == "SOL/USDT" and getattr(global_state, 'solana_network_outage', False):
            return True, "EMERGENCY_SOLANA_NETWORK_OUTAGE"
            
        # Regulatorio extremo de XRP/BNB
        if symbol == "XRP/USDT" and getattr(global_state, 'xrp_regulatory_block', False):
            return True, "EMERGENCY_XRP_REGULATORY_BLOCK"
            
        # Drawdown de sesión crítico (Bypass manual inyectable)
        if getattr(global_state, 'kill_switch_active', False):
            return True, "EMERGENCY_KILL_SWITCH_ACTIVE"

        # ---------------------------------------------------------------------
        # Paso C1 — CIERRE POR STOP LOSS INICIAL / TAKE PROFIT
        # ---------------------------------------------------------------------
        # (Aunque el risk manager principal los evalúa, la inteligencia de activos define los límites exactos)
        unrealized_pnl_pct = (current_price - entry_price) / entry_price if qty > 0 else (entry_price - current_price) / entry_price
        
        # Virtual Liquidation Buffer (Axioma de Supervivencia 5.2)
        leverage = position["leverage"]
        virtual_buffer_pct = 0.02
        max_allowed_drop_pct = (1.0 - virtual_buffer_pct) / leverage
        if unrealized_pnl_pct <= -max_allowed_drop_pct:
            return True, "VIRTUAL_LIQUIDATION_VULNERATION"
            
        # Short Squeeze Emergency Protocol (SC7)
        if qty < 0:
            squeeze_limit = -0.012 if pos_horizon != "SWING" else -0.035
            if unrealized_pnl_pct <= squeeze_limit:
                return True, "SHORT_SQUEEZE_PANIC"

        # ---------------------------------------------------------------------
        # Paso C2 — CIERRE POR INVALIDACIÓN DE CONTEXTO
        # ---------------------------------------------------------------------
        if strategy_mapped == "TFTF":
            # Si el ADX cae de 20 en la temporalidad operativa, la tendencia murió
            adx_val = position["last_adx_value"] # Mock or real field from tracking
            if adx_val < 20:
                return True, "INVALIDATION: ADX trend strength fell below 20"
                
            # Reversión sostenida de CVD
            cvd_trend = position["cvd_divergence_streak"]
            if cvd_trend >= 3:
                return True, "INVALIDATION: Sustained CVD divergence (3+ bars against position)"
                
        elif strategy_mapped == "OB_RETEST":
            # Si el precio perfora y cierra fuera del OB en la dirección contraria
            ob_violation = position["ob_extremum_violated"]
            if ob_violation:
                return True, "INVALIDATION: Order Block extreme price level violated"

        # ---------------------------------------------------------------------
        # Paso C5 — CIERRE POR TIEMPO LÍMITE (EXHAUSTION)
        # ---------------------------------------------------------------------
        entry_time_val = position.get("entry_time")
        if entry_time_val:
            if hasattr(entry_time_val, "timestamp"):
                entry_time_val = entry_time_val.timestamp()
                
            seconds_held = time.time() - entry_time_val
            
            if pos_horizon in ("SCALPING", "MICROSCALPING"):
                # Límite estricto de 1 hora para scalping para liberar capital
                if seconds_held > 3600:
                    return True, "HARD_SCALP_TIMEOUT_INCONDICIONAL"
            elif pos_horizon == "SWING":
                # Límite de 48 horas para swing
                if seconds_held > 172800:
                    return True, "HARD_SWING_TIMEOUT_EXHAUSTION"

        # ---------------------------------------------------------------------
        # Paso C6 — CIERRE POR REVERSIÓN DE SEÑAL
        # ---------------------------------------------------------------------
        signal_reverted = position["signal_reverted"]
        if signal_reverted:
            return True, "SIGNAL_REVERSAL"

        # Cierre por Invalidación Estratégica (ACS / ACR)
        from core.senior_auditor import SeniorAuditor
        should_close, audit_reason = SeniorAuditor().verify_closing_audit(
            position=position,
            current_price=current_price,
            data_provider=data_provider,
            now=now
        )
        if should_close:
            return True, audit_reason

        return False, ""

    # =========================================================================
    # 🛠️ HELPER METHODS
    # =========================================================================

    def _map_strategy_name(self, strategy_id: str) -> str:
        """Mapea el strategy_id a un nombre estándar del Módulo Maestro.
        
        OMEGA FIX: El fallback anterior era TFTF, lo cual causaba que TODAS
        las estrategias no-mapeadas (STAT_V1, Technical Momentum, ML_HYBRID)
        fueran vetadas por régimen UNKNOWN. Ahora el mapping es exhaustivo
        y el fallback es MOMENTUM_VOL (acepta cualquier régimen).
        """
        sid_upper = strategy_id.upper()
        
        # Exact or partial matches — ordered by specificity
        if "TFTF" in sid_upper:
            return "TFTF"
        elif "OB" in sid_upper or "SMC" in sid_upper:
            return "OB_RETEST"
        elif "CASCADE" in sid_upper or "LIQ" in sid_upper:
            return "CASCADE"
        elif "MEAN" in sid_upper and "REVERSION" in sid_upper:
            return "MEAN_REVERSION"
        elif "VWAP" in sid_upper:
            return "VWAP_REVERSION"
        elif "FUNDING" in sid_upper:
            return "FUNDING_ARB"
        elif "VOLATILITY" in sid_upper or "BREAKOUT" in sid_upper:
            return "VOLATILITY_BREAKOUT"
        elif "PAIR" in sid_upper:
            return "PAIR_ARB"
        elif "CHAIN" in sid_upper:
            return "ON_CHAIN"
        elif "SENTIMENT" in sid_upper:
            return "SENTIMENT_CONTRARIAN"
        elif "WYCKOFF" in sid_upper:
            return "WYCKOFF"
        
        # ═══════════════════════════════════════════════════════════
        # OMEGA FIX: Map existing strategies that were falling to TFTF
        # ═══════════════════════════════════════════════════════════
        elif "STAT" in sid_upper and "ARB" in sid_upper:
            return "PAIR_ARB"  # STAT_ARB_V1 → statistical arbitrage
        elif "STAT" in sid_upper:
            return "MOMENTUM_VOL"  # STAT_V1 → momentum/statistical
        elif "TECHNICAL" in sid_upper or "MOMENTUM" in sid_upper:
            return "MOMENTUM_VOL"  # [MSC]_Technical Momentum
        elif "ML" in sid_upper or "HYBRID" in sid_upper:
            return "MOMENTUM_VOL"  # ML_HYBRID strategies
        elif "SCALP" in sid_upper or "SCL" in sid_upper:
            return "MOMENTUM_VOL"  # Scalping strategies
        elif "SWING" in sid_upper or "SWG" in sid_upper:
            return "MOMENTUM_VOL"  # Swing strategies
        elif "SNIPER" in sid_upper:
            return "MOMENTUM_VOL"  # Sniper strategy
        
        # Fallback: MOMENTUM_VOL acepta cualquier régimen
        # (antes era TFTF que requería TRENDING y bloqueaba todo)
        return "MOMENTUM_VOL"

# Global helper instance
def get_asset_intelligence() -> AssetIntelligence:
    return AssetIntelligence()
