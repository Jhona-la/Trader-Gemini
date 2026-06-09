import os
import sys
import logging
from dotenv import load_dotenv
from core.enums import TimeFrame
from core.secure_store import SecureString

# Load environment variables from .env file (Phase 6 Absolute Path Fix)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# EXCLUSION LIST (Pairs with known data issues) - Defined globally for accessibility
# Reference: Master Bible v2.0.1 Phase 2.6
EXCLUDED_SYMBOLS_GLOBAL = ['SHIB/USDT', 'PEPE/USDT', 'BONK/USDT']

class EncryptedConfigMeta(type):
    """Metaclass to handle encrypted properties transparently"""
    # LEAN_MODE removed — single source of truth is Config.LEAN_MODE (L75)
    _secure_store = {}

    def _get_secure(cls, key, env_var):
        if key not in cls._secure_store:
            val = os.getenv(env_var, '')
            cls._secure_store[key] = SecureString(val)
        return cls._secure_store[key].get_unmasked()

    @property
    def BINANCE_API_KEY(cls):
        return cls._get_secure('BINANCE_API_KEY', 'BINANCE_API_KEY')
    
    @property
    def BINANCE_SECRET_KEY(cls):
        return cls._get_secure('BINANCE_SECRET_KEY', 'BINANCE_SECRET_KEY')
        
    @property
    def BINANCE_TESTNET_API_KEY(cls):
        return cls._get_secure('BINANCE_TESTNET_API_KEY', 'BINANCE_TESTNET_API_KEY')
        
    @property
    def BINANCE_TESTNET_SECRET_KEY(cls):
        return cls._get_secure('BINANCE_TESTNET_SECRET_KEY', 'BINANCE_TESTNET_SECRET_KEY')
        
    @property
    def BINANCE_DEMO_API_KEY(cls):
        return cls._get_secure('BINANCE_DEMO_API_KEY', 'BINANCE_DEMO_API_KEY')
        
    @property
    def BINANCE_DEMO_SECRET_KEY(cls):
        return cls._get_secure('BINANCE_DEMO_SECRET_KEY', 'BINANCE_DEMO_SECRET_KEY')
        
    @property
    def WANDB_API_KEY(cls):
        # Support both standard name and user's alias in .env
        return cls._get_secure('WANDB_API_KEY', 'WANDB_API_KEY') or os.getenv('WandB_Key', '')

class OmniscientRegistry:
    """
    Capa 2: Centralized Omniscient Registry.
    Tracks FIXED (immutable bounds) and ADAPTIVE values.
    Enforces priority of FIXED values over ADAPTIVE adjustments.
    """
    FIXED_VALUES = {
        'MAX_DRAWDOWN': 35.0,               # Max Drawdown before emergency stop (increased for micro accounts)
        'MAX_RISK_PER_TRADE': 0.05,         # 5% max risk per trade
        'MIN_NOTIONAL_USD': 5.05,           # Binance minimum order notional
        'MAX_LEVERAGE': 20,                 # Maximum allowed leverage
        'MAX_CONCURRENT_POSITIONS': 5,      # Concentrated capital limit
        'MAX_SL_PCT_LIMIT': 0.05,           # Hard limit for Stop Loss (5%)
        'MIN_PROFIT_AFTER_FEES': 0.0015,    # Minimum viability net threshold ( lowered to 0.15% for Scalping )
    }

    def __init__(self, config_ref):
        self._config = config_ref
        self._adaptive_cache = {}
        self.logger = logging.getLogger("OmniscientRegistry")

    def get_value(self, name: str, horizon: str = None) -> Any:
        # 1. First Priority: Check FIXED values
        if name in self.FIXED_VALUES:
            return self.FIXED_VALUES[name]
        
        # 2. Horizon-aware adaptive values
        if horizon:
            h_upper = horizon.upper()
            h_obj = getattr(self._config, "Horizons", None)
            if h_obj:
                # Find matching attribute in class Horizons
                h_name = "Microscalping" if h_upper == "MICROSCALPING" else ("Scalping" if h_upper == "SCALPING" else "Swing")
                h_dict = getattr(h_obj, h_name, None)
                if h_dict and name in h_dict:
                    return h_dict[name]
        
        # 3. Standard global configuration attributes
        if hasattr(self._config, name):
            return getattr(self._config, name)
            
        # Fallback to cache
        return self._adaptive_cache.get(name)

    def update_adaptive_value(self, name: str, new_val: Any, horizon: str = None) -> bool:
        # Check against FIXED boundaries
        if name == 'leverage' or name == 'BINANCE_LEVERAGE':
            max_lev = self.FIXED_VALUES['MAX_LEVERAGE']
            if new_val > max_lev:
                self.logger.warning(f"🛡️ [Capa 2 Registry] Blocked update for {name}: {new_val}x exceeds FIXED MAX_LEVERAGE ({max_lev}x)")
                return False
        elif name == 'sl_pct' or name == 'SL_PCT':
            max_sl = self.FIXED_VALUES['MAX_SL_PCT_LIMIT']
            if new_val > max_sl:
                self.logger.warning(f"🛡️ [Capa 2 Registry] Clipped {name}: {new_val*100:.2f}% exceeds FIXED MAX_SL_PCT_LIMIT ({max_sl*100:.2f}%)")
                new_val = max_sl
        
        # Write to config if attribute exists
        if horizon:
            h_obj = getattr(self._config, "Horizons", None)
            if h_obj:
                h_name = "Microscalping" if horizon.upper() == "MICROSCALPING" else ("Scalping" if horizon.upper() == "SCALPING" else "Swing")
                h_dict = getattr(h_obj, h_name, None)
                if h_dict and name in h_dict:
                    h_dict[name] = new_val
                    return True
        
        if hasattr(self._config, name):
            setattr(self._config, name, new_val)
            return True
            
        self._adaptive_cache[name] = new_val
        return True

from typing import Any

class Config(metaclass=EncryptedConfigMeta):
    # ========================================================================
    # GLOBAL SETTINGS
    # ========================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEBUG_TRACE_ENABLED = False
    # FIX-FORENSIC-V82: Removed duplicate LEAN_MODE = True (was shadowed by L75 anyway)

    # ════════════════════════════════════════════════════════════════
    # 🎯 INTEGRAL MODE — CONSENSUS-BASED OPERATION
    # QUÉ: Flag que mantiene protecciones de micro-cuenta ($13) activas
    #   mientras permite que TODAS las estrategias operen en consenso.
    # POR QUÉ: LEAN_MODE v1 amputó estrategias (0 ML). Ahora usamos
    #   Consenso Ponderado v1.0: ninguna estrategia tiene veto absoluto.
    #   Sophia/Oracle PENALIZAN strength pero no BLOQUEAN.
    # PARA QUÉ: Sistema integral — scalping + swing + ML + técnica
    #   trabajando juntas. Position Rotation libera margin automáticamente.
    # CÓMO: LEAN_MODE=True preserva: wider kill switch (35% DD),
    #   margin cap 95%, wider headroom floors. ML re-habilitado.
    # ════════════════════════════════════════════════════════════════
    LEAN_MODE = False  # Set to False to run ALL strategies (Full Mode)
    ACTIVE_TRADING_LIMIT = 10  # Solo opera los Top 10, mide los siguientes 16
    LEAN_ML_ENABLED = False  # ML Strategy: TEMPORARILY DISABLED (Fix #1 to avoid short bias loss)
    LEAN_TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Focus on high liquidity


    # ========================================================================
    # BINANCE API CREDENTIALS (Loaded from .env file)
    # ========================================================================
    
    # 🔐 PHASE 17: ENCRYPTED KEYS
    # Keys are now managed dynamically by EncryptedConfigMeta.
    # We do not define them here as static attributes to keep RAM clean.
    
    # 🔐 PHASE 17 (SOVEREIGN-DEPLOY): HARDENED PRODUCTION KEYS
    # OVERRIDEN BY BLOCK G PROTOCOL: PAPER TRADING ENABLED
    BINANCE_USE_TESTNET = os.getenv('BINANCE_USE_TESTNET', 'True').lower() == 'true'
    BINANCE_USE_DEMO = os.getenv('BINANCE_USE_DEMO', 'True').lower() == 'true'
    
    # === BINANCE FUTURES SETTINGS ===
    # Default: USDT-Margined Futures (standard). 
    # For COIN-Margined, code modifications in binance_executor would be needed (defaultType='delivery').
    # BUG #33 FIX: Changed default to False to allow Spot mode. CLI --mode argument will override this.
    BINANCE_USE_FUTURES = True  # Set to True to trade on Binance Futures instead of Spot
    BINANCE_LEVERAGE = 20  # FORENSIC FIX: 20x leverage for $13 account to bypass BTC 0.001 ($70) notional limits
    LEVERAGE_SWING = 20  # FORENSIC FIX: Swing needs 20x to reach $70 minimum notional with $3.90 margin
    BINANCE_MARGIN_TYPE = "ISOLATED"  # Options: "ISOLATED" or "CROSS"
    BINANCE_TAKER_FEE_BNB = 0.000375 # 0.0375% (with BNB discount)
    BINANCE_MAKER_FEE_BNB = 0.0002   # 0.02% (LIMIT orders = Maker fee, FORENSIC FIX #4)
    
    # Symbols format standardization (Phase 6 Fix)
    # The API expects SYMBOLUSDT, but we prefer SYMBOL/USDT for UI.
    # We will enforce '/' in Config and remove it in API calls.
    @staticmethod
    def get_clean_pairs(pairs_list):
        return [p.replace('/', '') for p in pairs_list]
    
    # Dynamic Data Directory (Separate Spot & Futures)
    # Default base. main.py will override this to 'dashboard/data/futures' or 'spot'
    DATA_DIR = "dashboard/data/futures" 
    
    # Ensure directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Interactive Brokers Settings (Disabled - focusing on crypto only)
    IBKR_HOST = "127.0.0.1"
    IBKR_PORT = 7497
    IBKR_CLIENT_ID = 1

    # === TRADING PAIRS CONFIGURATION ===
    
    # Expose global exclusion list as class attribute
    EXCLUDED_SYMBOLS = EXCLUDED_SYMBOLS_GLOBAL

    # SPOT Trading Pairs (All available in Binance Spot)
    # NOTE: EXCLUDED pairs are filtered out by DataProvider
    _RAW_SPOT_PAIRS = [
        # Top 10 Major Coins
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "DOT/USDT",
        
        # High Volume Altcoins
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
        "LTC/USDT", "ETC/USDT", "FLOKI/USDT", "WIF/USDT",
        "AAVE/USDT", "COMP/USDT", "SAND/USDT", "MANA/USDT"
    ]
    # Filter Exclusion List
    CRYPTO_SPOT_PAIRS = [s for s in _RAW_SPOT_PAIRS if s not in EXCLUDED_SYMBOLS_GLOBAL]
    
    # BUG FIX #13: Binance Testnet SPOT has LIMITED pairs available
    # Only basic major coins work in Testnet SPOT
    BINANCE_TESTNET_SPOT_PAIRS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", 
        "XRP/USDT", "DOGE/USDT", "ADA/USDT",
        "DOT/USDT", "SOL/USDT", "LTC/USDT"
    ]
    # Total: 9 pairs (verified in Binance Testnet SPOT)
    
    # FUTURES Trading Pairs (Verified available in Demo Trading)
    # Note: SHIB, PEPE, FLOKI, BONK not available in Futures Demo
    # CIRUGÍA-V131: Concentrate $13 capital in Top 5 Elite pairs to prevent margin fragmentation
    CORE_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT"
    ]
    
    PROSPECT_SYMBOLS = [
        "DOT/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "NEAR/USDT",
        "FTM/USDT", "FIL/USDT", "LDO/USDT", "OP/USDT", "ARB/USDT",
        "APT/USDT", "SUI/USDT", "PEPE/USDT", "AAVE/USDT", "COMP/USDT",
        "WIF/USDT"
    ]
    
    CRYPTO_FUTURES_PAIRS = CORE_SYMBOLS + PROSPECT_SYMBOLS
    
    # Auto-select correct pairs based on mode
    # BUG #14 FIX: Binance Testnet SPOT is UNRELIABLE (most pairs don't exist)
    # Solution: SPOT only works in PRODUCTION, Testnet/Demo users should use FUTURES
    # SINGLE SOURCE OF TRUTH: Initial capital for $13 micro-scalping strategy (SOVEREIGN-DEPLOY)
    INITIAL_CAPITAL = 13.0  # Base capital for sizing and HWM

    # ================================================================
    # FORENSIC-V156: PER-SYMBOL TRADING PROFILES
    # QUÉ: Cada moneda tiene características únicas de volatilidad,
    #   spread, y bias direccional que requieren parámetros diferenciados.
    # POR QUÉ: BTC LONG = 0% WR (-$0.91), BTC SHORT = 100% WR (+$0.26).
    #   El sistema trataba todas las monedas igual → pérdidas evitables.
    # PARA QUÉ: Adaptar SL/TP/confianza por moneda para maximizar
    #   rentabilidad y minimizar trades tóxicos.
    # CÓMO: Multipliers sobre SL/TP base + bias direccional sobre confianza.
    # CUÁNDO: Evaluado en consensus_filter.py y risk_manager.py.
    # DÓNDE: Config.SymbolProfiles
    # QUIÉN: Arquitecto Senior + Quant Developer
    # ================================================================
    class SymbolProfiles:
        """Per-symbol trading characteristics based on forensic backtest data."""
        PROFILES = {
            "BTC/USDT": {
                "sl_mult": 1.4,
                "tp_mult": 1.0,
                "long_bias": -0.03,
                "short_bias": +0.05,
                "min_confidence": 0.45,
                "max_concurrent": 1,
                "category": "MAJOR",
                "base_leverage": 20,         # SOVEREIGN: Max leverage due to low vol
                "max_risk_pct": 0.05,        # SOVEREIGN: 5% risk cap
                "default_atr_pct": 0.0025,
                "trail_protect_atr": 1.0,
                "trail_pursue_atr": 1.8,
                "trail_capture_atr": 0.6,
                "momentum_sensitivity": 0.8,
                "sl_min": 72, "sc_min": 78, "isn_threshold": 10,
                "kelly_factor_long": 1.00, "kelly_factor_short": 0.90,
                "atr_stop_long": 1.5, "atr_stop_short": 1.8,
            },
            "ETH/USDT": {
                "sl_mult": 1.1,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": +0.05,
                "min_confidence": 0.42,
                "max_concurrent": 2,
                "category": "MAJOR",
                "base_leverage": 15,         # SOVEREIGN: Medium-high leverage
                "max_risk_pct": 0.04,        # SOVEREIGN: 4% risk cap
                "default_atr_pct": 0.0045,
                "trail_protect_atr": 0.9,
                "trail_pursue_atr": 1.5,
                "trail_capture_atr": 0.5,
                "momentum_sensitivity": 1.0,
                "sl_min": 68, "sc_min": 74, "isn_threshold": 10,
                "kelly_factor_long": 1.00, "kelly_factor_short": 0.85,
                "atr_stop_long": 1.7, "atr_stop_short": 2.0,
            },
            "SOL/USDT": {
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.45,
                "max_concurrent": 2,
                "category": "ALT",
                "base_leverage": 10,         # SOVEREIGN: Volatile, lower leverage
                "max_risk_pct": 0.03,        # SOVEREIGN: 3% risk cap
                "default_atr_pct": 0.0065,
                "trail_protect_atr": 0.8,
                "trail_pursue_atr": 1.2,
                "trail_capture_atr": 0.4,
                "momentum_sensitivity": 1.3,
                "sl_min": 65, "sc_min": 70, "isn_threshold": 10,
                "kelly_factor_long": 0.80, "kelly_factor_short": 0.70,
                "atr_stop_long": 2.0, "atr_stop_short": 2.5,
            },
            "BNB/USDT": {
                "sl_mult": 1.1,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.50,
                "max_concurrent": 1,
                "category": "ALT",
                "base_leverage": 15,
                "max_risk_pct": 0.04,
                "default_atr_pct": 0.0040,
                "trail_protect_atr": 0.9,
                "trail_pursue_atr": 1.4,
                "trail_capture_atr": 0.5,
                "momentum_sensitivity": 1.0,
                "sl_min": 65, "sc_min": 70, "isn_threshold": 10,
                "kelly_factor_long": 0.90, "kelly_factor_short": 0.75,
                "atr_stop_long": 2.0, "atr_stop_short": 2.3,
            },
            "XRP/USDT": {
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.50,
                "max_concurrent": 1,
                "category": "ALT",
                "base_leverage": 10,
                "max_risk_pct": 0.03,
                "default_atr_pct": 0.0055,
                "trail_protect_atr": 0.8,
                "trail_pursue_atr": 1.3,
                "trail_capture_atr": 0.4,
                "momentum_sensitivity": 1.2,
                "sl_min": 68, "sc_min": 75, "isn_threshold": 10,
                "kelly_factor_long": 0.75, "kelly_factor_short": 0.65,
                "atr_stop_long": 2.0, "atr_stop_short": 2.3,
            },
            "DOGE/USDT": {
                "sl_mult": 1.5,              # Wide SL for extreme noise
                "tp_mult": 1.2,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.55,
                "max_concurrent": 1,
                "category": "MEME",
                "base_leverage": 5,          # SOVEREIGN: Strict leverage on memecoins
                "max_risk_pct": 0.02,        # SOVEREIGN: Lowest risk cap
                "default_atr_pct": 0.0080,
                "trail_protect_atr": 0.7,
                "trail_pursue_atr": 1.1,
                "trail_capture_atr": 0.3,
                "momentum_sensitivity": 1.5,
                "sl_min": 60, "sc_min": 65, "isn_threshold": 10,
                "kelly_factor_long": 0.60, "kelly_factor_short": 0.50,
                "atr_stop_long": 2.5, "atr_stop_short": 3.0,
            },
            "ADA/USDT": {
                "sl_mult": 1.1,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.50,
                "max_concurrent": 1,
                "category": "ALT",
                "base_leverage": 10,
                "max_risk_pct": 0.03,
                "default_atr_pct": 0.0050,
                "trail_protect_atr": 0.9,
                "trail_pursue_atr": 1.4,
                "trail_capture_atr": 0.5,
                "momentum_sensitivity": 1.0,
                "sl_min": 65, "sc_min": 72, "isn_threshold": 10,
                "kelly_factor_long": 0.80, "kelly_factor_short": 0.75,
                "atr_stop_long": 2.0, "atr_stop_short": 2.5,
            },
            "AVAX/USDT": {
                "sl_mult": 1.2,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.50,
                "max_concurrent": 1,
                "category": "ALT",
                "base_leverage": 8,
                "max_risk_pct": 0.03,
                "default_atr_pct": 0.0060,
                "trail_protect_atr": 0.8,
                "trail_pursue_atr": 1.3,
                "trail_capture_atr": 0.4,
                "momentum_sensitivity": 1.2,
                "sl_min": 65, "sc_min": 70, "isn_threshold": 10,
                "kelly_factor_long": 0.75, "kelly_factor_short": 0.70,
                "atr_stop_long": 2.2, "atr_stop_short": 2.6,
            },
            "LINK/USDT": {
                "sl_mult": 1.1,
                "tp_mult": 1.0,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "min_confidence": 0.50,
                "max_concurrent": 1,
                "category": "ALT",
                "base_leverage": 10,
                "max_risk_pct": 0.03,
                "default_atr_pct": 0.0050,
                "trail_protect_atr": 0.9,
                "trail_pursue_atr": 1.4,
                "trail_capture_atr": 0.5,
                "momentum_sensitivity": 1.1,
                "sl_min": 65, "sc_min": 70, "isn_threshold": 10,
                "kelly_factor_long": 0.80, "kelly_factor_short": 0.75,
                "atr_stop_long": 2.0, "atr_stop_short": 2.4,
            },
        }
        DEFAULT = {
            "sl_mult": 1.0, "tp_mult": 1.0,
            "long_bias": 0.0, "short_bias": 0.0,
            "min_confidence": 0.50, "max_concurrent": 1,
            "category": "OTHER",
            "base_leverage": 5,          # SOVEREIGN: Safe default
            "max_risk_pct": 0.02,        # SOVEREIGN: Safe default risk
            "default_atr_pct": 0.0050,
            "trail_protect_atr": 0.9,
            "trail_pursue_atr": 1.4,
            "trail_capture_atr": 0.5,
            "momentum_sensitivity": 1.0,
            "sl_min": 65, "sc_min": 72, "isn_threshold": 15,
            "kelly_factor_long": 0.70, "kelly_factor_short": 0.60,
            "atr_stop_long": 2.0, "atr_stop_short": 2.5,
        }

        @classmethod
        def get(cls, symbol: str) -> dict:
            """Get profile for a symbol, normalizing BTC/USDT and BTCUSDT formats."""
            norm = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            return cls.PROFILES.get(norm, cls.DEFAULT)

    # ════════════════════════════════════════════════════════════════
    # 🏃 MOTOR DE PERSECUCIÓN DINÁMICA DE GANANCIAS (SISTEMA V7)
    # ════════════════════════════════════════════════════════════════
    class Trailing:
        """Configuración maestra para el Motor Dinámico de Persecución de Ganancias (v7)"""
        TRAILING_ASSET_PROFILES = {
            "BTC/USDT": {"pullback_tol": 0.40, "trail_f1": 1.5, "trail_f2": 1.2, "trail_f3": 1.0, "trail_runner": 0.8},
            "ETH/USDT": {"pullback_tol": 0.45, "trail_f1": 1.7, "trail_f2": 1.4, "trail_f3": 1.2, "trail_runner": 1.0},
            "BNB/USDT": {"pullback_tol": 0.35, "trail_f1": 2.0, "trail_f2": 1.6, "trail_f3": 1.3, "trail_runner": 1.1},
            "SOL/USDT": {"pullback_tol": 0.60, "trail_f1": 2.0, "trail_f2": 1.8, "trail_f3": 1.5, "trail_runner": 1.3},
            "XRP/USDT": {"pullback_tol": 0.45, "trail_f1": 1.8, "trail_f2": 1.5, "trail_f3": 1.3, "trail_runner": 1.0},
            "DOGE/USDT": {"pullback_tol": 0.30, "trail_f1": 2.5, "trail_f2": 2.0, "trail_f3": 1.5, "trail_runner": 1.0},
            "DEFAULT": {"pullback_tol": 0.40, "trail_f1": 2.0, "trail_f2": 1.5, "trail_f3": 1.2, "trail_runner": 1.0}
        }
        
        STRATEGY_FAMILY_PROFILES = {
            "MOMENTUM": {"r1_pct": 0.25, "r2_pct": 0.25, "runner_pct": 0.50},
            "MEAN_REVERSION": {"r1_pct": 0.40, "r2_pct": 0.50, "runner_pct": 0.10},
            "STRUCTURE": {"r1_pct": 0.50, "r2_pct": 0.30, "runner_pct": 0.20},
            "ORDERFLOW": {"r1_pct": 0.60, "r2_pct": 0.30, "runner_pct": 0.10},
            "DEFAULT": {"r1_pct": 0.50, "r2_pct": 0.30, "runner_pct": 0.20}
        }

        @classmethod
        def get_asset_profile(cls, symbol: str) -> dict:
            norm = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            return cls.TRAILING_ASSET_PROFILES.get(norm, cls.TRAILING_ASSET_PROFILES["DEFAULT"])
            
        @classmethod
        def get_family_profile(cls, family: str) -> dict:
            return cls.STRATEGY_FAMILY_PROFILES.get(family.upper(), cls.STRATEGY_FAMILY_PROFILES["DEFAULT"])

    # MANDATORY TAGGING SYSTEM - SCALPING VS SWING
    STRATEGY_LABELS = {
        "technical": "Technical Momentum",
        "ml_strategy": "ML Trend Swing",
        "sniper_strategy": "Sniper Ultra",
        "statistical": "Statistical Mean Reversion",
        "arbitrage": "Arbitrage Flow",
        "phalanx": "Phalanx Multi-Signal"
    }

    # === TRADING SETTINGS ===
    TIMEFRAME = TimeFrame.M5  # M5 Timeframe (Scalping P1 Fee Drag fix)
    MAX_SIGNAL_AGE = 300      # 300s for M5 timeframe to allow for execution
    
    # MULTI-SYMBOL EXPANSION: Top Liquidity Assets for Scalping
    # Defaulting to all Futures Pairs (24 total)
    TRADING_PAIRS = CRYPTO_FUTURES_PAIRS

    
    # Risk settings for Multi-Symbol Coordination
    MAX_CONCURRENT_POSITIONS = 5   # CIRUGÍA-V131: 5 (one per Elite symbol) — concentrate $13 capital
    COOLDOWN_PERIOD_SECONDS = 300  # FORENSIC-V17: 300s — prevent immediate toxic re-entries
    MAX_POSITIONS_PER_SYMBOL = 1   # Still 1 per symbol to prevent double-spending on same asset
    
    # Position Sizing Configuration
    POSITION_SIZE_MICRO_ACCOUNT = 0.05   # [HOTFIX] Bajado de 30% a 5% por trade → $0.65 margen × 10x = $6.5 notional
    POSITION_SIZE_SMALL_ACCOUNT = 0.15   # Lowered from 20%
    
    # Trade Validation Thresholds
    MIN_PROFIT_AFTER_FEES = 0.0015 # 0.15% minimum net profit
    MIN_RR_RATIO = 1.5             # 1.5:1 R:R minimum
    
    # Risk Management
    # ════════════════════════════════════════════════════════════════
    # 🔒 RISK PARAMETERS — SINGLE SOURCE OF TRUTH
    # QUÉ: Delegados a Config.Risk. Estos proxies garantizan
    #   que Config.MAX_RISK_PER_TRADE == Config.Risk.MAX_RISK_PER_TRADE.
    # POR QUÉ: Antes había 0.10 aquí y 0.05 en Risk, causando que
    #   el RiskManager usara 10% pero la validación esperaba 5%.
    # PARA QUÉ: Un solo valor → el motor evolutivo muta correctamente.
    # ════════════════════════════════════════════════════════════════
    MAX_RISK_PER_TRADE = 0.05  # [UNIFIED] 5% — delegates to Config.Risk.MAX_RISK_PER_TRADE
    STOP_LOSS_PCT = 0.02       # 2% stop loss — SYNCED with Config.Risk.STOP_LOSS_PCT
    MAX_SLIPPAGE_PCT = 0.001   # 0.1% max slippage — SYNCED with Config.Risk.MAX_SLIPPAGE_PCT
    
    # === RISK CAPITOL HIERARCHY (USER RULE PRESERVATION) ===
    class Risk:
        MAX_DRAWDOWN = 35.0           # 35.0% max drawdown (SOVEREIGN LIMIT) - Increased from 15% for $13 micro-accounts
        DEFAULT_BOOTSTRAP_WR = 0.52 
        BOOTSTRAP_TRADES = 20
        MAX_RISK_PER_TRADE = 0.05  
        STOP_LOSS_PCT = 0.02       
        MAX_SLIPPAGE_PCT = 0.001
        USE_PREDICTIVE_TP = False     # CIRUGÍA-V100: DISABLED — net PnL -4.65 after fees (769 trades, 47% WR). Trailing stops are superior.
        TOXIC_ASSETS = ["DOT/USDT", "ATOM/USDT", "XRP/USDT"]  # XRP blacklisted due to -1.30 PnL drag.
        
        # ════════════════════════════════════════════════════════════════
        # 🛡️ RISK THRESHOLDS — ERADICATING MAGIC NUMBERS (PHASE 2)
        # QUÉ: Centralización de umbrales extraídos de risk_manager.py.
        # POR QUÉ: Permite que el sistema mute y modifique dinámicamente
        #   sus reglas de riesgo sin requerir cambios en el código duro.
        # PARA QUÉ: Integración completa con Hyper-Evolver.
        # ════════════════════════════════════════════════════════════════
        RISK_THRESHOLDS = {
            'petim_exhaustion_mult': 2.0,
            'petim_exhaustion_pnl': -0.02,
            'swing_min_equity_block': 50.0,
            'zombie_hours_held': 7.5,
            'zombie_pnl_max': 0.005,  # 0.5% (typo fix: was 0.5 = 50% profit)
            'merit_win_rate_min': 0.60,
            'merit_score_high': 1.2,
            'merit_factor_expansion': 1.5,
        }
    
    # === INTELLIGENT REVERSE (FLIPPING) PARAMETERS (Phase 5) ===
    # PROFESSOR METHOD:
    # QUÉ: Umbrales de viabilidad para el cambio de dirección táctico.
    # POR QUÉ: Evita "whipsaws" (ser sierra) en mercados picados/laterales.
    # PARA QUÉ: Garantizar que el Flip tenga un valor esperado positivo real.
    FLIP_MIN_ATR_PCT = 0.005      # 0.5% ATR mínimo para autorizar Flip
    FLIP_MIN_POTENTIAL_RR = 2.0    # R:R esperado para la nueva dirección
    FLIP_MAX_DAILY_COUNT = 99      # NO LIMIT per user request
    FLIP_COST_THRESHOLD = 0.002    # 0.2% max cost (Fees + Slippage)
    FLIP_COOLDOWN_SECONDS = 0      # NO COOLDOWN per user request
    
    # Position Sizing Thresholds
    POSITION_SIZE_SMALL_THRESHOLD = 1000   
    POSITION_SIZE_LARGE_THRESHOLD = 10000  
    POSITION_SIZE_MEDIUM_ACCOUNT = 0.15  
    POSITION_SIZE_LARGE_ACCOUNT = 0.10   
    
    # Strategy Filters (Restored)
    PATTERN_BULLISH_RSI_MAX = 60
    PATTERN_BEARISH_RSI_MIN = 40

    # === DATA CAPTURE AND MULTI-TIMEFRAME (PHASE 28) ===
    class Data:
        # Map backtest horizon (days) to optimal resolution
        # QUÉ: Mapeo de tiempo. POR QUÉ: Extraer ruido. PARA QUÉ: Rentabilidad Swing.
        HORIZON_RESOLUTION_MAP = {
            1: '1m',   # Scalping (HFT Microstructure)
            7: '15m',  # Intraday
            15: '1h',  # Swing
            30: '4h'   # Position/Mensual
        }
        
        @classmethod
        def get_resolution_for_horizon(cls, horizon_days):
            if horizon_days <= 1: return '1m'
            elif horizon_days <= 7: return '15m'
            elif horizon_days <= 15: return '1h'
            else: return '4h'

    # ═══════════════════════════════════════════════════════════════
    # MARGIN SILOS: Data-driven allocation from backtest 3051676c.
    # Scalping: 85% — Fast recycled margin (Módulo Omega)
    # Swing: 15% — Complementary slow trades (Módulo Omega)
    # ═══════════════════════════════════════════════════════════════
    MICROSCALPING_MARGIN_CAP = 0.00  # DISABLED (Merged into Scalping)
    SCALPING_MARGIN_CAP = 0.85       # PRIMARY: Módulo Omega Compounding Engine
    SWING_MARGIN_CAP = 0.15          # SECONDARY: Complementary

    # === HORIZON SETTINGS ===
    class Horizons:
        Microscalping = {
            # FORENSIC-V155: TP 0.60%→0.35%, SL stays 0.50%
            # DATA: 0/35 trades hit TP at 0.60%. Winners exit TIMEOUT at +0.10-0.73%.
            # 0.35% TP is achievable in 2-5 candles. SL 0.50% survives BTC noise.
            # WR improvement compensates for lower R:R (Kelly-optimal at WR>40%).
            'tp_pct': 0.0035,         # 0.35% TP (optimized for M1-M3 micro movements)
            'sl_pct': 0.0050,         # 0.50% SL (optimized to survive M1-M3 noise)
            'max_hold_time': 1800,    # 30 minutes max hold time
            'rsi_period': 5,
            'rsi_buy': 30,
            'rsi_sell': 70,
            'bb_period': 10,
            'bb_std': 1.5,
            'ema_fast': 8,
            'ema_slow': 21,
            'ema_trend': 50,
            'atr_period': 7,
            'adx_period': 7,
            'timeframes': ['1m', '3m', '5m'],
            'primary_tf': '1m', 
            'min_volume_ratio': 0.5,
            'cooldown_seconds': 5,
            'max_hold_bars': 30,
            'strength_threshold': 0.35, 
            'atr_sl_mult': 2.0,
            'atr_tp_mult': 1.5,       # FORENSIC-V155: 2.5→1.5 (tighter TP capture)
            'sophia_refit': 10,
        }

        Scalping = {
            # FORENSIC-V156: TP 0.35%, SL 0.35% (1:1 Base RR)
            # TP 0.35% is within reach. SL 0.35% survives normal noise but protects better.
            'tp_pct': 0.0035,         # 0.35% TP (optimized for M5 scalping)
            'sl_pct': 0.0035,         # 0.35% SL (balanced with TP for higher WR)
            'max_hold_time': 7200,    # [ROUND4-OPT] Límite estricto e incondicional de tiempo en segundos (2 horas)
            'rsi_period': 5,          # GOLDEN: RSI ultra-rápido
            'rsi_buy': 35,            # GOLDEN: Wider zone
            'rsi_sell': 65,           # GOLDEN: Wider zone
            'bb_period': 10,          # GOLDEN: Bollinger rápido
            'bb_std': 1.5,            # GOLDEN: Tighter bands
            'ema_fast': 8,            # GOLDEN: EMA rápida
            'ema_slow': 21,           # GOLDEN: EMA lenta
            'ema_trend': 50,          # GOLDEN: Trend filter
            'atr_period': 7,          # GOLDEN: ATR corto
            'adx_period': 7,          # GOLDEN: ADX rápido
            'timeframes': ['1m', '5m', '15m'],
            'primary_tf': '1m', 
            'min_volume_ratio': 0.4,
            'cooldown_seconds': 15,
            'max_hold_bars': 120,      # [ROUND3-OPT] Alineado con max_hold_time a 2 horas (120m) para dar espacio real
            'strength_threshold': 0.45, 
            'atr_sl_mult': 2.0,       # V156: Lowered from 3.0 to tighten stops and improve R:R
            'atr_tp_mult': 2.0,       # V156: Lowered from 3.5 to make TP more reachable
            'sophia_refit': 50,
        }
        
        Swing = {
            'tp_pct': 0.045,          # 4.5% TP
            'sl_pct': 0.025,          # 2.5% SL
            'rsi_period': 14,         # RSI estándar
            'rsi_buy': 35,            # Oversold conservador
            'rsi_sell': 65,           # Overbought conservador
            'bb_period': 20,          # Bollinger estándar
            'bb_std': 2.0,            # Bandas estándar
            'ema_fast': 20,           # EMA estándar
            'ema_slow': 50,           # EMA media
            'ema_trend': 200,         # Golden Cross filter
            'atr_period': 14,         # ATR estándar
            'adx_period': 14,         # ADX estándar
            'timeframes': ['1h', '4h', '1d'],  
            'primary_tf': '1h',       
            'min_volume_ratio': 1.0,  
            'cooldown_seconds': 3600, 
            'max_hold_bars': 96,      
            'strength_threshold': 0.45, # [ROUND3-OPT] Reducido de 0.55 a 0.45 para reactivar Swing y evitar sobre-filtrado
            'atr_sl_mult': 3.0,       
            'atr_tp_mult': 4.5,       
            'sophia_refit': 24,       
        }
        
        Mutations = {
            'min_atr_required': 0.0004,    
            'adx_threshold': 18,           
            'strength_threshold': 0.45,    
            'max_tp_cap': 0.0035,          # Tighter cap for 1m micro-scalping
            'max_sl_cap': 0.0025,          # Defensive SL cap
        }
        
        GlobalThresholds = {
            'rsi_pullback_uptrend': 40,
            'rsi_rally_downtrend': 60,
            'rsi_extreme_low': 30,
            'rsi_extreme_high': 70,
            'vol_ratio_btc': 1.2,
            'vol_ratio_alts': 1.1,
            'vol_ratio_expansion': 1.5,
            'vol_ratio_high': 1.2,
            'vol_ratio_low': 0.8,
            'volatility_gate_pct': 0.025,
            'bb_pos_lower_prox': 0.25,
            'bb_pos_upper_prox': 0.75,
            # ═══════════════════════════════════════════════════════════════
            # AUDIT FIX: Sophia thresholds data-driven from backtest:
            #   conf < 0.60: 22% WR (worse than coin-flip)
            #   conf 0.60-0.65: 33% WR (still below breakeven)
            #   conf >= 0.65: 50% WR (actionable edge)
            # POR QUÉ: 0.65 is the inflection point where WR crosses 50%.
            # ═══════════════════════════════════════════════════════════════
            'sophia_win_prob_min': 0.50,  # FORENSIC-V150: Lowered from 0.65→0.50 (0.65 blocked 170+ signals, actual WR=46% not 100%)
            'sophia_win_prob_high': 0.75,
            'sophia_win_prob_supreme': 0.88,
            'sophia_superposition_divine': 0.72,
            'sophia_superposition_harmonic': 0.58,
            'sophia_resonance_index': 0.52,
            'sophia_butterfly_force': 1.2,
            'sophia_path_score': 0.65,
            'sophia_hurst_trend': 0.55,
            'sophia_hurst_mean_rev': 0.42,
            'sophia_whale_ratio': 5.0,
        }

    # === STRATEGY SETTINGS (Nesting required by loader) ===
    class Strategies:
        
        # ================================================================
        # 🏆 GOLDEN BASELINE — FROZEN PARAMETERS (2024-04-24 to 2024-05-09)
        # QUÉ: Configuración protegida que logró 88.2% WR (15W/2L).
        # POR QUÉ: Estabilidad demostrada en micro-cuentas de $13 USD.
        # PARA QUÉ: Evitar degradación por sobre-optimización o deriva.
        # ================================================================
        GOLDEN_BASELINE_LOCKED = True # SET TO FALSE TO ALLOW EVOLUTION
        
        # ════════════════════════════════════════════════════════════════
        # 🏆 GOLDEN EXIT SYSTEMS — PROTEGIDOS (No modificar sin auditoría)
        # QUÉ: Los 2 sistemas de cierre con mejor rendimiento del sistema.
        # POR QUÉ: FLIP_EXIT y TURBO_BE son los exits con WR más alto
        #   y mejor relación PnL neto, validados en producción.
        # PARA QUÉ: Preservar y documentar para que nunca se degraden.
        # CÓMO: FLIP_EXIT cierra atómicamente al detectar cambio de
        #   dirección. TURBO_BE protege capital cuando peak PnL alcanza
        #   50% (SCALP) o 60% (SWING) del TP y luego retrocede.
        # CUÁNDO: FLIP_EXIT en generate_order() de RiskManager.
        #         TURBO_BE en check_stops() de RiskManager.
        # DÓNDE: risk/risk_manager.py → L1630 (FLIP), L2311 (TURBO)
        # QUIÉN: RiskManager (Risk Manager role)
        # ════════════════════════════════════════════════════════════════
        GOLDEN_EXITS = {
            'FLIP_EXIT': {
                'enabled': True,
                'description': 'Atomic direction flip closure — closes existing position when opposite signal arrives',
                'uses_limit_bbo': True,
                'location': 'risk_manager.py::generate_order() L1630',
            },
            'TURBO_BE': {
                'enabled': True,
                'description': 'Turbo breakeven — locks in fee×2 when peak PnL reaches threshold, exits if price crashes back',
                'scalping_threshold_pct_of_tp': 0.80,  # [FORENSIC-V90.2] Raised from 50% to 80% — stops intercepting winning trades
                'swing_threshold_pct_of_tp': 0.60,     # 60% of TP target
                'min_scalping_pct': 0.48,              # Floor (80% of 0.60% TP)
                'min_swing_pct': 1.00,                 # Floor
                'fee_buffer_formula': 'fee_buffer × 2',  # [FORENSIC-V90.2] Guarantees meaningful net profit
                'location': 'risk_manager.py::check_stops() L2529/L2675',
            },
        }
        

        
        # ════════════════════════════════════════════════════════════════
        # 📉 DCA AUTOMÁTICO SWING — PROMEDIAR PRECIO ESCALONADO
        # QUÉ: Sistema de Dollar Cost Averaging automático para posiciones
        #   Swing que están en drawdown, aprovechando el 70% de margen libre.
        # POR QUÉ: Con $13 USD, una posición Swing usa solo 30% del silo
        #   (≈$1.56 margen). El 70% restante (≈$3.64) puede usarse para
        #   promediar el precio de entrada si el trade va en contra.
        # PARA QUÉ: Reducir el avg_price → acercar el TP → recuperar
        #   capital más rápido cuando el mercado revierte a la media.
        # CÓMO: 3 layers escalonados a -2%, -4%, -6% de drawdown.
        #   Cada layer usa una fracción decreciente del margen disponible.
        # CUÁNDO: Evaluado en cada tick de check_stops() para posiciones Swing.
        # DÓNDE: Config.Strategies.DCA → core/swing_dca_engine.py → risk_manager.py
        # QUIÉN: Risk Manager (sizing) + SwingDCAEngine (trigger logic)
        # ════════════════════════════════════════════════════════════════
        class DCA:
            ENABLED = True
            MAX_LAYERS = 3                          # Máximo 3 entradas DCA por posición
            TRIGGERS = [-0.020, -0.040, -0.060]     # Umbrales: -2%, -4%, -6% drawdown
            SIZE_MULTS = [0.25, 0.30, 0.35]         # % del margen disponible Swing por layer
            COOLDOWN_SECONDS = 1800                 # 30 min mínimo entre DCAs
            REGIME_BLOCK_BEAR = True                # Bloquear DCA LONG en Bear market
            ATR_SAFETY_MULT = 2.5                   # Bloquear si ATR > 2.5x normal (cisne negro)
            RECALC_TP = True                        # Recalcular TP tras promediar
            RECALC_SL = False                       # NO ensanchar SL (mantener riesgo controlado)
            MIN_MARGIN_FOR_DCA = 0.50               # Mínimo $0.50 margen libre para DCA
        # ═══════════════════════════════════════════════════════════════
        # AUDIT FIX: SCALPING_PARAMS and SWING_PARAMS REMOVED here.
        # POR QUÉ: Estas definiciones eran SIEMPRE sobreescritas por
        #   L918-920: Config.Strategies.SCALPING_PARAMS = Config.Horizons.Scalping
        # PARA QUÉ: Eliminar código muerto que confunde y puede causar
        #   divergencias silenciosas (ej: strength_threshold 0.55 vs 0.45).
        # DÓNDE: La ÚNICA fuente de verdad es Config.Horizons.Scalping/Swing
        #   (asignada a Strategies.SCALPING_PARAMS/SWING_PARAMS en L918-920).
        # ═══════════════════════════════════════════════════════════════
        # SCALPING_PARAMS → assigned from Config.Horizons.Scalping at module bottom
        # SWING_PARAMS → assigned from Config.Horizons.Swing at module bottom

        # ML Strategy settings
        ML_RETRAIN_INTERVAL = 240   
        ML_MIN_CONFIDENCE = 0.015   
        ML_LOOKBACK_BARS = 5000     
        ML_INCREMENTAL_UPDATE_BARS = 30 
        ML_ORACLE_VERBOSE = False   
        
        # ════════════════════════════════════════════════════════════════
        # 🧠 ML THRESHOLDS — ERADICATING MAGIC NUMBERS (PHASE 2)
        # QUÉ: Centralización de umbrales duros extraídos de ml_strategy.py.
        # POR QUÉ: Para permitir que el motor Genotipo/Evolutivo pueda
        #   mutar estos valores dinámicamente según el régimen de mercado.
        # PARA QUÉ: Evolutividad real. Si estos valores están en config,
        #   el bot puede aprender que en Bear Market necesita más confidence.
        # ════════════════════════════════════════════════════════════════
        ML_THRESHOLDS = {
            # Regime Detection (ml_strategy.py L730-760)
            'regime_adx_trend': 25,
            'regime_trend_strength': 0.025,
            'regime_atr_trend_max': 0.03,
            'regime_vol_volatility_max': 0.5,
            'regime_atr_volatile_min': 0.035,
            'regime_rsi_std_volatile': 18,
            'regime_price_vol_volatile': 0.035,
            'regime_adx_range_max': 20,
            'regime_rsi_std_range_max': 10,
            'regime_atr_range_max': 0.015,
            'regime_atr_zombie_1': 0.0005,
            'regime_spread_zombie': 0.0002,
            'regime_ident_bars_zombie': 0.85,
            'regime_atr_zombie_2': 0.0015,
            'mixed_regime_max_score': 0.60,
            
            # Confidence & Transitions (ml_strategy.py L827, L940)
            'confidence_regime_change': 0.55,
            'hmm_transition_risk_high': 0.40,
            
            # Decay & Exits (ml_strategy.py L3105-3109)
            'decay_confidence_long': 0.40,
            'decay_confidence_short': 0.60,
            
            # Filters
            'min_success_rate_excellent': 0.70,
            'min_success_rate_poor': 0.20,
            'clash_threshold': 0.85,
            'final_confidence_entry': 0.60,
            'final_confidence_strong': 0.70,
        }
        
        # Mean Reversion parameters
        STAT_WINDOW = 20
        STAT_Z_ENTRY = 2.0
        # FORENSIC-V139: Require overshoot to cover fees. Was 0.0.
        STAT_Z_EXIT = -0.3
        
        # --- PHASE 4-6 MATH PARAMETERS ---
        # Statistical
        STAT_RANSAC_WINDOW = 50       # Window for Robust Regression
        STAT_HURST_LAG = 20           # Lag for Hurst Exponent (Trend vs MeanRev)
        STAT_HURST_THRESHOLD = 0.5    # 0.5 = Random Walk
        
        # ML / Risk
        ML_KELLY_FRACTION = 1.0       # [ShadowDarwin] 100% Full Kelly for Aggressive Exponential Compounding
        ANALYTICS_EXPECTANCY_WINDOW = 20 # Rolling window for Kill Switch
        
        # Adaptive Technical
        TECH_DYNAMIC_RSI_VOL_THRESHOLD = 0.005 # 0.5% ATR for band expansion

        # ════════════════════════════════════════════════════════════════
        # 🧬 HYPER-EVOLVER-V2 GOLDEN GENOTYPE — PERMANENT MUTATIONS
        # QUÉ: Parámetros óptimos descubiertos por Optuna (50 trials bayesianos).
        # POR QUÉ: 80% WR con capital $13, convergencia en top-5 trials.
        # PARA QUÉ: Override permanente que technical.py consume en get_symbol_params().
        # CÓMO: Config.Mutations leído en technical.py L301-305 para Gen 0.
        # CUÁNDO: Siempre activo (producción y backtest).
        # DÓNDE: config.py → consumido por strategies/technical.py
        # QUIÉN: Hyper-Evolver Optuna Engine
        # ════════════════════════════════════════════════════════════════

    
        # ════════════════════════════════════════════════════════════════
        # 📐 TECHNICAL THRESHOLDS — ERADICATING MAGIC NUMBERS (PHASE 2)
        # QUÉ: Centralización de umbrales duros extraídos de technical.py.
        # POR QUÉ: Extraer constantes mágicas (1.5, 0.85, 0.025) que bloqueaban
        #   la evolución y hardcodeaban el comportamiento de Sophia y Riesgo.
        # PARA QUÉ: Mutaciones dinámicas sobre setups y filtros Oracle.
        # ════════════════════════════════════════════════════════════════
        TECHNICAL_THRESHOLDS = {
            'rsi_pullback_uptrend': 40,
            'rsi_rally_downtrend': 60,
            'rsi_extreme_low': 30,
            'rsi_extreme_high': 70,
            'vol_ratio_btc': 1.2,
            'vol_ratio_alts': 1.1,
            'vol_ratio_expansion': 1.5,
            'vol_ratio_high': 1.2,
            'vol_ratio_low': 0.8,
            'volatility_gate_pct': 0.025,
            'bb_pos_lower_prox': 0.25,
            'bb_pos_upper_prox': 0.75,
            # AUDIT FIX: Synced with GlobalThresholds above
            'sophia_win_prob_min': 0.50,  # SSOT: synced from GlobalThresholds (FORENSIC-V150)
            'sophia_win_prob_high': 0.72,
            'sophia_win_prob_supreme': 0.85,
            'sophia_superposition_divine': 0.70,
            'sophia_superposition_harmonic': 0.55,
            'sophia_resonance_index': 0.50,
            'sophia_butterfly_force': 1.2,
            'sophia_path_score': 0.65,
            'sophia_hurst_trend': 0.55,
            'sophia_hurst_mean_rev': 0.42,
            'sophia_whale_ratio': 5.0,
        }
    # ========================================================================
    # === BIDIRECTIONAL INTELLIGENCE MODULE (Oportunidad Dual) ===
    # ========================================================================
    class DualDirectional:
        # UA3: Umbrales mínimos de señal (SL/SC) e ISN por activo
        THRESHOLDS = {
            'BTC/USDT': {'sl_min': 72, 'isn_long': 10, 'sc_min': 78, 'isn_short': -10},
            'ETH/USDT': {'sl_min': 68, 'isn_long': 10, 'sc_min': 74, 'isn_short': -10},
            'BNB/USDT': {'sl_min': 65, 'isn_long': 10, 'sc_min': 70, 'isn_short': -10},
            'SOL/USDT': {'sl_min': 65, 'isn_long': 10, 'sc_min': 70, 'isn_short': -10},
            'XRP/USDT': {'sl_min': 68, 'isn_long': 10, 'sc_min': 75, 'isn_short': -10},
            'DEFAULT_T3': {'sl_min': 65, 'isn_long': 15, 'sc_min': 72, 'isn_short': -15},
            'DEFAULT_T4': {'sl_min': 78, 'isn_long': 20, 'sc_min': 85, 'isn_short': -20}
        }
        
        # UA5: Multiplicadores ATR para Stops (Long vs Short)
        ATR_STOP_MULT = {
            'BTC/USDT': {'long': 1.5, 'short': 1.8},
            'ETH/USDT': {'long': 1.7, 'short': 2.0},
            'BNB/USDT': {'long': 2.0, 'short': 2.3},
            'SOL/USDT': {'long': 2.0, 'short': 2.5},
            'XRP/USDT': {'long': 2.0, 'short': 2.3},
            'DEFAULT_T3': {'long': 2.0, 'short': 2.5},
            'DEFAULT_T4': {'long': 2.5, 'short': 3.0}
        }
        
        # UA5: Factor Kelly Asimétrico (Reducción de exposición para cortos)
        KELLY_FACTOR_SHORT = {
            'BTC/USDT': 0.90,
            'ETH/USDT': 0.85,
            'BNB/USDT': 0.75,
            'SOL/USDT': 0.70,
            'XRP/USDT': 0.65,
            'DEFAULT_T3': 0.60,
            'DEFAULT_T4': 0.40
        }
        
        # CIERRE UNIFICADO (Parte VI): Nivel de R1 y R2 en ATR
        CLOSURE_ATR_TARGETS = {
            'BTC/USDT': {'long_r1': 1.5, 'short_r1': 1.2, 'long_r2': 3.0, 'short_r2': 2.0},
            'ETH/USDT': {'long_r1': 1.8, 'short_r1': 1.4, 'long_r2': 3.0, 'short_r2': 2.0},
            'SOL/USDT': {'long_r1': 2.0, 'short_r1': 1.0, 'long_r2': 4.0, 'short_r2': 1.5},
            'XRP/USDT': {'long_r1': 1.5, 'short_r1': 1.0, 'long_r2': 3.0, 'short_r2': 2.0},
            'DEFAULT_T3': {'long_r1': 1.5, 'short_r1': 1.2, 'long_r2': 3.0, 'short_r2': 2.0},
            'DEFAULT_T4': {'long_r1': 1.5, 'short_r1': 1.0, 'long_r2': 3.0, 'short_r2': 2.0}
        }
        
        # CIERRE UNIFICADO (Parte VI): Porcentajes de Cierre
        CLOSURE_PCT = {
            'long_r1_pct': 0.25,
            'short_r1_pct': 0.50,
            'long_r2_pct': 0.30,
            'short_r2_pct': 0.30,
            'long_runner': 0.35,
            'short_runner': 0.20
        }
        
        # Trailing Stop en ATR
        TRAILING_ATR = {
            'BTC/USDT': {'long': 1.5, 'short': 1.0},
            'ETH/USDT': {'long': 1.7, 'short': 1.2},
            'SOL/USDT': {'long': 2.0, 'short': 1.5},
            'DEFAULT_T4': {'long': 2.5, 'short': 2.0}
        }
        
        # Funding Rate Rules (Base 8h)
        FUNDING_EXTREME_POS = 0.0005 # 0.05%
        FUNDING_MODERATE_POS = 0.0001 # 0.01%
        FUNDING_MODERATE_NEG = -0.00005 # -0.005%
        FUNDING_EXTREME_NEG = -0.0002 # -0.02%
    # ========================================================================
    # === SCORING & BREAKEVEN (Phase 1 Antifragil) ===
    # ========================================================================
    class Scoring:
        MIN_PASS_SCORE = 75
        BREAKEVEN_SAFETY_MARGIN_MULTIPLIER = 1.5

    # Phase 99: WandB Tracking
    WANDB_ENTITY = "jhonala-none"
    
    # ========================================================================
    # INSTITUTIONAL POLICY ENFORCEMENT (Phase 2.6)
    # ========================================================================
    WANDB_PROJECT = "trader-gemini"

    # ========================================================================
    # === OBSERVABILITY & ANALYTICS (Phase 4) ===
    # ========================================================================
    class Observability:
        # --- TELEGRAM ---
        TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'False').lower() == 'true'
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # --- EMAIL (SMTP) ---
        EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
        SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', 587))
        EMAIL_USER = os.getenv('EMAIL_FROM', '')
        EMAIL_PASS = os.getenv('EMAIL_PASSWORD', '')
        EMAIL_RECEIVER = os.getenv('EMAIL_TO', '')
        
        # --- THRESHOLDS & ALERTS ---
        ALERT_MAX_DRAWDOWN = 0.05      # 5% Drawdown triggers warning
        ALERT_MIN_SHARPE = 1.2         # Sharpe < 1.2 triggers warning
        ALERT_CRITICAL_ERRORS = True   # Alertas por fallos de API/Engine
        
        # --- REPORTING ---
        PDF_REPORTS_ENABLED = True
        REPORT_FREQUENCY_DAYS = 1      # Daily reports
        
        # --- PRIORITIES ---
        MIN_LOG_LEVEL_NOTIFY = 2 # 2=Warning
        
        # --- ENHANCED NOTIFICATION SETTINGS (Phase 4.5) ---
        # QUÉ: Controles granulares para cada tipo de notificación.
        # POR QUÉ: No todas las notificaciones son necesarias en todo momento.
        # PARA QUÉ: Personalizar qué información recibir sin tocar código.
        NOTIFICATION_TRADE_OPEN = True          # Notificar apertura de trades
        NOTIFICATION_TRADE_CLOSE = True         # Notificar cierre de trades
        NOTIFICATION_RISK_ALERTS = True         # Alertas de riesgo (drawdown, kill switch)
        NOTIFICATION_DAILY_REPORT = True        # Reporte diario de performance
        NOTIFICATION_PERFORMANCE_UPDATE = True  # Updates periódicos de performance
        NOTIFICATION_UPDATE_FREQUENCY_MIN = 60  # Frecuencia updates (minutos)
        
        # Alert thresholds
        NOTIFICATION_DRAWDOWN_WARNING = 0.005   # 0.5% drawdown → WARNING
        NOTIFICATION_DRAWDOWN_CRITICAL = 0.010  # 1.0% drawdown → CRITICAL
        NOTIFICATION_LOSS_STREAK_ALERT = 3      # 3 pérdidas consecutivas → alerta
        NOTIFICATION_MAX_MESSAGES_PER_MIN = 300 # Rate limit ampliado para permitir TODO EL SPAM

    class Analytics:
        RISK_FREE_RATE = 0.02 
        TRADING_DAYS = 365
        SORTINO_MIN_RETURN = 0.0
        WINRATE_LOOKBACK_TRADES = 50

    # ========================================================================
    # === GLOBAL SETTINGS ===
    # ========================================================================
    # DEBUG_TRACE_ENABLED already defined at L59 — DO NOT RE-DEFINE HERE
    # (Removed duplicate to prevent shadowing. Ref: Frankenstein Audit Phase 2)
    
    # ========================================================================
    # === EXECUTION ENGINE — "MAKER PROFIT, TAKER PANIC" (BBO Architecture) ===
    # ========================================================================
    # QUÉ: Configuración del motor de ejecución Limit BBO.
    # POR QUÉ: Market orders pagan Taker 0.0375%, Limit BBO paga Maker 0.02%.
    #   Con $13 USD y 10+ trades/día, la diferencia es ~4% de capital en 15 días.
    # PARA QUÉ: Maximizar retención de capital en micro-cuenta.
    # CÓMO: Entries y exits normales usan LIMIT BBO con Post-Only (GTX).
    #   Solo emergencias (Kill Switch) usan MARKET.
    # CUÁNDO: En cada orden generada por RiskManager y ejecutada por BinanceExecutor.
    # DÓNDE: Config.Execution.* referenciado por risk_manager.py y binance_executor.py.
    # QUIÉN: Arquitecto Senior + Risk Manager.
    class Execution:
        # ── BBO Post-Only Configuration ──
        USE_LIMIT_BBO_ENTRIES = True       # Entries use LIMIT at Best Bid/Offer
        USE_LIMIT_BBO_EXITS = True         # [CIRUGÍA #7] Enabled to save 0.035% Taker fee. Scalping micro-profits CANNOT survive Taker fees.
        USE_LIMIT_PROTECTIVE_ORDERS = True # SL/TP on exchange: STOP (Limit) instead of STOP_MARKET
        POST_ONLY_GTX = True               # GTX = Post-Only guarantee → Maker fee always
        
        # ── Chase / Fallback Configuration ──
        MAX_CHASE_ATTEMPTS = 3             # Max repricing attempts before MARKET fallback
        CHASE_TIMEOUT_SECONDS = 5          # TTL per chase attempt for exit orders
        ENTRY_TTL_SECONDS = 30             # TTL for entry LIMIT orders
        
        # ── Emergency Override ──
        EMERGENCY_FALLBACK_MARKET = True   # Kill Switch / Flash Crash → MARKET nuclear
        
        # ── Protective Order Pricing ──
        # For STOP (Limit): How far below/above trigger to set the limit price
        # 0.001 = 0.1% tolerance → if SL triggers at $100, limit at $99.90
        STOP_LIMIT_TOLERANCE_PCT = 0.001
        # For TAKE_PROFIT (Limit): Set at exact trigger (Post-Only)
        TP_LIMIT_TOLERANCE_PCT = 0.0005    # [FORENSIC V99] Allow TP to cross spread by 0.05% to avoid Zombie Trades

    # ========================================================================
    # === AEGIS-ULTRA PROTOCOL (Hardware & Math) ===
    # ========================================================================
    class Aegis:
        ENABLED = True
        CORE_PINNING = True        # Enable Processor Affinity (Ryzen 5700U)
        PROCESS_PRIORITY = "HIGH"  # Win32 High Priority Class
        USE_AVX2 = True            # Enable Numba Vectorization
        ZERO_COPY_DATA = True      # Enable RingBuffer direct access


    # ========================================================================
    # === SNIPER STRATEGY SETTINGS (ALL OR NOTHING PROTOCOL) ===
    # ========================================================================
    class Sniper:
        """
        HIGH-RISK configuration for $12 → $240 target.
        WARNING: This configuration has ~99% probability of total loss.
        """
        # ✅ DYNAMIC REGIME ADAPTATION (EVOLUTIONARY SNIPER)
        # ==========================================================
        # The bot decides autonomy level based on Market Regime.
        # No more manual "Sniper Mode" switch.
        
        DYNAMIC_ADAPTATION = True
        ENABLED = True # Master Switch for Sniper Mode
        
        # REGIME MAP: Defines aggression per market state
        # key: Regime Name 
        # value: (Leverage Limit, Threshold Modifier, Position Scale)
        # FORENSIC FIX #7: Leverage floors raised for micro-accounts ($13).
        # POR QUÉ: Con $13 y leverage 1-3x, el notional ($13-$39) apenas supera
        #   el mínimo de Binance ($5). Esto mata el sizing y genera rechazos.
        # CÓMO: Mínimo 8x en RANGING/CHOPPY para garantizar notional ≥ $40.
        #   Bear/Zombie mantienen 1x como defensa (no operar, no arriesgar).
        REGIME_MAP = {
            'TRENDING_BULL': {'leverage': 10, 'threshold_mod': -0.05, 'scale': 1.0}, # SNIPER BEHAVIOR (Full power)
            'TRENDING_BEAR': {'leverage': 5,  'threshold_mod': +0.10, 'scale': 0.3}, # DEFENSE (Reduced, not zero)
            'RANGING':       {'leverage': 8,  'threshold_mod': +0.00, 'scale': 0.8}, # SCALPING (Must be viable)
            'CHOPPY':        {'leverage': 1,  'threshold_mod': +0.05, 'scale': 0.0}, # RESCUE PROTOCOL: Choppy = DEAD (1x)
            'ZOMBIE':        {'leverage': 1,  'threshold_mod': +1.00, 'scale': 0.0}, # DEAD MARKET (No Trade)
        }
        
        # MAINNET SWITCH - Set to True for REAL trading
        USE_MAINNET = True   # LIVE PRODUCTION MODE
        
        # WHITELIST - Only ultra-high liquidity pairs (low spread)
        WHITELIST = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        
        # DYNAMIC LEVERAGE (ATR-based)
        MIN_LEVERAGE = 5     # FORENSIC FIX: was 3 — micro-account needs 5x+ for viable notional
        MAX_LEVERAGE = 10    # FORENSIC FIX: was 8 — aligned with Config.BINANCE_LEVERAGE
        DEFAULT_LEVERAGE = 8 # FORENSIC FIX: was 5 — default must produce viable notional
        
        # FEES (Binance Futures) — SYNCED with global Config.BINANCE_*_FEE_BNB
        TAKER_FEE = 0.000375  # [UNIFIED] 0.0375% with BNB discount (was 0.0005 — CONFLICTING)
        MAKER_FEE = 0.0002    # 0.02% (matches Config.BINANCE_MAKER_FEE_BNB)
        
        # TRADE VALIDATION - FOR MICRO ACCOUNT (FEE PROTECTION)
        MIN_RR_RATIO = 2.5       # Increased for fee drag buffer
        MIN_PROFIT_AFTER_FEES = 0.012  # 1.2% minimum net profit required to exit
        
        # VOLUME ANOMALY DETECTION (Whale detection)
        VOLUME_LOOKBACK_BARS = 1440  # 24h of 1m candles
        VOLUME_SIGMA_THRESHOLD = 3.0  # 3 standard deviations
        STAT_Z_ENTRY = 2.0  # Entry Threshold (Standard Deviations)
        STAT_Z_EXIT = 0.0   # Exit Threshold (Mean Reversion)
        
        # Phase 6: Permissive Mode (Demo Competition)
        PERMISSIVE_MODE = True  # Enable lower thresholds in Demo
        PERMISSIVE_CONFIDENCE_THRESHOLD = 0.60 # Lower barrier for entry
        DEMO_EQUAL_WEIGHTING = 0.05 # 5% Fixed Size for fair comparison
        # ORDER BOOK ANALYSIS
        ORDERBOOK_DEPTH = 20  # Levels to analyze
        IMBALANCE_THRESHOLD = 0.3  # 30% imbalance for signal
        MAX_SPREAD_PCT = 0.05  # Skip if spread > 0.05%
        
        # CONFLUENCE REQUIREMENTS
        CONFLUENCE_THRESHOLD = 4  # All 4 layers must pass
        
        # TECHNICAL INDICATORS (Layer A)
        RSI_PERIOD = 14
        RSI_OVERSOLD = 30
        RSI_OVERBOUGHT = 70
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        BB_PERIOD = 20
        BB_STD = 2.0
        
        # SESSION HOURS (UTC) - Volatility Filter
        ACTIVE_SESSIONS = {
            'london_open': 8,
            'london_close': 16,
            'ny_open': 13,
            'ny_close': 21
        }
        # SESSION FILTER
        # If True, only trades during London/NY overlaps (08:00 - 22:00 UTC)
        REQUIRE_ACTIVE_SESSION = False # Set to False for 24/7 Crypto mode
        
        # COMPOUNDING
        COMPOUND_PROFITS = True  # Reinvest 100% of profits
        
        # TESTNET VALIDATION
        TESTNET_TRADES_REQUIRED = 3  # Successful trades before MAINNET


    @classmethod
    def check_types(cls):
        """Type validation for critical parameters"""
        try:
            assert isinstance(cls.BINANCE_LEVERAGE, int), "Leverage must be int"
            assert 1 <= cls.BINANCE_LEVERAGE <= 125, "Leverage out of bounds"
            
            assert isinstance(cls.MAX_RISK_PER_TRADE, float), "Risk must be float"
            assert 0 < cls.MAX_RISK_PER_TRADE <= 1.0, "Risk must be 0-1"
            
            assert isinstance(cls.INITIAL_CAPITAL, (int, float)), "Initial capital must be number"
            assert cls.INITIAL_CAPITAL > 0, "Capital must be positive"
            
            return True
        except AssertionError as e:  # Note: Python built-in is AssertionError
            print(f"❌ CONFIG TYPE ERROR: {e}")
            sys.exit(1)

# ============================================================================
# CONFIGURATION VALIDATION (Fail Fast on Missing Credentials)
# ============================================================================
def validate_config():
    """
    Validates that required API credentials are present.
    Fails fast with clear error messages if configuration is incomplete.
    """
    errors = []
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        errors.append("❌ ERROR: .env file not found!")
        errors.append("   → Solution: Copy .env.example to .env and fill in your API keys")
    
    # Determine which keys are required based on mode
    if Config.BINANCE_USE_DEMO or Config.BINANCE_USE_TESTNET:
        if not Config.BINANCE_USE_FUTURES:
            if not Config.BINANCE_TESTNET_API_KEY:
                errors.append("❌ ERROR: BINANCE_TESTNET_API_KEY not found in .env")
        else:
            if not Config.BINANCE_DEMO_API_KEY:
                errors.append("❌ ERROR: BINANCE_DEMO_API_KEY not found in .env")
    else:
        if not Config.BINANCE_API_KEY:
            errors.append("❌ ERROR: BINANCE_API_KEY not found in .env")
    
    # Configuration warnings
    warnings = []
    if Config.BINANCE_LEVERAGE > 25:
        warnings.append(f"⚠️  WARNING: Leverage set to {Config.BINANCE_LEVERAGE}x (exceeds recommended 25x)")
    
    if Config.MAX_RISK_PER_TRADE > 0.05:
        warnings.append(f"⚠️  WARNING: Risk per trade is {Config.MAX_RISK_PER_TRADE*100}%")
    
    # Print all
    if warnings:
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION WARNINGS")
        for warning in warnings: print(warning)
        print("="*70 + "\n")
    
    if errors:
        print("\n" + "="*70)
        print("🚨 CONFIGURATION ERRORS - BOT CANNOT START")
        for error in errors: print(error)
        print("="*70)
        sys.exit(1)
    
    return True

def validate_institutional_policy():
    """
    DF-A3: Institutional Grade Policy Validation.
    Enforces strict risk limits when operating in PRODUCTION (Mainnet).
    """
    is_production = not (Config.BINANCE_USE_TESTNET or Config.BINANCE_USE_DEMO)
    
    if is_production:
        errors = []
        # 1. Sniper Mode Safety
        if Config.Sniper.ENABLED and not Config.Sniper.USE_MAINNET:
             # Sniper is Enabled but USE_MAINNET is False -> Contradiction or Safety Catch
             # If we are in Production, we cannot use Testnet Sniper settings.
             # We explicitly fail to prevent accidental high-risk trading.
             if not os.getenv("FORCE_SNIPER_MAINNET") == "TRUE":
                 errors.append("❌ SAFETY: Sniper Mode is ENABLED in Production without FORCE_SNIPER_MAINNET=TRUE.")
        
        # 2. Leverage Caps
        if Config.BINANCE_LEVERAGE > 5 and not Config.Sniper.ENABLED:
            errors.append(f"❌ RISK: Leverage {Config.BINANCE_LEVERAGE}x exceeds Institutional Limit (5x).")
            
        # 3. Risk Limits
        if Config.MAX_RISK_PER_TRADE > 0.051: # [SOVEREIGN-DEPLOY] Elevated Max Risk to 5% for micro-accounts
            errors.append(f"❌ RISK: Max Risk {Config.MAX_RISK_PER_TRADE*100}% exceeds Institutional Limit (5%).")
            
        if errors:
            print("\n" + "="*70)
            print("🛡️ INSTITUTIONAL POLICY VIOLATION")
            for e in errors: print(e)
            print("="*70)
            sys.exit(1)
    
    return True

# Unificación de alias para evitar divergencias entre Horizons y Strategies (SSOT)
Config.Strategies.MICROSCALPING_PARAMS = Config.Horizons.Microscalping
Config.Strategies.SCALPING_PARAMS = Config.Horizons.Scalping
Config.Strategies.SWING_PARAMS = Config.Horizons.Swing
Config.registry = OmniscientRegistry(Config)

# ═══════════════════════════════════════════════════════════════
# SSOT SYNC: Sophia thresholds → single source from GlobalThresholds
# POR QUÉ: Antes había valores duplicados en GlobalThresholds y
#   TECHNICAL_THRESHOLDS que podían divergir silenciosamente.
# PARA QUÉ: Un cambio en GlobalThresholds se propaga a todas partes.
# ═══════════════════════════════════════════════════════════════
_sophia_keys = [k for k in Config.Horizons.GlobalThresholds if k.startswith('sophia_')]
for _k in _sophia_keys:
    Config.Strategies.TECHNICAL_THRESHOLDS[_k] = Config.Horizons.GlobalThresholds[_k]

# Run validation on import
Config.check_types()
# validate_config() # Called internally or explicitly in main
validate_institutional_policy() # Enforce Policy Verification on Import
