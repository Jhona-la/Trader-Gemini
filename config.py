import os
import sys
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

class Config(metaclass=EncryptedConfigMeta):
    # ========================================================================
    # GLOBAL SETTINGS
    # ========================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEBUG_TRACE_ENABLED = False


    # ========================================================================
    # BINANCE API CREDENTIALS (Loaded from .env file)
    # ========================================================================
    
    # 🔐 PHASE 17: ENCRYPTED KEYS
    # Keys are now managed dynamically by EncryptedConfigMeta.
    # We do not define them here as static attributes to keep RAM clean.
    
    # 🔐 PHASE 17 (SOVEREIGN-DEPLOY): HARDENED PRODUCTION KEYS
    # OVERRIDEN BY BLOCK G PROTOCOL: PAPER TRADING ENABLED
    BINANCE_USE_TESTNET = True
    BINANCE_USE_DEMO = True
    
    # === BINANCE FUTURES SETTINGS ===
    # Default: USDT-Margined Futures (standard). 
    # For COIN-Margined, code modifications in binance_executor would be needed (defaultType='delivery').
    # BUG #33 FIX: Changed default to False to allow Spot mode. CLI --mode argument will override this.
    BINANCE_USE_FUTURES = True  # Set to True to trade on Binance Futures instead of Spot
    BINANCE_LEVERAGE = 10  # Leverage for Futures trading (CONTROLLED: 10x for $13 to bypass $5 notional limit)
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
    # Total: 26 pairs (Institutional Basket)
    CRYPTO_FUTURES_PAIRS = [
        "ADA/USDT", "ARB/USDT", "ATOM/USDT", "AVAX/USDT", "BNB/USDT",
        "BTC/USDT", "DOGE/USDT", "DOT/USDT", "ETC/USDT", "ETH/USDT",
        "NEAR/USDT", "OP/USDT", "PAXG/USDT", "POL/USDT", "RENDER/USDT", 
        "SOL/USDT", "SUI/USDT", "TIA/USDT", "UNI/USDT", "WIF/USDT", 
        "XRP/USDT"
    ]
    # Total: 24 pairs (optimized for liquidity and volatility)
    
    # Auto-select correct pairs based on mode
    # BUG #14 FIX: Binance Testnet SPOT is UNRELIABLE (most pairs don't exist)
    # Solution: SPOT only works in PRODUCTION, Testnet/Demo users should use FUTURES
    # SINGLE SOURCE OF TRUTH: Initial capital for $13 micro-scalping strategy (SOVEREIGN-DEPLOY)
    INITIAL_CAPITAL = 13.0  # Base capital for sizing and HWM

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
    TIMEFRAME = TimeFrame.M1  # M1 Timeframe (Microscalping)
    MAX_SIGNAL_AGE = 60       # 60s for M1 timeframe to allow for fast execution
    
    # MULTI-SYMBOL EXPANSION: Top Liquidity Assets for Scalping
    # Defaulting to all Futures Pairs (24 total)
    TRADING_PAIRS = CRYPTO_FUTURES_PAIRS

    
    # Risk settings for Multi-Symbol Coordination
    MAX_CONCURRENT_POSITIONS = 26 # FORENSIC-V17: 26 (one per symbol) — NO LIMITS per user request
    COOLDOWN_PERIOD_SECONDS = 0    # FORENSIC-V17: 0s — allow immediate re-entry for micro-scalping
    MAX_POSITIONS_PER_SYMBOL = 1   # Still 1 per symbol to prevent double-spending on same asset
    
    # Position Sizing Configuration
    POSITION_SIZE_MICRO_ACCOUNT = 0.30   # GOD MODE: 30% per trade → $3.90 margin × 10x = $39 notional
    POSITION_SIZE_SMALL_ACCOUNT = 0.15   # Lowered from 20%
    
    # Trade Validation Thresholds
    MIN_PROFIT_AFTER_FEES = 0.003  # 0.3% minimum net profit
    MIN_RR_RATIO = 1.5             # 1.5:1 R:R minimum
    
    # Risk Management
    MAX_RISK_PER_TRADE = 0.10  # 10% of capital max per trade for $13 account (Exponential Growth)
    STOP_LOSS_PCT = 0.02       # 2% stop loss
    MAX_SLIPPAGE_PCT = 0.001   # 0.1% max slippage (Sovereign-Deploy)
    
    # === RISK CAPITOL HIERARCHY (USER RULE PRESERVATION) ===
    class Risk:
        MAX_DRAWDOWN = 15.0           # 15.0% max drawdown (SOVEREIGN LIMIT) - Increased from 1.5% for $13 micro-accounts
        DEFAULT_BOOTSTRAP_WR = 0.52 
        BOOTSTRAP_TRADES = 20
        MAX_RISK_PER_TRADE = 0.05  
        STOP_LOSS_PCT = 0.02       
        MAX_SLIPPAGE_PCT = 0.001
    
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
    STAT_WINDOW = 20
    STAT_Z_ENTRY = 1.5
    STAT_Z_EXIT = 0.0

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

    # === STRATEGY SETTINGS (Nesting required by loader) ===
    class Strategies:
        # Technical Strategy settings
        TECH_RSI_PERIOD = 14
        TECH_RSI_BUY = 35    
        TECH_RSI_SELL = 65   
        TECH_EMA_FAST = 20 # SHORT-TERM (Changed from 50)
        TECH_EMA_SLOW = 50 # MEDIUM-TERM (Changed from 200)
        TECH_ADX_THRESHOLD = 25
        TECH_BB_PERIOD = 20
        TECH_BB_STD = 2.0
        TECH_TP_PCT = 0.015
        TECH_SL_PCT = 0.02
        SYMMETRIC_SHORTS_SCALPING = True  # Habilita cortos en contra-tendencia para micro-cuentas
        SHORT_SL_MULTIPLIER = 1.2         # 20% tightening on SL for shorts (e.g. 0.5% / 1.2 = 0.41%)
        SHORT_TP_MULTIPLIER = 0.8         # 20% tightening on TP for shorts (e.g. 1.0% * 0.8 = 0.8%)
        
        # ================================================================
        # HORIZON-SPECIALIZED PARAMETERS (Phase Forensic-1)
        # QUÉ: Parámetros diferenciados para Scalping vs Swing
        # POR QUÉ: Un TP de 1.5% es inalcanzable en scalping 1min pero
        #   insuficiente para swing 4h. Cada horizonte necesita su propio DNA.
        # PARA QUÉ: Maximizar frecuencia de wins en Scalping (0.3-0.5% TP)
        #   y capturar movimientos grandes en Swing (2-5% TP).
        # CÓMO: technical.py lee self.horizon y selecciona el dict correcto.
        # ================================================================
        SCALPING_PARAMS = {
            'tp_pct': 0.0040,         # FORENSIC-V13: 0.40% TP — realista para 1M-5M (was 1.20% inalcanzable)
            'sl_pct': 0.0025,         # FORENSIC-V13: 0.25% SL — tight pero alcanzable, R:R = 1.6:1
            'rsi_period': 5,          # RSI ultra-rápido (5 velas)
            'rsi_buy': 35,            # FORENSIC FIX: was 25 — wider zone to capture more setups
            'rsi_sell': 65,           # FORENSIC FIX: was 75 — wider zone to capture more setups
            'bb_period': 10,          # Bollinger rápido
            'bb_std': 1.5,            # FORENSIC FIX: was 1.8 — tighter bands = more touches
            'ema_fast': 8,            # EMA rápida
            'ema_slow': 21,           # EMA lenta
            'ema_trend': 50,          # Trend filter ligero
            'atr_period': 7,          # ATR corto
            'adx_period': 7,          # ADX rápido
            'timeframes': ['1m', '5m', '15m'],  # Solo timeframes cortos
            'primary_tf': '5m',       # Timeframe principal
            'min_volume_ratio': 0.4,  # FORENSIC FIX: was 0.6 — less filtering for micro-account
            'cooldown_seconds': 15,   # FORENSIC FIX: aligned with global COOLDOWN_PERIOD_SECONDS
            'max_hold_bars': 120,     # FIX FORENSIC-7: 120 bars = 2h for TP to be reached
            'strength_threshold': 0.55, # FORENSIC-V13: Raised from 0.40 — filter coin-flip signals (52-54% win prob)
            'atr_sl_mult': 1.0,       # SL pegado (1x ATR)
            'atr_tp_mult': 2.0,       # TP 2x ATR (ratio 2:1)
            'sophia_refit': 50,       # Recalibrate clusters every 50 bars (M1/M5)
        }
        
        SWING_PARAMS = {
            'tp_pct': 0.035,          # 3.5% TP — captura de tendencia
            'sl_pct': 0.015,          # 1.5% SL — respira pero protege
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
            'timeframes': ['1h', '4h', '1d'],  # Solo timeframes largos
            'primary_tf': '1h',       # Timeframe principal
            'min_volume_ratio': 1.0,  # Volumen confirmado
            'cooldown_seconds': 3600, # 1h cooldown entre trades
            'max_hold_bars': 96,      # Max 96 velas (4 días en 1h)
            'strength_threshold': 0.55, # Umbral medio
            'atr_sl_mult': 2.0,       # SL amplio (2x ATR)
            'atr_tp_mult': 4.5,       # TP grande (4.5x ATR, ratio 2.25:1)
            'sophia_refit': 24,       # Recalibrate clusters once per day (H1/H4)
        }
        
        # ML Strategy settings
        ML_RETRAIN_INTERVAL = 240   
        ML_MIN_CONFIDENCE = 0.015   
        ML_LOOKBACK_BARS = 5000     
        ML_INCREMENTAL_UPDATE_BARS = 30 
        ML_ORACLE_VERBOSE = False   
        
        # Mean Reversion parameters
        STAT_Z_ENTRY = 1.5
        STAT_Z_EXIT = 0.0
        
        # --- PHASE 4-6 MATH PARAMETERS ---
        # Statistical
        STAT_RANSAC_WINDOW = 50       # Window for Robust Regression
        STAT_HURST_LAG = 20           # Lag for Hurst Exponent (Trend vs MeanRev)
        STAT_HURST_THRESHOLD = 0.5    # 0.5 = Random Walk
        
        # ML / Risk
        ML_KELLY_FRACTION = 0.1       # Sovereign-Deploy: Fractional Kelly (f*/10)
        ANALYTICS_EXPECTANCY_WINDOW = 20 # Rolling window for Kill Switch
        
        # Adaptive Technical
        TECH_DYNAMIC_RSI_VOL_THRESHOLD = 0.005 # 0.5% ATR for band expansion

    # Phase 99: WandB Tracking
    WANDB_ENTITY = "jhonala-none"
    # MAX_RISK_PER_TRADE overridden above -> 0.05
    
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
        NOTIFICATION_MAX_MESSAGES_PER_MIN = 30  # Rate limit Telegram API

    class Analytics:
        RISK_FREE_RATE = 0.02 
        TRADING_DAYS = 365
        SORTINO_MIN_RETURN = 0.0
        WINRATE_LOOKBACK_TRADES = 50

    # ========================================================================
    # === GLOBAL SETTINGS ===
    # ========================================================================
    DEBUG_TRACE_ENABLED = False
    #DEBUG_TRACE_ENABLED = True hace un seguimiento microscópico de todo lo que hace el bot. 
    #Lo que hace: Imprime en la consola cada vez que el código ENTRA ([ENTER]) y SALE ([EXIT]) de una función importante, y cuánto tiempo tardó.
    #Para qué sirve: Para encontrar "cuellos de botella" (funciones que tardan demasiado) o ver exactamente en qué línea se congela el programa.
    #Problema: Genera miles de líneas de texto por segundo, llenando la terminal de "ruido" y dificultando ver lo importante (precios, señales, errores).
    #Recomendación: Mantenlo en False (Apagado) para operar. Solo enciéndelo (True) si el bot se clava y no sabes por qué.
    
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
        USE_LIMIT_BBO_EXITS = True         # Normal exits (TP, trailing) use LIMIT BBO
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
        TP_LIMIT_TOLERANCE_PCT = 0.0       # 0% = limit AT trigger price (pure Maker)

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
            'CHOPPY':        {'leverage': 8,  'threshold_mod': +0.05, 'scale': 0.5}, # CAUTION (Was 1x = dead)
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
        
        # FEES (Binance Futures)
        TAKER_FEE = 0.0005  # 0.05%
        MAKER_FEE = 0.0002  # 0.02%
        
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

# Run validation on import
Config.check_types()
# validate_config() # Called internally or explicitly in main
validate_institutional_policy() # Enforce Policy Verification on Import
