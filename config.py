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
    DEBUG_TRACE_ENABLED = False

    # ========================================================================
    # BINANCE API CREDENTIALS (Loaded from .env file)
    # ========================================================================
    
    # 🔐 PHASE 17: ENCRYPTED KEYS
    # Keys are now managed dynamically by EncryptedConfigMeta.
    # We do not define them here as static attributes to keep RAM clean.
    
    # Binance Testnet (Spot)
    BINANCE_USE_TESTNET = os.getenv('BINANCE_USE_TESTNET', 'False').lower() == 'true'
    
    # Binance Demo Trading (Futures with virtual capital)
    BINANCE_USE_DEMO = os.getenv('BINANCE_USE_DEMO', 'False').lower() == 'true'
    
    # === BINANCE FUTURES SETTINGS ===
    # Default: USDT-Margined Futures (standard). 
    # For COIN-Margined, code modifications in binance_executor would be needed (defaultType='delivery').
    # BUG #33 FIX: Changed default to False to allow Spot mode. CLI --mode argument will override this.
    BINANCE_USE_FUTURES = True  # Set to True to trade on Binance Futures instead of Spot
    BINANCE_LEVERAGE = 3  # Leverage for Futures trading (CONTROLLED: 3x for $15)
    BINANCE_MARGIN_TYPE = "ISOLATED"  # Options: "ISOLATED" or "CROSS"
    BINANCE_TAKER_FEE_BNB = 0.000375 # 0.0375% (with BNB discount)
    
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
        "FIL/USDT", "INJ/USDT", "LINK/USDT", "LTC/USDT", "MATIC/USDT", 
        "NEAR/USDT", "OP/USDT", "PAXG/USDT", "POL/USDT", "RENDER/USDT", 
        "SOL/USDT", "SUI/USDT", "TIA/USDT", "UNI/USDT", "WIF/USDT", 
        "XRP/USDT"
    ]
    # Total: 24 pairs (optimized for liquidity and volatility)
    
    # Auto-select correct pairs based on mode
    # BUG #14 FIX: Binance Testnet SPOT is UNRELIABLE (most pairs don't exist)
    # Solution: SPOT only works in PRODUCTION, Testnet/Demo users should use FUTURES
    # SINGLE SOURCE OF TRUTH: Initial capital for $15 micro-scalping strategy
    INITIAL_CAPITAL = 15.0  # Base capital for sizing and HWM

    # === TRADING SETTINGS ===
    TIMEFRAME = TimeFrame.M1  # M1 Timeframe (Microscalping)
    MAX_SIGNAL_AGE = 60       # 60s for M1 timeframe to allow for fast execution
    
    # MULTI-SYMBOL EXPANSION: Top Liquidity Assets for Scalping
    # Defaulting to all Futures Pairs (24 total)
    TRADING_PAIRS = CRYPTO_FUTURES_PAIRS

    
    # Risk settings for Multi-Symbol Coordination
    MAX_CONCURRENT_POSITIONS = 3  # Maximum simultaneous trades for $15 account
    COOLDOWN_PERIOD_SECONDS = 1800 # 30 minutes between trades on same symbol
    MAX_POSITIONS_PER_SYMBOL = 1   # One layer per symbol for safety
    
    # Position Sizing Configuration
    POSITION_SIZE_MICRO_ACCOUNT = 0.18   # Fine-tuned to 18% to pass Risk Shield (Target < 15% DD)
    POSITION_SIZE_SMALL_ACCOUNT = 0.15   # Lowered from 20%
    
    # Trade Validation Thresholds
    MIN_PROFIT_AFTER_FEES = 0.003  # 0.3% minimum net profit
    MIN_RR_RATIO = 1.5             # 1.5:1 R:R minimum
    
    # Risk Management
    MAX_RISK_PER_TRADE = 0.01  # 1% of capital
    STOP_LOSS_PCT = 0.02       # 2% stop loss
    
    # === INTELLIGENT REVERSE (FLIPPING) PARAMETERS (Phase 5) ===
    # PROFESSOR METHOD:
    # QUÉ: Umbrales de viabilidad para el cambio de dirección táctico.
    # POR QUÉ: Evita "whipsaws" (ser sierra) en mercados picados/laterales.
    # PARA QUÉ: Garantizar que el Flip tenga un valor esperado positivo real.
    FLIP_MIN_ATR_PCT = 0.005      # 0.5% ATR mínimo para autorizar Flip
    FLIP_MIN_POTENTIAL_RR = 2.0    # R:R esperado para la nueva dirección
    FLIP_MAX_DAILY_COUNT = 1       # Máximas reversiones por símbolo al día (Recomendación final)
    FLIP_COST_THRESHOLD = 0.002    # 0.2% max cost (Fees + Slippage)
    FLIP_COOLDOWN_SECONDS = 300    # 5 min de espera tras un Flip exitoso
    
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
        ML_KELLY_FRACTION = 0.5       # Half-Kelly for safety
        ANALYTICS_EXPECTANCY_WINDOW = 20 # Rolling window for Kill Switch
        
        # Adaptive Technical
        TECH_DYNAMIC_RSI_VOL_THRESHOLD = 0.005 # 0.5% ATR for band expansion

    # Phase 99: WandB Tracking
    WANDB_ENTITY = "jhonala-none"
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
        REGIME_MAP = {
            'TRENDING_BULL': {'leverage': 8, 'threshold_mod': -0.05, 'scale': 1.0}, # SNIPER BEHAVIOR (Aggressive)
            'TRENDING_BEAR': {'leverage': 1, 'threshold_mod': +0.10, 'scale': 0.0}, # DEFENSE BEHAVIOR (Cash)
            'RANGING':       {'leverage': 3, 'threshold_mod': +0.00, 'scale': 0.8}, # SCALPING BEHAVIOR (Moderate)
            'CHOPPY':        {'leverage': 1, 'threshold_mod': +0.05, 'scale': 0.5}, # CAUTION BEHAVIOR (Low Risk)
            'ZOMBIE':        {'leverage': 1, 'threshold_mod': +1.00, 'scale': 0.0}, # DEAD MARKET (No Trade)
        }
        
        # MAINNET SWITCH - Set to True for REAL trading
        USE_MAINNET = True   # LIVE PRODUCTION MODE
        
        # WHITELIST - Only ultra-high liquidity pairs (low spread)
        WHITELIST = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        
        # DYNAMIC LEVERAGE (ATR-based)
        MIN_LEVERAGE = 3
        MAX_LEVERAGE = 8
        DEFAULT_LEVERAGE = 5
        
        # FEES (Binance Futures)
        TAKER_FEE = 0.0005  # 0.05%
        MAKER_FEE = 0.0002  # 0.02%
        
        # TRADE VALIDATION
        MIN_RR_RATIO = 2.0       # Minimum Risk/Reward ratio
        MIN_PROFIT_AFTER_FEES = 0.005  # 0.5% minimum net profit
        
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
        except AssertionError as e:
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
        if Config.MAX_RISK_PER_TRADE > 0.02:
            errors.append(f"❌ RISK: Max Risk {Config.MAX_RISK_PER_TRADE*100}% exceeds Institutional Limit (2%).")
            
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
