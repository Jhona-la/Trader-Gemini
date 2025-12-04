import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # ========================================================================
    # BINANCE API CREDENTIALS (Loaded from .env file)
    # ========================================================================
    
    # Production Keys (Leave empty in .env if using Demo/Testnet)
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Binance Testnet (Spot)
    BINANCE_USE_TESTNET = True  # Hardcoded to True for safety
    BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
    BINANCE_TESTNET_SECRET_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY')
    
    # Binance Demo Trading (Futures with virtual capital)
    BINANCE_USE_DEMO = True  # Enable Demo Trading
    BINANCE_DEMO_API_KEY = os.getenv('BINANCE_DEMO_API_KEY')
    BINANCE_DEMO_SECRET_KEY = os.getenv('BINANCE_DEMO_SECRET_KEY')

    
    # === BINANCE FUTURES SETTINGS ===
    # Default: USDT-Margined Futures (standard). 
    # For COIN-Margined, code modifications in binance_executor would be needed (defaultType='delivery').
    # BUG #33 FIX: Changed default to False to allow Spot mode. CLI --mode argument will override this.
    BINANCE_USE_FUTURES = False  # Set to True to trade on Binance Futures instead of Spot
    BINANCE_LEVERAGE = 20  # Leverage for Futures trading (AGGRESSIVE: 20x)
    BINANCE_MARGIN_TYPE = "ISOLATED"  # Options: "ISOLATED" or "CROSS"
    
    # Dynamic Data Directory (Separate Spot & Futures)
    if BINANCE_USE_FUTURES:
        DATA_DIR = "dashboard/data/futures"
    else:
        DATA_DIR = "dashboard/data/spot"  # Explicitly separate Spot data
        
    # Ensure directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Interactive Brokers Settings (Disabled - focusing on crypto only)
    IBKR_HOST = "127.0.0.1"
    IBKR_PORT = 7497
    IBKR_CLIENT_ID = 1

    # === TRADING PAIRS CONFIGURATION ===
    
    # SPOT Trading Pairs (All available in Binance Spot)
    CRYPTO_SPOT_PAIRS = [
        # Top 10 Major Coins
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "DOT/USDT", "SHIB/USDT", "PEPE/USDT",
        
        # High Volume Altcoins
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
        "LTC/USDT", "ETC/USDT", "FLOKI/USDT", "BONK/USDT", "WIF/USDT",
        "AAVE/USDT", "COMP/USDT", "SAND/USDT", "MANA/USDT"
    ]
    # Total: 23 pairs
    
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
    CRYPTO_FUTURES_PAIRS = [
        # Top Tier (Major coins - 100% available)
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "DOT/USDT",
        
        # High Volume Altcoins (verified in Demo Trading)
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
        "LTC/USDT", "ETC/USDT", "AAVE/USDT", "COMP/USDT", 
        "SAND/USDT", "MANA/USDT", "WIF/USDT"
    ]
    # Total: 19 pairs (all Demo Trading compatible)
    
    # Auto-select correct pairs based on mode
    # BUG #14 FIX: Binance Testnet SPOT is UNRELIABLE (most pairs don't exist)
    # Solution: SPOT only works in PRODUCTION, Testnet/Demo users should use FUTURES
    if BINANCE_USE_FUTURES:
        TRADING_PAIRS = CRYPTO_FUTURES_PAIRS  # 19 pairs for Futures
    else:
        # SPOT mode - but WARN if trying to use in Testnet
        TRADING_PAIRS = BINANCE_TESTNET_SPOT_PAIRS  # Start with limited pairs
        # If user has real API keys (not demo/testnet), allow full pairs
        if BINANCE_API_KEY and not ("test" in BINANCE_API_KEY.lower() or "demo" in BINANCE_API_KEY.lower()):
            TRADING_PAIRS = CRYPTO_SPOT_PAIRS  # 23 pairs for Production SPOT
    
    # STOCKS (Disabled - crypto only)
    STOCKS = []
    
    # FOREX (Disabled - crypto only)  
    FOREX = []

    # Risk Management
    MAX_RISK_PER_TRADE = 0.01  # 1% of capital
    STOP_LOSS_PCT = 0.02       # 2% stop loss
    
    # BUG #47 FIX: Position Sizing Configuration (moved from risk_manager.py)
    # Dynamic position sizing based on account size
    POSITION_SIZE_SMALL_THRESHOLD = 1000   # Capital < $1000 = Small account
    POSITION_SIZE_LARGE_THRESHOLD = 10000  # Capital > $10000 = Large account
    
    # Position size as % of capital per trade
    POSITION_SIZE_SMALL_ACCOUNT = 0.20   # 20% - Aggressive growth for small accounts
    POSITION_SIZE_MEDIUM_ACCOUNT = 0.15  # 15% - Balanced for medium accounts
    POSITION_SIZE_LARGE_ACCOUNT = 0.10   # 10% - Conservative wealth preservation


# ============================================================================
# CONFIGURATION VALIDATION (Fail Fast on Missing Credentials)
# ============================================================================
def validate_config():
    """
    Validates that required API credentials are present.
    Fails fast with clear error messages if configuration is incomplete.
    This prevents the bot from starting with missing/invalid credentials.
    """
    errors = []
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        errors.append("❌ ERROR: .env file not found!")
        errors.append("   → Solution: Copy .env.example to .env and fill in your API keys")
        errors.append("   → Command: copy .env.example .env  (Windows)")
        errors.append("   →          cp .env.example .env    (Linux/Mac)")
    
    # Determine which keys are required based on mode
    if Config.BINANCE_USE_DEMO or Config.BINANCE_USE_TESTNET:
        # Demo/Testnet mode - check for demo keys
        if not Config.BINANCE_USE_FUTURES:
            # Spot Testnet mode
            if not Config.BINANCE_TESTNET_API_KEY:
                errors.append("❌ ERROR: BINANCE_TESTNET_API_KEY not found in .env")
                errors.append("   → Required for Spot Testnet mode")
            if not Config.BINANCE_TESTNET_SECRET_KEY:
                errors.append("❌ ERROR: BINANCE_TESTNET_SECRET_KEY not found in .env")
        else:
            # Futures Demo mode
            if not Config.BINANCE_DEMO_API_KEY:
                errors.append("❌ ERROR: BINANCE_DEMO_API_KEY not found in .env")
                errors.append("   → Required for Futures Demo mode")
            if not Config.BINANCE_DEMO_SECRET_KEY:
                errors.append("❌ ERROR: BINANCE_DEMO_SECRET_KEY not found in .env")
    else:
        # Production mode - check for real keys
        if not Config.BINANCE_API_KEY:
            errors.append("❌ ERROR: BINANCE_API_KEY not found in .env")
            errors.append("   → Required for PRODUCTION mode")
        if not Config.BINANCE_SECRET_KEY:
            errors.append("❌ ERROR: BINANCE_SECRET_KEY not found in .env")
    
    # Configuration warnings (non-fatal but important)
    warnings = []
    
    if Config.BINANCE_LEVERAGE > 25:
        warnings.append(f"⚠️  WARNING: Leverage set to {Config.BINANCE_LEVERAGE}x (exceeds recommended 25x)")
        warnings.append("   → High leverage = High risk of liquidation")
    
    if Config.MAX_RISK_PER_TRADE > 0.05:
        warnings.append(f"⚠️  WARNING: Risk per trade is {Config.MAX_RISK_PER_TRADE*100}% (exceeds recommended 5%)")
        warnings.append("   → High risk percentage can lead to rapid capital depletion")
    
    if Config.BINANCE_USE_FUTURES and len(Config.CRYPTO_FUTURES_PAIRS) == 0:
        errors.append("❌ ERROR: FUTURES mode enabled but no futures pairs configured")
    
    # Print all errors and warnings
    if warnings:
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION WARNINGS")
        print("="*70)
        for warning in warnings:
            print(warning)
        print("="*70 + "\n")
    
    if errors:
        print("\n" + "="*70)
        print("🚨 CONFIGURATION ERRORS - BOT CANNOT START")
        print("="*70)
        for error in errors:
            print(error)
        print("="*70)
        print("\n💡 QUICK FIX:")
        print("   1. Ensure .env file exists in project root")
        print("   2. Copy from template: copy .env.example .env")
        print("   3. Edit .env and add your API keys")
        print("   4. Restart the bot\n")
        sys.exit(1)  # Exit with error code
    
    return True

# Run validation on import (fails fast if config invalid)
validate_config()
