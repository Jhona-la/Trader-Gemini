import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance Keys
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Binance Testnet
    BINANCE_USE_TESTNET = os.getenv("BINANCE_USE_TESTNET", "True").lower() == "true"
    BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "QccFqERnEE2UkiveU16Sf5idz5QMzIJnEAh0mx1MnnrleYvEyojeIXr5a2stoHXb")
    BINANCE_TESTNET_SECRET_KEY = os.getenv("BINANCE_TESTNET_SECRET_KEY", "Us0NzceQjiGOv1h10bnsZRBGhvNVnKDNvpDCotlChPePouounL8eULzh2PGM6E4m")
    
    # === BINANCE DEMO TRADING (NEW) ===
    # Demo Trading es el reemplazo de Testnet para Futures
    BINANCE_USE_DEMO = True  # Habilitar Demo Trading (capital virtual)
    BINANCE_DEMO_API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "")
    BINANCE_DEMO_SECRET_KEY = os.getenv("BINANCE_DEMO_SECRET_KEY", "")
    
    # === BINANCE FUTURES SETTINGS ===
    BINANCE_USE_FUTURES = True  # Set to True to trade on Binance Futures instead of Spot
    BINANCE_LEVERAGE = 20  # Leverage for Futures trading (AGGRESSIVE: 20x)
    BINANCE_MARGIN_TYPE = "ISOLATED"  # Options: "ISOLATED" or "CROSS"
    
    # Dynamic Data Directory (Separate Spot & Futures)
    if BINANCE_USE_FUTURES:
        DATA_DIR = "dashboard/data/futures"
    else:
        DATA_DIR = "dashboard/data"
        
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
    TRADING_PAIRS = CRYPTO_FUTURES_PAIRS if BINANCE_USE_FUTURES else CRYPTO_SPOT_PAIRS
    
    # STOCKS (Disabled - crypto only)
    STOCKS = []
    
    # FOREX (Disabled - crypto only)  
    FOREX = []

    # Risk Management
    MAX_RISK_PER_TRADE = 0.01  # 1% of capital
    STOP_LOSS_PCT = 0.02       # 2% stop loss
