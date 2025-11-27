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
    
    # === SPOT MODE CONFIGURATION ===
    BINANCE_USE_DEMO = False  # Demo only exists for Futures
    BINANCE_DEMO_API_KEY = ""
    BINANCE_DEMO_SECRET_KEY = ""
    
    # === SPOT SETTINGS ===
    BINANCE_USE_FUTURES = False  # SPOT MODE
    BINANCE_LEVERAGE = 1  # No leverage in Spot
    BINANCE_MARGIN_TYPE = "NONE"
    
    # Data Directory for Spot
    DATA_DIR = "dashboard/data/spot"
        
    # Ensure directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    # Interactive Brokers Settings (Disabled)
    IBKR_HOST = "127.0.0.1"
    IBKR_PORT = 7497
    IBKR_CLIENT_ID = 1

    # === SPOT TRADING PAIRS (23 total) ===
    TRADING_PAIRS = [
        # Top 10 Major Coins
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "DOT/USDT", "SHIB/USDT", "PEPE/USDT",
        
        # High Volume Altcoins
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
        "LTC/USDT", "ETC/USDT", "FLOKI/USDT", "BONK/USDT", "WIF/USDT",
        "AAVE/USDT", "COMP/USDT", "SAND/USDT", "MANA/USDT"
    ]
    
    # STOCKS (Disabled)
    STOCKS = []
    
    # FOREX (Disabled)  
    FOREX = []

    # Risk Management
    MAX_RISK_PER_TRADE = 0.01  # 1% of capital
    STOP_LOSS_PCT = 0.02       # 2% stop loss
    
    # Initial Capital for Spot
    INITIAL_CAPITAL = 10000.0
