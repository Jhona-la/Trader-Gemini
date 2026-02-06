
import asyncio
import os
from execution.liquidity_guardian import LiquidityGuardian
from config import Config
from utils.logger import logger
import ccxt

async def live_liquidity_check():
    print("🔍 [LIVE VERIFICATION] Liquidity Guardian - Read-only Check")
    
    # 1. Setup Exchange (Read-only)
    api_key = Config.BINANCE_API_KEY
    api_secret = Config.BINANCE_SECRET_KEY
    
    # Use Demo symbols if configured, or just standard BTC/USDT
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    if Config.BINANCE_USE_TESTNET or (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO):
        exchange.set_sandbox_mode(True)
        print("💡 Mode: SANDBOX/DEMO")
    else:
        print("🛡️ Mode: LIVE PRODUCTION")

    guardian = LiquidityGuardian(exchange)
    
    print("-" * 50)
    for symbol in test_symbols:
        try:
            print(f"📊 Analyzing {symbol}...")
            # Simulate a small BUY order (0.01 qty)
            analysis = guardian.analyze_liquidity(symbol, 0.01, "BUY")
            
            print(f"   Status: {'✅ SAFE' if analysis['is_safe'] else '❌ BLOCKED'}")
            print(f"   Reason: {analysis['reason']}")
            print(f"   Avg Fill Price: ${analysis['avg_fill_price']:.2f}")
            print(f"   Slippage: {analysis['slippage_pct']*100:.4f}%")
            
            walls = analysis['walls']
            if walls['ask_walls']:
                print(f"   🚩 SELL WALLS DETECTED: {walls['ask_walls'][0]}")
            else:
                print(f"   ✨ No immediate institutional walls detected.")
                
            print("-" * 50)
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {e}")

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(live_liquidity_check())
