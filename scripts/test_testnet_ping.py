import os
import sys
import asyncio
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config

load_dotenv()

async def ping_testnet():
    print("🚀 Iniciando Ping a Binance Futures Testnet...")
    
    api_key = os.getenv("BINANCE_DEMO_API_KEY")
    secret = os.getenv("BINANCE_DEMO_SECRET_KEY")
    
    if not api_key:
        print("❌ Faltan llaves de Testnet en .env")
        return
        
    exchange = ccxt_async.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    exchange.set_sandbox_mode(True)
    
    try:
        print("📥 Consultando Balance...")
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {})
        total = usdt_balance.get('total', 0.0)
        free = usdt_balance.get('free', 0.0)
        
        print(f"✅ Conexión Exitosa!")
        print(f"💰 Balance USDT: {total} (Libre: {free})")
        
        if total < 13.0:
            print("⚠️ WARNING: El balance en Testnet es menor al objetivo base de 13 USD.")
        else:
            print("✅ El capital en Testnet es suficiente para emular el reto de 13 USD.")
            
    except Exception as e:
        print(f"❌ Falló conexión: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(ping_testnet())
