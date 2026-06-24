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

async def ping_demo():
    print("🚀 Iniciando Ping a Binance Futures DEMO...")
    
    api_key = os.getenv("BINANCE_DEMO_API_KEY")
    secret = os.getenv("BINANCE_DEMO_SECRET_KEY")
    
    if not api_key:
        print("❌ Faltan llaves de Demo en .env")
        return
        
    exchange = ccxt_async.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    # Esta es la funcion correcta para habilitar Futuros de Prueba (testnet.binancefuture.com)
    exchange.set_sandbox_mode(True)
    # Sin embargo, en CCXT reciente para Binance Futures Demo Trading puede requerir algo diferente, pero probaremos.
    if hasattr(exchange, 'enable_demo_trading'):
        exchange.enable_demo_trading(True)
        print("✅ Demo Trading Endpoint Set (fapi testnet)")
    
    try:
        print("📥 Consultando Balance...")
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {})
        total = usdt_balance.get('total', 0.0)
        free = usdt_balance.get('free', 0.0)
        
        print(f"✅ Conexión Exitosa a FUTUROS TESTNET!")
        print(f"💰 Balance USDT Demo: {total} (Libre: {free})")
        
    except Exception as e:
        print(f"❌ Falló conexión: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(ping_demo())
