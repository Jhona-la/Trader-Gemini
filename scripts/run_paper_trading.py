import os
import sys
import asyncio

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.logger import logger

def configure_paper_trading():
    """
    Force configuration into Paper Trading mode.
    This guarantees 0% risk to real capital while executing production logic.
    """
    logger.info("🔧 Configurando entorno para PAPER TRADING (0 Riesgo)")
    
    # Enable Demo/Mock Executor Flag
    Config.BINANCE_USE_DEMO = True
    Config.BINANCE_USE_TESTNET = True 
    
    # Override any live execution protections just to be safe
    # But keep the core logic identical to production.
    
    # Disable telegram notifications to avoid spamming during paper trading
    Config.TELEGRAM_ALERTS = False

async def main():
    logger.info("=========================================================")
    logger.info("🚀 INICIANDO TRADER GEMINI - MODO PAPER TRADING")
    logger.info("=========================================================")
    
    configure_paper_trading()
    
    # Import main AFTER config override so main.py reads the new config
    import main as production_main
    
    try:
        await production_main.main()
    except KeyboardInterrupt:
        logger.info("🛑 PAPER TRADING detenido por el usuario.")
    except Exception as e:
        logger.critical(f"❌ Error crítico en Paper Trading: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
