
from core.shadow_optimizer import ShadowOptimizer
from data.binance_loader import BinanceData
from config import Config
import asyncio

async def trigger_shadow_audit():
    print("🤖 [RUNNER] Iniciando Shadow Optimizer (Auditoría de Domingo)")
    
    # 1. Setup Data Provider (Read-only)
    import queue
    loader = BinanceData(events_queue=queue.Queue(), symbol_list=Config.TRADING_PAIRS)
    
    # 2. Init Optimizer
    optimizer = ShadowOptimizer(loader)
    
    # 3. Audit primary symbols
    for symbol in Config.TRADING_PAIRS[:2]: # BTC and ETH
        report = optimizer.run_weekly_audit(symbol)
        
        if report.get('status') == 'error':
            print(f"❌ Error en auditoría de {symbol}: {report.get('message', report.get('reason'))}")
            continue
            
        params = report['recommended_params']['params']
        print(f"✅ Reporte generado para {symbol}:")
        print(f"   - Recomendación Z: {params.get('STAT_Z_ENTRY')}")
        print(f"   - Recomendación RSI: {params.get('RSI_LOWER_BOUND')}")
        print(f"   - PnL Simulado: {params.get('SIM_PNL')}")
        print(f"   - WinRate: {params.get('SIM_WINRATE')}")
        print(f"   - Razón: {report['recommended_params']['REASON']}")

if __name__ == "__main__":
    asyncio.run(trigger_shadow_audit())
