import asyncio
import logging
from config import Config
Config.IS_BACKTEST = True
from core.engine import Engine
from core.portfolio import Portfolio
from core.events import BoundedPriorityQueue
from risk.risk_manager import RiskManager
from data.historical_loader import get_binance_data_async

async def run_wealth_test():
    print("🚀 [WEALTH PACT] Inicializando Auditoría de Acumulación Exponencial (Phase 1)...")
    
    events = BoundedPriorityQueue()
    engine = Engine(events)
    portfolio = Portfolio(events, initial_capital=13.0) # $13 USD Start
    engine.register_portfolio(portfolio)
    
    risk_manager = RiskManager(portfolio=portfolio)
    engine.register_risk_manager(risk_manager)
    
    print(f"✅ Capital Inicial: ${portfolio.get_total_equity():.2f}")
    
    # 1. Simulate a mock fill to check sizing limits
    print("\n🔍 Simulando Sizing en Fase 1 (<$50)...")
    sizing = risk_manager.size_position("BTCUSDT", risk_pct=0.03, multiplier=1.0, horizon="SCALPING", current_price=100000.0)
    
    if sizing:
        print(f"✅ Sizing Aprobado: Notional = ${sizing['notional']:.2f} | Lev = {sizing['leverage']}x | Qty = {sizing['quantity']:.8f}")
    else:
        print("❌ Sizing Rechazado (Fallo en Minimum Notional o Cap)")
        
    print("\n🔍 Simulando Sizing en Swing para Fase 1 (Debería ser bloqueado)...")
    sizing_swing = risk_manager.size_position("BTCUSDT", risk_pct=0.03, multiplier=1.0, horizon="SWING", current_price=100000.0)
    if sizing_swing:
        print("❌ FALLO: Swing no fue bloqueado en Fase 1.")
    else:
        print("✅ ÉXITO: Swing fue correctamente bloqueado para evitar Funding Drag.")
        
    print("\n📊 Análisis de Parámetros Activos:")
    print(f"SL: {Config.Horizons.Scalping['sl_pct']*100:.2f}%")
    print(f"TP: {Config.Horizons.Scalping['tp_pct']*100:.2f}%")
    
    # 2. Check telegram Notifier (Simulate call)
    from utils.notifier import Notifier
    print("\n📩 Probando Notifier (Telegram/Logging)...")
    Notifier.send_risk_alert({
        'type': 'WEALTH_PACT_START',
        'level': 'INFO',
        'message': 'Auditoría de Acumulación iniciada correctamente. Modos Fase 1 activos.',
        'balance': 13.0
    })

if __name__ == "__main__":
    asyncio.run(run_wealth_test())
