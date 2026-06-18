import asyncio
import time
from core.events import SignalEvent, SignalType
from core.engine import GodModeEngine

async def test_rejection_flow():
    print("🚀 [TEST 3] TEST DE RECHAZO (SEÑALES INVÁLIDAS)")
    print("------------------------------------------------------")
    
    engine = GodModeEngine()
    
    # Inyectar señal de un activo que no existe y con strength negativo
    invalid_signal = SignalEvent(
        symbol="INVALID/USDT",
        datetime=time.time(),
        signal_type=SignalType.LONG,
        strength=-100.0,
        strategy_id="TEST_REJECTION",
        horizon="SCALPING",
        metadata={"mock_trace": True}
    )
    
    print(f"[{time.time_ns()}] 🟢 Emitiendo Señal Inválida: {invalid_signal.symbol} con strength {invalid_signal.strength}")
    engine.events.put_nowait(invalid_signal)
    
    print(f"[{time.time_ns()}] ⚙️ Procesando...")
    try:
        async def run_once():
            event = await engine.events.get()
            await engine.process_event(event)
        await asyncio.wait_for(run_once(), timeout=2.0)
    except Exception as e:
        print(f"❌ Error durante el paso de la señal: {e}")
        
    print(f"[{time.time_ns()}] 🏁 Test finalizado. Revisa los logs para ver qué módulo atrapó la excepción (Engine, Arbitrator, o RiskManager).")

if __name__ == "__main__":
    asyncio.run(test_rejection_flow())
