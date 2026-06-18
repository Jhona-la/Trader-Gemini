import asyncio
import time
from core.events import SignalEvent, SignalType
from core.engine import GodModeEngine

async def test_signal_injection():
    print("🚀 [TEST 1] INYECCIÓN DIRECTA DE SEÑAL DE DIAGNÓSTICO")
    print("------------------------------------------------------")
    
    # 1. Crear Motor Aislado
    engine = GodModeEngine()
    
    # 2. Fabricar Señal Pura
    fake_signal = SignalEvent(
        symbol="BTC/USDT",
        datetime=time.time(),
        signal_type=SignalType.LONG,
        strength=0.99,
        strategy_id="TEST_MOCK_01",
        horizon="SCALPING",
        metadata={"mock_trace": True, "source": "Plan_Delta_Injector"}
    )
    
    print(f"[{time.time_ns()}] 🟢 Emitiendo Señal: {fake_signal.symbol} LONG (Strength: {fake_signal.strength})")
    
    # 3. Inyectar directamente en el buzón del motor
    engine.events.put_nowait(fake_signal)
    
    # 4. Procesar la cola (un solo tick)
    print(f"[{time.time_ns()}] ⚙️ Motor procesando cola...")
    try:
        # Simulamos ejecución temporal
        async def run_once():
            event = await engine.events.get()
            await engine.process_event(event)
        await asyncio.wait_for(run_once(), timeout=2.0)
    except asyncio.TimeoutError:
        print("⚠️ Timeout: El motor quedó esperando (quizás la señal se atoró en el MetaArbitrator o RiskManager)")
    except Exception as e:
        print(f"❌ Error durante el paso de la señal: {e}")
        
    print(f"[{time.time_ns()}] 🏁 Trazado finalizado. Revisa los logs arriba para ver si el Arbitrator la vetó o el RiskManager la transformó en Orden.")

if __name__ == "__main__":
    asyncio.run(test_signal_injection())
