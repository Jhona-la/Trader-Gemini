import asyncio
import pytest
from core.engine import Engine
from core.events import MarketEvent, SignalEvent
from core.enums import SignalType
from datetime import datetime, timezone
import time
import multiprocessing

@pytest.mark.asyncio
async def test_async_engine_flow():
    """Verifica que el Engine procesa eventos de forma asíncrona sin bloquear"""
    engine = Engine()
    
    # Mock de procesamiento de señal
    signal_processed = asyncio.Event()
    
    async def mock_signal_processor(event):
        await asyncio.sleep(0.1) # Simular latencia
        signal_processed.set()
        
    engine._process_signal_event = mock_signal_processor
    
    # Iniciar motor en segundo plano
    engine_task = asyncio.create_task(engine.start())
    
    # Enviar evento
    test_event = SignalEvent(
        strategy_id="TEST",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG
    )
    
    engine.events.put(test_event)
    
    # Esperar confirmación con timeout
    try:
        await asyncio.wait_for(signal_processed.wait(), timeout=2.0)
        assert engine.metrics['processed_events'] >= 1
        print("✅ Engine processed async signal successfully.")
    finally:
        engine.stop()
        await engine_task

def test_ml_process_isolation():
    """Verifica que el proceso de inferencia de ML es independiente"""
    # Esta es una prueba de arquitectura básica
    from strategies.ml_strategy import ml_inference_worker_task
    
    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()
    
    # Mock de modelos
    class MockModel:
        def predict_proba(self, X):
            return [[0.1, 0.9]]
            
    p = multiprocessing.Process(target=ml_inference_worker_task, args=(in_q, out_q))
    p.start()
    
    try:
        in_q.put({
            'X': [[0]], 
            'rf': MockModel(), 'xgb': MockModel(), 'gb': MockModel(),
            'ts': time.time()
        })
        
        result = out_q.get(timeout=5)
        assert 'confidence' in result
        assert result['confidence'] > 0.8
        print("✅ ML Worker Process responded correctly.")
    finally:
        p.terminate()

if __name__ == "__main__":
    asyncio.run(test_async_engine_flow())
    test_ml_process_isolation()
