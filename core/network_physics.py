import asyncio
import time
import logging
from core.schemas import TickEvent

class PhysicalDataHandlerWrapper:
    """
    Simulador Termodinámico de Red (Fase III Metamorfosis).
    
    Envuelve el Data Handler. Inyecta retraso termodinámico ANTES de procesar
    los ticks, obligando al backtest a ver el mercado con la misma latencia
    que tendría en producción. No penaliza la orden (eso sería predecir el futuro),
    penaliza la recepción de la información.
    """
    def __init__(self, raw_data_stream_generator, base_latency_ms=45):
        self.raw_stream = raw_data_stream_generator
        self.base_latency_ms = base_latency_ms
        self.queue = asyncio.Queue()
        self.logger = logging.getLogger("NetworkPhysics")

    async def _latency_injector_worker(self):
        """Consume el stream en crudo y encola con timestamp de liberación futuro."""
        try:
            async for raw_tick in self.raw_stream:
                # El tick se "envía" ahora, pero tardará 'base_latency_ms' en llegar
                release_time = time.time() + (self.base_latency_ms / 1000.0)
                await self.queue.put((release_time, raw_tick))
        except asyncio.CancelledError:
            pass

    async def get_next_tick(self) -> TickEvent:
        """
        El motor llama a este método para obtener el siguiente tick.
        Bloquea (Slippage por tiempo) si el tick aún está 'viajando' por la red simulada.
        """
        while True:
            release_time, raw_tick_dict = await self.queue.get()
            now = time.time()
            if now < release_time:
                # El tick aún viaja por los cables submarinos
                await asyncio.sleep(release_time - now)
            
            # El tick cruza la Frontera Rígida.
            # Fuerza un fallo inmediato si los datos de la vela simulada no encajan.
            return TickEvent(**raw_tick_dict)

