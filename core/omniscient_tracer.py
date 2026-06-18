"""
OMNISCIENT TRACER (Capa 0: The Weaver / La Telaraña)
====================================================
QUÉ: Registrador de topología en tiempo real y ultrabaja latencia (<1ms).
POR QUÉ: Necesitamos probar que el sistema es integral, adaptativo y coherente
         (opera el top 10 y monitoriza a los 16 restantes) sin fallos fantasma.
PARA QUÉ: Generar un grafo ("red neuronal") que muestre Origen -> Destino,
          Coherencia y Latencia en cada decisión.
CÓMO: Inyección mediante Decoradores (@omniscient_trace) que escriben en RAM
      y delegan la escritura a disco a un hilo de fondo (QueueListener).
DÓNDE: core/omniscient_tracer.py
QUIÉN: SRE/DevOps & Arquitecto Senior
"""

import time
import os
import json
import logging
from functools import wraps
from typing import Optional, Dict, Any
from queue import Queue
from logging.handlers import QueueHandler, QueueListener, MemoryHandler

# Initialize Fast JSON
try:
    import orjson
    def fast_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    def fast_dumps(obj):
        return json.dumps(obj)

class OmniscientGraphLogger:
    """Singleton Async Logger para la Telaraña Neuronal."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OmniscientGraphLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        log_dir = "dashboard/data"
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "omniscient_graph.jsonl")

        # Configuración del Logger Nativo de Python (Async + Memory Buffer)
        self.logger = logging.getLogger("OmniscientTracer")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # FORENSIC FIX: Evitar inundar stdout
        
        # Evitar duplicados
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # File Handler base
        file_handler = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        # Formatter simple, ya que pasaremos el JSON crudo en el message
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Memory Buffer Handler (Acumula 2000 eventos en RAM antes de tocar SSD)
        memory_handler = MemoryHandler(capacity=2000, flushLevel=logging.CRITICAL, target=file_handler)
        
        # Async Queue Handler (Hilo separado para no bloquear la ejecución principal)
        self.queue = Queue(-1)
        queue_handler = QueueHandler(self.queue)
        self.logger.addHandler(queue_handler)
        
        self.listener = QueueListener(self.queue, memory_handler, respect_handler_level=True)
        self.listener.start()
        
    def log_edge(self, source_layer: str, target_func: str, latency_us: float, metadata: Dict[str, Any]):
        """Registra una arista (conexión) en el grafo del sistema."""
        edge_data = {
            "ts": time.time(),
            "source": source_layer,
            "target": target_func,
            "latency_us": latency_us,
            "metadata": metadata
        }
        # Inyecta en la cola asíncrona (Costo de CPU: ~0.005 ms)
        self.logger.info(fast_dumps(edge_data))
        
    def stop(self):
        self.listener.stop()

# Global Tracer Instance
tracer = OmniscientGraphLogger()

def omniscient_trace(layer: str, emit_args: bool = False):
    """
    Decorador Cuántico: 
    Mide la cohesión del sistema midiendo la latencia de entrada-salida
    y rastreando la topología de la función ejecutada dentro de la "Telaraña".
    
    layer: El "Lóbulo Cerebral" de la red (e.g. CORTEX, AMYGDALA, MEMORY)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                # Ejecución de la función subyacente
                result = func(*args, **kwargs)
                t_end = time.perf_counter()
                
                # Extracción rápida de meta-estado (evitar sobrecarga)
                latency_us = (t_end - t_start) * 1_000_000
                
                # Auto-discover symbol if present in args/kwargs (común en Trader Gemini)
                symbol = kwargs.get('symbol', None)
                if not symbol and args and isinstance(args[0], str) and "/" in args[0]:
                    symbol = args[0]
                elif not symbol and len(args) > 1 and isinstance(args[1], str) and "/" in args[1]:
                    symbol = args[1]
                
                meta = {"status": "SUCCESS"}
                if symbol:
                    meta["symbol"] = symbol
                
                # Reportar el Edge al Grafo
                tracer.log_edge(source_layer=layer, target_func=func.__name__, latency_us=latency_us, metadata=meta)
                
                return result
            except Exception as e:
                t_end = time.perf_counter()
                latency_us = (t_end - t_start) * 1_000_000
                
                meta = {
                    "status": "ERROR",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                }
                tracer.log_edge(source_layer=layer, target_func=func.__name__, latency_us=latency_us, metadata=meta)
                raise e
        return wrapper
    return decorator
