"""
═══════════════════════════════════════════════════════════════
SHARED POOLS: Centralized Thread/Process Pool Management

QUÉ: Pool de ejecutores compartido para todo el sistema Trader Gemini.
POR QUÉ: Cada MLStrategyHybridUltimate creaba su propio ThreadPoolExecutor(2).
     Con 42 instancias = 84 threads + ProcessPool(6) = 90+ threads.
     Cada thread stack consume ~1MB de RAM.
PARA QUÉ: Reducir de ~90 threads a ~10, eliminando context-switching
     y liberando ~80MB de RAM en thread stacks.
CÓMO: Singleton pattern con pools compartidos para inferencia y entrenamiento.
CUÁNDO: Se inicializa al primer uso (lazy singleton).
DÓNDE: core/shared_pools.py
QUIÉN: MLStrategyHybridUltimate, HybridScalpingStrategy

OPTIMIZACIÓN RAM: ~80MB liberados (80 threads eliminados × 1MB stack cada uno)
═══════════════════════════════════════════════════════════════
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils.logger import logger


class SharedPools:
    """
    Singleton pool manager for the entire Trader Gemini system.
    
    Provides:
    - inference_pool: For fast ML inference tasks (4 threads)
    - training_pool: For heavy ML training tasks (2 workers)
    - io_pool: For I/O operations like model saving/loading (2 threads)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'SharedPools':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # CPU count detection for optimal sizing
        cpu_count = os.cpu_count() or 4
        
        # Inference: Fast, lightweight tasks (predict)
        # 4 threads is enough: even with 42 symbols, inference is <1ms per call
        self._inference_pool = ThreadPoolExecutor(
            max_workers=min(4, cpu_count),
            thread_name_prefix="MLInference"
        )
        
        # Training: Heavy, CPU-bound tasks (fit models)
        # 2 workers max to avoid thermal throttling on Ryzen 5700U
        max_train_workers = max(1, min(2, cpu_count // 4))
        self._training_pool = ThreadPoolExecutor(
            max_workers=max_train_workers,
            thread_name_prefix="MLTraining"
        )
        
        # I/O: Model save/load, database operations
        self._io_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="SharedIO"
        )
        
        logger.info(
            f"🔧 [SharedPools] Initialized: "
            f"Inference={min(4, cpu_count)} threads, "
            f"Training={max_train_workers} threads, "
            f"IO=2 threads (Total: {min(4, cpu_count) + max_train_workers + 2})"
        )
    
    @property
    def inference_pool(self) -> ThreadPoolExecutor:
        """Pool for fast ML inference (predict calls)."""
        return self._inference_pool
    
    @property
    def training_pool(self) -> ThreadPoolExecutor:
        """Pool for heavy ML training (fit calls)."""
        return self._training_pool
    
    @property
    def io_pool(self) -> ThreadPoolExecutor:
        """Pool for I/O operations (model save/load)."""
        return self._io_pool
    
    def submit_inference(self, fn, *args, **kwargs):
        """Submit a fast inference task."""
        return self._inference_pool.submit(fn, *args, **kwargs)
    
    def submit_training(self, fn, *args, **kwargs):
        """Submit a heavy training task."""
        return self._training_pool.submit(fn, *args, **kwargs)
    
    def submit_io(self, fn, *args, **kwargs):
        """Submit an I/O task."""
        return self._io_pool.submit(fn, *args, **kwargs)
    
    def shutdown(self, wait=True):
        """Graceful shutdown of all pools."""
        logger.info("🔧 [SharedPools] Shutting down all pools...")
        self._inference_pool.shutdown(wait=wait)
        self._training_pool.shutdown(wait=wait)
        self._io_pool.shutdown(wait=wait)


# Module-level convenience function
def get_shared_pools() -> SharedPools:
    """Get the singleton SharedPools instance."""
    return SharedPools.get_instance()
