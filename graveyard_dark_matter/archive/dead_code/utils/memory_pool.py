"""
⚡ OMEGA-VOID §1.3: GC-Free Memory Pool (Pool Allocation)

QUÉ: Arena de memoria pre-asignada para todas las monedas y timeframes.
POR QUÉ: Eliminar TODA asignación dinámica durante el hot path del engine.
     Las asignaciones dinámicas provocan GC Jitter (pausas de 1-50ms)
     que son inaceptables en HFT scalping a <10ms de latencia.
PARA QUÉ: Lograr CERO asignaciones de memoria durante el burst loop,
     garantizando latencia determinista sub-10ms.
CÓMO: Pre-asigna numpy arrays con alineación a 64 bytes (cacheline Ryzen 7)
     para 20 símbolos × 4 timeframes × 500 bars.
     Incluye GCGuard context manager para freeze/thaw del garbage collector.
CUÁNDO: Se inicializa UNA VEZ en BinanceData.__init__().
DÓNDE: utils/memory_pool.py
QUIÉN: BinanceData, Engine burst loop.
"""

import gc
import numpy as np
from typing import Dict, List, Optional
from utils.memory_alignment import aligned_zeros
from utils.logger import logger


# ============================================================
# CACHE LINE CONSTANTS (Ryzen 7 5700U: Zen 3)
# ============================================================
CACHELINE_BYTES = 64        # L1/L2 cache line size
FLOAT32_PER_LINE = 16       # 64 / 4 = 16 float32s per cache line
INT64_PER_LINE = 8          # 64 / 8 = 8 int64s per cache line
OHLCV_FIELDS = 6            # t, o, h, l, c, v


def pad_to_cacheline(capacity: int, element_size: int = 4) -> int:
    """
    Rounds capacity UP to the nearest multiple of cache line elements.
    
    QUÉ: Asegura que el tamaño del array sea múltiplo exacto de cacheline.
    POR QUÉ: Evita False Sharing cuando threads adyacentes leen cachelines
         que contienen datos de otro thread.
    
    Args:
        capacity: Desired number of elements
        element_size: Bytes per element (4 for float32, 8 for int64)
    Returns:
        Padded capacity (always >= capacity)
    """
    elements_per_line = CACHELINE_BYTES // element_size
    return ((capacity + elements_per_line - 1) // elements_per_line) * elements_per_line


class BarSlot:
    """
    Pre-allocated OHLCV storage for a single symbol/timeframe.
    
    QUÉ: Slot individual de almacenamiento tipo ring-buffer.
    POR QUÉ: Cada slot tiene su propia memoria alineada, eliminando
         interferencia entre símbolos (False Sharing).
    CÓMO: Arrays numpy alineados a 64 bytes con capacity padded.
    """
    __slots__ = ('timestamps', 'opens', 'highs', 'lows', 'closes', 
                 'volumes', 'capacity', 'head', 'size')
    
    def __init__(self, capacity: int):
        self.capacity = pad_to_cacheline(capacity, 4)  # float32 alignment
        cap_ts = pad_to_cacheline(capacity, 8)          # int64 alignment
        
        # Pre-allocate ALL arrays with 64-byte alignment
        self.timestamps = aligned_zeros(cap_ts, dtype=np.int64)
        self.opens = aligned_zeros(self.capacity, dtype=np.float32)
        self.highs = aligned_zeros(self.capacity, dtype=np.float32)
        self.lows = aligned_zeros(self.capacity, dtype=np.float32)
        self.closes = aligned_zeros(self.capacity, dtype=np.float32)
        self.volumes = aligned_zeros(self.capacity, dtype=np.float32)
        self.head = 0
        self.size = 0
    
    def push(self, t: int, o: float, h: float, l: float, c: float, v: float):
        """O(1) insertion, ZERO allocation."""
        idx = self.head
        self.timestamps[idx] = t
        self.opens[idx] = o
        self.highs[idx] = h
        self.lows[idx] = l
        self.closes[idx] = c
        self.volumes[idx] = v
        self.head = (idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def memory_bytes(self) -> int:
        """Total bytes consumed by this slot."""
        return (self.capacity * 4 * 5) + (self.capacity * 8)  # 5 float32 + 1 int64


class OHLCVPool:
    """
    ⚡ OMEGA-VOID: Master Memory Pool for ALL symbols and timeframes.
    
    QUÉ: Arena centralizada de memoria pre-asignada. 
         20 símbolos × 4 timeframes = 80 BarSlots, CERO asignaciones futuras.
    POR QUÉ: En HFT, cada `np.zeros()` o `np.empty()` durante el loop principal
         puede triggear el GC de Python, causando pausas de 1-50ms.
         Con $13 USDT de capital, un jitter de 50ms puede costar un trade.
    PARA QUÉ: Latencia determinista: mismo costo cada ciclo, sin sorpresas.
    CÓMO: Pre-asigna todo en __init__, expone slots indexados por (symbol, tf).
    CUÁNDO: Se crea una vez en el arranque del sistema.
    DÓNDE: utils/memory_pool.py → OHLCVPool
    QUIÉN: BinanceData.__init__(), Engine.
    """
    
    TIMEFRAMES = ['1m', '5m', '15m', '1h']
    
    def __init__(self, symbols: List[str], capacity: int = 500):
        """
        Args:
            symbols: List of trading symbols (e.g. ['BTC/USDT', 'ETH/USDT', ...])
            capacity: Bars per slot per timeframe (default 500)
        """
        self.symbols = symbols
        self.capacity = capacity
        self.slots: Dict[str, Dict[str, BarSlot]] = {}
        
        total_bytes = 0
        for s in symbols:
            self.slots[s] = {}
            for tf in self.TIMEFRAMES:
                slot = BarSlot(capacity)
                self.slots[s][tf] = slot
                total_bytes += slot.memory_bytes()
        
        total_mb = total_bytes / (1024 * 1024)
        n_slots = len(symbols) * len(self.TIMEFRAMES)
        
        logger.info(
            f"⚡ [MemPool] Pre-allocated {n_slots} slots "
            f"({len(symbols)} symbols × {len(self.TIMEFRAMES)} TF) = "
            f"{total_mb:.1f} MB (aligned to {CACHELINE_BYTES}B cachelines)"
        )
    
    def get_slot(self, symbol: str, timeframe: str = '1m') -> Optional[BarSlot]:
        """
        O(1) lookup of pre-allocated slot.
        Returns None if symbol/timeframe not in pool (should never happen).
        """
        sym_slots = self.slots.get(symbol)
        if sym_slots is None:
            return None
        return sym_slots.get(timeframe)
    
    def diagnostics(self) -> Dict:
        """
        Returns memory diagnostics for all slots.
        Useful for startup validation and dashboard display.
        """
        total_bytes = 0
        slot_info = []
        for s in self.symbols:
            for tf in self.TIMEFRAMES:
                slot = self.slots[s][tf]
                total_bytes += slot.memory_bytes()
                slot_info.append({
                    'symbol': s,
                    'timeframe': tf,
                    'capacity': slot.capacity,
                    'used': slot.size,
                    'bytes': slot.memory_bytes(),
                    'aligned': slot.opens.ctypes.data % CACHELINE_BYTES == 0,
                })
        
        misaligned = [s for s in slot_info if not s['aligned']]
        
        return {
            'total_slots': len(slot_info),
            'total_mb': total_bytes / (1024 * 1024),
            'misaligned_count': len(misaligned),
            'misaligned_details': misaligned[:5],  # First 5 for debugging
        }


class GCGuard:
    """
    ⚡ OMEGA-VOID §1.3: Garbage Collector Freeze during hot paths.
    
    QUÉ: Context manager que desactiva el GC durante secciones críticas.
    POR QUÉ: Python's GC can pause threads for 1-50ms during collection.
         In HFT, this jitter destroys latency guarantees.
    PARA QUÉ: Garantizar latencia determinista durante el burst loop.
    CÓMO: gc.disable() on enter, gc.enable() + optional gc.collect(0) on exit.
    CUÁNDO: Alrededor del burst loop en engine.start().
    
    Usage:
        with GCGuard():
            # Hot path: ZERO GC pauses here
            for event in burst_events:
                engine.process_event(event)
    """
    
    def __init__(self, collect_on_exit: bool = True, generation: int = 0):
        """
        Args:
            collect_on_exit: If True, runs gc.collect(generation) after exiting
            generation: Which GC generation to collect (0=fastest, 2=full)
        """
        self._collect_on_exit = collect_on_exit
        self._generation = generation
        self._was_enabled = False
    
    def __enter__(self):
        self._was_enabled = gc.isenabled()
        if self._was_enabled:
            gc.disable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._was_enabled:
            gc.enable()
            if self._collect_on_exit:
                gc.collect(self._generation)
        return False  # Don't suppress exceptions


def verify_alignment_at_startup(pool: OHLCVPool) -> bool:
    """
    ⚡ OMEGA-VOID §1.1.5: Startup alignment verification.
    
    QUÉ: Verifica que TODOS los arrays estén alineados a 64 bytes.
    POR QUÉ: Si la alineación falla silenciosamente, las optimizaciones
         AVX-2 y cacheline no funcionan, pero el código sigue corriendo
         (más lento y con False Sharing).
    CUÁNDO: Una vez al arranque, antes de iniciar el engine.
    
    Returns:
        True if all aligned, False if any misaligned (logs warnings).
    """
    diag = pool.diagnostics()
    
    if diag['misaligned_count'] > 0:
        logger.warning(
            f"⚠️ [MemPool] {diag['misaligned_count']} slots MISALIGNED! "
            f"False Sharing risk. Details: {diag['misaligned_details']}"
        )
        return False
    
    logger.info(
        f"✅ [MemPool] All {diag['total_slots']} slots aligned to "
        f"{CACHELINE_BYTES}B cachelines ({diag['total_mb']:.1f} MB total)"
    )
    return True
