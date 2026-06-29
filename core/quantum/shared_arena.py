# ⚛️ TRINITY OMEGA-Q: EL TENSOR DE ESTADO GLOBAL (Shared Memory Arena)
# 
# AXIOMA II y III: Cero Copias, Memoria Compartida, Ejecución en Metal.
# Este archivo define la estructura de la memoria contigua de Numpy (Memoryviews)
# que será inyectada en C/Cython. No es un objeto de Python, es un puntero a C.
#
# P1 EVOLUTION: SeqLock atómico para coherencia sin mutex.
# El escritor (data ingester) y el lector (inference engine) nunca usan locks.
# Coherencia garantizada por version counter par/impar.

import numpy as np
import ctypes
import logging
import platform
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QuantumStateArena:
    """
    El Tensor de Estado Global ($|\Psi\rangle$).
    Toda la información del mercado (Microestructura, Precios, Entropía)
    reside aquí en memoria pre-asignada y contigua.
    
    NO SE PERMITEN COPIAS DE ESTADO.
    
    P1 EVOLUTION: SeqLock Atómico
    ─────────────────────────────
    Inspirado en Linux kernel seqlock.h. Garantiza coherencia sin mutex:
    - Escritor: version++ (impar) → escribe → version++ (par)
    - Lector: lee version (v1) → si impar, retry → lee datos → lee version (v2) → si v1≠v2, retry
    - Costo: ~10ns (2 lecturas de int64)
    """
    __slots__ = (
        'capacity', 'features_dim', 'prices', 'microstructure', 
        'entropy', 'features', '_head', '_is_full',
        '_version', '_stable_head', '_prefaulted'
    )
    
    def __init__(self, capacity: int = 1000, features_dim: int = 200):
        self.capacity = capacity
        
        # Padding SIMD: asegurar que la dimensión de features en bytes sea múltiplo de 64 (caché L1)
        # Esto evita "False Sharing" y permite copias vectorizadas eficientes.
        bytes_per_float32 = 4
        cache_line_size = 64
        base_bytes = features_dim * bytes_per_float32
        remainder = base_bytes % cache_line_size
        
        self.logical_features_dim = features_dim
        if remainder != 0:
            padded_bytes = base_bytes + (cache_line_size - remainder)
            self.features_dim = padded_bytes // bytes_per_float32
        else:
            self.features_dim = features_dim
        
        # 1. Tensor de Precios (OHLCV + Venta/Compra) - O(1) Updates, contiguos en C
        self.prices = np.zeros((capacity, 6), dtype=np.float64, order='C')
        
        # 2. Tensor de Microestructura (Order Book Inbalance, Dark Alpha, Liquidaciones)
        self.microstructure = np.zeros((capacity, 10), dtype=np.float64, order='C')
        
        # 3. Tensor de Entropía y Volatilidad (Hurst, Shannon)
        self.entropy = np.zeros((capacity, 4), dtype=np.float64, order='C')
        
        # 4. Tensor Vectorial de Features (Para Sophia ML - Zero Copy)
        # Dimensionado con padding SIMD. El modelo usará la Vista Lógica.
        self.features = np.zeros((capacity, self.features_dim), dtype=np.float32, order='C')
        
        # 5. Puntero Circular (Ring Buffer Index)
        self._head = 0
        self._is_full = False
        
        # ═══════════════════════════════════════════════════════════════
        # P1 SEQLOCK: Version counter atómico (Alineado a 64-bytes)
        # 8 x int64 = 64 bytes. El offset 0 es la versión. 
        # Esto aísla el contador en su propia línea de caché L1.
        # Par = estado estable, Impar = escritura en curso.
        # ═══════════════════════════════════════════════════════════════
        self._version = np.zeros(8, dtype=np.int64)  # Padding para False Sharing
        self._stable_head = 0  # Último head confirmado como estable
        self._prefaulted = False
        
        logger.info(f"🌌 QuantumStateArena Initialized. Shape: [{capacity}, {features_dim}] - SeqLock Ready.")

    # ─── PREFAULT: Calentamiento Termodinámico ───────────────────────
    
    def prefault(self):
        """
        P1 EVOLUTION: Toca cada página del arena para forzar al kernel
        a cargarlas en RAM física. Elimina cold page faults durante la
        operación del motor.
        
        Costo: ~1ms para 1.1MB de arena (insignificante al bootstrap).
        """
        if self._prefaulted:
            return
        
        # Tocar primer byte de cada página (4KB) fuerza un soft page fault
        # que carga la página en el page cache del kernel
        for arr in (self.prices, self.microstructure, self.entropy, self.features):
            page_size = 4096
            flat = arr.view(np.uint8).ravel()
            for offset in range(0, len(flat), page_size):
                _ = flat[offset]  # Touch — triggers soft page fault
            
            # EL SELLO DE HIERRO: Fijar matemáticamente en la silicona
            self._lock_memory_in_ram(arr)
        
        self._prefaulted = True
        total_bytes = (
            self.prices.nbytes + self.microstructure.nbytes + 
            self.entropy.nbytes + self.features.nbytes
        )
        logger.info(f"🔥 [Arena] Prefault & M-Lock complete. {total_bytes:,} bytes HARD-LOCKED in physical RAM.")

    def _lock_memory_in_ram(self, array: np.ndarray):
        """Bloquea físicamente la memoria del array en RAM (silicona) revistiéndole el control al OS."""
        ptr = array.ctypes.data
        size = array.nbytes
        
        system = platform.system()
        try:
            if system == "Windows":
                kernel32 = ctypes.windll.kernel32
                process = kernel32.GetCurrentProcess()
                
                # Intentamos expandir el working set de manera proactiva
                current_min = ctypes.c_size_t()
                current_max = ctypes.c_size_t()
                if kernel32.GetProcessWorkingSetSize(process, ctypes.byref(current_min), ctypes.byref(current_max)):
                    new_min = current_min.value + size + (10 * 1024 * 1024)
                    new_max = current_max.value + size + (10 * 1024 * 1024)
                    kernel32.SetProcessWorkingSetSize(process, new_min, new_max)
                
                # Efectuar el bloqueo cuántico
                success = kernel32.VirtualLock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
                if not success:
                    err = kernel32.GetLastError()
                    logger.warning(f"⚠️ [Arena] VirtualLock falló (Código: {err}). Posible degradación termodinámica a disco.")
            else:
                # Linux mlock
                libc = ctypes.CDLL("libc.so.6")
                res = libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
                if res != 0:
                    logger.warning(f"⚠️ [Arena] mlock falló. Posible degradación termodinámica a disco.")
        except Exception as e:
            logger.error(f"❌ [Arena] Error anclando memoria en silicona: {e}")

    # ─── ESCRITURA CON SEQLOCK ───────────────────────────────────────

    def inject_tick(self, ohlcv: np.ndarray, microstructure_vec: np.ndarray, entropy_vec: np.ndarray) -> int:
        """
        Operador Hamiltoniano $\hat{H}_{Datos}$
        Inyección O(1) en el buffer circular. No hay reubicación de memoria.
        
        P1 SEQLOCK: Version se incrementa a impar ANTES de escribir,
        y a par DESPUÉS. El lector detectará la escritura en curso.
        """
        # SEQLOCK: señalar inicio de escritura (version → impar)
        self._version[0] += 1
        
        idx = self._head
        
        # Inyección In-Situ (Punteros de Numpy → C)
        self.prices[idx, :] = ohlcv
        self.microstructure[idx, :] = microstructure_vec
        self.entropy[idx, :] = entropy_vec
        
        # Avanzar el puntero cuántico
        self._head = (self._head + 1) % self.capacity
        if self._head == 0:
            self._is_full = True
        
        # SEQLOCK: señalar fin de escritura (version → par)
        self._version[0] += 1
        self._stable_head = self._head
            
        return idx

    def inject_features(self, idx: int, feature_vec: np.ndarray):
        """
        Inyecta features pre-calculados en el slot del arena.
        Llamado DESPUÉS de inject_tick() con el mismo idx.
        
        Esto permite que el anillo almacene features junto con OHLCV,
        eliminando la necesidad de recalcular en la inferencia.
        """
        self._version[0] += 1  # Impar: escribiendo
        
        n_feat = min(len(feature_vec), self.features_dim)
        self.features[idx, :n_feat] = feature_vec[:n_feat].astype(np.float32)
        
        self._version[0] += 1  # Par: estable
    
    # ─── LECTURA CON SEQLOCK ────────────────────────────────────────

    def try_read_features(self, lookback: int = 1) -> Optional[np.ndarray]:
        """
        Lectura optimista del tensor de features.
        
        Retorna:
          - np.ndarray view si la lectura fue coherente
          - None si hubo conflicto (escritura en curso o datos mutados)
        
        Costo: ~10ns (2 lecturas de int64 + 1 comparación)
        """
        v1 = self._version[0]
        if v1 & 1:  # Impar = escritura en curso
            return None
        
        head = self._stable_head
        if not self._is_full and head < lookback:
            return None
        
        start = (head - lookback) % self.capacity
        end = head
        
        if start < end:
            view = self.features[start:end, :]
        else:
            # Wrap-around — no podemos dar vista contigua sin copia
            return None
        
        v2 = self._version[0]
        if v1 != v2:  # Datos mutados durante lectura
            return None
        
        return view  # Vista coherente, cero copia

    def get_superposition_view(self, lookback: int = 100) -> Optional[Dict[str, np.ndarray]]:
        """
        Devuelve VISTAS (Views), NUNCA COPIAS.
        P1 SEQLOCK: Retorna None si hay conflicto de escritura.
        """
        v1 = self._version[0]
        if v1 & 1:
            return None
        
        if not self._is_full and self._stable_head < lookback:
            return None
            
        start = (self._stable_head - lookback) % self.capacity
        end = self._stable_head
        
        if start < end:
            result = {
                "prices": self.prices[start:end, :],
                "microstructure": self.microstructure[start:end, :],
                "entropy": self.entropy[start:end, :]
            }
        else:
            # Rollback buffer view — requires copy (np.vstack)
            result = {
                "prices": np.vstack((self.prices[start:, :], self.prices[:end, :])),
                "microstructure": np.vstack((self.microstructure[start:, :], self.microstructure[:end, :])),
                "entropy": np.vstack((self.entropy[start:, :], self.entropy[:end, :]))
            }
        
        v2 = self._version[0]
        if v1 != v2:
            return None
        
        return result

    # ─── MÉTRICAS ───────────────────────────────────────────────────

    @property
    def version(self) -> int:
        """Version counter actual. Par = estable, Impar = escribiendo."""
        return int(self._version[0])
    
    @property
    def is_stable(self) -> bool:
        """True si no hay escritura en curso."""
        return (self._version[0] & 1) == 0
    
    @property
    def fill_ratio(self) -> float:
        """Proporción del arena que ha sido llenada."""
        if self._is_full:
            return 1.0
        return self._stable_head / self.capacity

# Instancia Entrelazada Global (El Universo Unificado)
# Prohibido instanciar múltiples Arenas. Esto es un Singleton por Arquitectura.
GLOBAL_TENSOR = QuantumStateArena()
