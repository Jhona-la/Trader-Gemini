
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import logging

logger = logging.getLogger("SharedMemoryManager")

class SharedMemoryManager:
    """
    Gestor de contexto para manejo seguro de Memoria Compartida (Zero-Copy).
    """
    def __init__(self, data_array):
        """
        Crea un bloque de memoria compartida inicializado con data_array.
        Args:
            data_array (np.ndarray): Array a compartir.
        """
        self.data_array = data_array
        self.shm = None
        self.name = None
        self.shape = data_array.shape
        self.dtype = data_array.dtype
        self.nbytes = data_array.nbytes

    def __enter__(self):
        try:
            # Crear bloque de memoria compartida
            self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
            self.name = self.shm.name
            
            # Crear array numpy respaldado por shared memory
            shared_arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            
            # Copiar datos al buffer compartido (ÚNICA COPIA)
            # Esto es mucho más rápido que pickle para grandes volúmenes
            shared_arr[:] = self.data_array[:]
            
            return self
        except Exception as e:
            logger.error(f"❌ SharedMemory creation failed: {e}")
            if self.shm:
                self.shm.close()
                self.shm.unlink()
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink() # Liberar memoria del OS inmediatamente
                # logger.debug(f"🧹 Unlinked SharedMemory: {self.name}")
            except Exception as e:
                logger.error(f"Error cleaning up SharedMemory: {e}")

def load_shared_array(name, shape, dtype):
    """
    Carga un array desde SharedMemory existente (Worker Side).
    Args:
        name (str): Nombre del bloque shm.
        shape (tuple): Dimensiones.
        dtype (dtype): Tipo de dato.
    Returns:
        tuple: (numpy_array, shm_object) - EL CALLER DEBE CERRAR SHM_OBJECT.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr, shm
    except FileNotFoundError:
        return None, None
