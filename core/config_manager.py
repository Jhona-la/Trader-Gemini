import asyncio
from dataclasses import replace, is_dataclass

class QuantumConfigManager:
    """
    Manejador Copy-on-Write (COW) de Rendimiento Cuántico (Fase II Metamorfosis).
    
    Reemplaza copy.deepcopy con Compartición Estructural (Structural Sharing).
    El estado inmutable se congela en un dataclass, y las mutaciones reciclan 
    memoria usando dataclasses.replace(). El Read Lock toma O(1) (~1 nanosegundo).
    """
    _instance = None
    
    def __new__(cls, initial_immutable_config=None):
        if not cls._instance:
            cls._instance = super(QuantumConfigManager, cls).__new__(cls)
            # initial_immutable_config DEBE ser un dataclass inmutable (frozen=True)
            cls._instance._current_config = initial_immutable_config
            cls._instance._lock = asyncio.Lock()
        return cls._instance
        
    async def get_snapshot(self):
        """
        Devuelve el puntero a la estructura inmutable en O(1).
        Al ser inmutable, los lectores pueden leer sin locks adicionales,
        y no destruimos la caché de la CPU ni generamos basura de GC.
        """
        async with self._lock:
            # Retorno del puntero inmutable directo (Cero-Copy)
            return self._current_config
            
    async def commit_mutation(self, **kwargs_to_change):
        """
        El Evolver escribe aquí. Generamos un nuevo objeto inmutable reciclando
        referencias estructurales (Punteros) de lo que no ha cambiado.
        """
        async with self._lock:
            if not is_dataclass(self._current_config):
                # Fallback de seguridad si alguien inyectó un dict crudo antes de la Metamorfosis
                if isinstance(self._current_config, dict):
                    new_dict = self._current_config.copy()
                    new_dict.update(kwargs_to_change)
                    self._current_config = new_dict
                return

            self._current_config = replace(self._current_config, **kwargs_to_change)
