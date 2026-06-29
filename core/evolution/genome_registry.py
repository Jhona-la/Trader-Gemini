import os
import json
import logging
from datetime import datetime
from threading import Lock

class GenomeRegistry:
    """
    [ShadowDarwin] Persistencia de Genomas.
    Almacena los parámetros óptimos descubiertos por el Algoritmo Genético
    para no perder el "aprendizaje" entre reinicios.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, config=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GenomeRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config=None):
        if getattr(self, '_initialized', False):
            return
            
        self.logger = logging.getLogger("GenomeRegistry")
        self.config = config
        
        # Resolve registry path
        try:
            base_dir = self.config.BASE_DIR if self.config else os.getcwd()
            self.registry_dir = os.path.join(base_dir, "core", "evolution", "data")
        except AttributeError:
            self.registry_dir = os.path.join(os.getcwd(), "core", "evolution", "data")
            
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "active_genomes.json")
        
        self.genomes = self._load()
        self._initialized = True

    def _load(self):
        """Carga los genomas desde disco."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"🧬 [GenomeRegistry] Loaded {len(data)} genomes.")
                    return data
            except Exception as e:
                self.logger.error(f"❌ [GenomeRegistry] Error loading {self.registry_file}: {e}")
        return {}

    def _save(self):
        """Guarda atómicamente."""
        with self._lock:
            temp_path = self.registry_file + ".tmp"
            try:
                with open(temp_path, 'w') as f:
                    json.dump(self.genomes, f, indent=4)
                os.replace(temp_path, self.registry_file)
            except Exception as e:
                self.logger.error(f"❌ [GenomeRegistry] Error saving: {e}")

    def update_genome(self, symbol: str, horizon: str, genes: dict, fitness: float):
        """
        Actualiza el genoma de un activo/horizonte si el nuevo fitness es mejor.
        """
        key = f"{symbol}_{horizon}"
        if key not in self.genomes or fitness > self.genomes[key]['fitness']:
            self.genomes[key] = {
                'genes': genes,
                'fitness': fitness,
                'timestamp': datetime.utcnow().isoformat()
            }
            self._save()
            self.logger.debug(f"🧬 [GenomeRegistry] Upgraded {key} with fitness {fitness:.4f}")
            return True
        return False

    def get_genes(self, symbol: str, horizon: str) -> dict:
        """Obtiene los genes activos para inyección en technical.py."""
        key = f"{symbol}_{horizon}"
        if key in self.genomes:
            return self.genomes[key]['genes']
        return None

    def apply_rl_gradient(self, symbol: str, horizon: str, gradients: dict):
        """
        [NanoRL] Aplica un gradiente (ascenso/descenso estocástico) a los genes activos.
        Muta el valor en memoria y lo guarda.
        """
        key = f"{symbol}_{horizon}"
        if key not in self.genomes:
            return
            
        genes = self.genomes[key]['genes']
        
        for g_key, grad in gradients.items():
            if g_key in genes:
                # Si el gen es entero, acumulamos el gradiente fraccional en una memoria fantasma?
                # Para simplificar y por performance, simplemente usamos flotantes o redondeamos si es explícitamente int.
                current_val = genes[g_key]
                if isinstance(current_val, int) and not isinstance(current_val, bool):
                    # Acumuladores probabilísticos (Estocástico)
                    import random
                    if random.random() < abs(grad):
                        genes[g_key] += 1 if grad > 0 else -1
                else:
                    genes[g_key] += grad
                    
                # Sanity Caps (Hard limits)
                if 'rsi' in g_key:
                    genes[g_key] = max(10, min(90, genes[g_key]))
                elif 'mult' in g_key or 'std' in g_key:
                    genes[g_key] = max(0.5, min(10.0, genes[g_key]))

        self.genomes[key]['timestamp'] = datetime.utcnow().isoformat()
        self._save()
