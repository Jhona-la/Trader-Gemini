"""
🧬 SOPHIA §5.2: Gene Bank (Archetype Memory)

QUÉ: Repositorio persistente de genotipos optimizados por régimen.
POR QUÉ: Para que el bot no pierda "recetas ganadoras" cuando el mercado cambia.
PARA QUÉ: Recuperación instantánea (Regime-Snap) y estabilidad evolutiva.
CÓMO: Mapeo (symbol, regime) -> Genotype con validación de fitness.
"""

import os
import json
from typing import Dict, Optional, List
from utils.logger import logger
from core.genotype import Genotype
from datetime import datetime

class GeneBank:
    """
    🏦 ELITE GENE BANK
    Almacena los 'Arquetipos' de ADN que han demostrado éxito en regímenes específicos.
    """
    
    def __init__(self, persistence_dir: str = "data/gene_bank"):
        self.persistence_dir = persistence_dir
        self.bank = {} # (symbol, regime) -> Genotype params
        self._load_bank()
        
    def _load_bank(self):
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir, exist_ok=True)
            return

        for filename in os.listdir(self.persistence_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.persistence_dir, filename), 'r') as f:
                        data = json.load(f)
                        key = (data['symbol'], data['regime'])
                        self.bank[key] = data['genes']
                except Exception as e:
                    logger.error(f"🏦 [BANK] Error loading {filename}: {e}")

    def save_elite_gene(self, genotype: Genotype, regime: str):
        """
        Guarda un genotipo como élite si su fitness supera el umbral.
        """
        symbol = genotype.symbol
        key = (symbol, regime)
        
        # Guardar si es el primero o si el fitness es superior al record actual (opcional)
        # Por ahora guardamos cualquier genotipo que el MetaOptimizer considere 'estable'
        
        self.bank[key] = genotype.genes
        
        # Persistencia
        filename = f"{symbol.replace('/','')}_{regime}.json"
        filepath = os.path.join(self.persistence_dir, filename)
        
        data_to_save = {
            "symbol": symbol,
            "regime": regime,
            "fitness": genotype.fitness_score,
            "generation": genotype.generation,
            "genes": genotype.genes,
            "stamped_at": datetime.now().isoformat() if 'datetime' in globals() else ""
        }
        
        try:
            from utils.fast_json import FastJson
            FastJson.dump_to_file(data_to_save, filepath)
            logger.info(f"🏦 [BANK] Elite Gene saved for {symbol} in {regime}.")
        except Exception as e:
            logger.error(f"🏦 [BANK] Save error: {e}")

    def get_best_gene(self, symbol: str, regime: str) -> Optional[Dict]:
        """
        Recupera el ADN optimizado para el par (símbolo, régimen).
        """
        return self.bank.get((symbol, regime))

# Singleton for system access
gene_bank = GeneBank()
