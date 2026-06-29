import os
import logging
from typing import Dict, List, Any
from config import Config
from core.genotype import Genotype
from strategies.technical import HybridScalpingStrategy # Will need wrapper or refactor

logger = logging.getLogger("SymbolFactory")

class SymbolFactory:
    """
    Orquestador de Vida (Trinidad Omega - Block A).
    Gestiona la creación, monitoreo y destrucción de instancias de estrategia por símbolo.
    """
    def __init__(self, engine):
        self.engine = engine
        self.active_organisms: Dict[str, Any] = {} # symbol -> strategy_instance
        self.genotype_dir = "data/genotypes"
        
        if not os.path.exists(self.genotype_dir):
            os.makedirs(self.genotype_dir)

    def scan_watchlist(self):
        """
        Escanea la Watchlist y asegura que exista un organismo para cada símbolo.
        """
        target_symbols = Config.CRYPTO_FUTURES_PAIRS[:20] # Limit to 20 for HFT checks
        
        # 1. Spawn missing organisms
        for symbol in target_symbols:
            if symbol not in self.active_organisms:
                self.spawn_organism(symbol)
                
        # 2. Kill retired organisms
        active_symbols = list(self.active_organisms.keys())
        for symbol in active_symbols:
            if symbol not in target_symbols:
                self.kill_organism(symbol)

    def spawn_organism(self, symbol: str):
        """
        Crea una nueva instancia de estrategia (Organismo) para el símbolo.
        Carga su ADN (Genotype) o crea uno nuevo si no existe.
        """
        try:
            logger.info(f"🧬 Spawning Organism: {symbol}")
            
            # 1. Load or Create Genotype
            gene_path = os.path.join(self.genotype_dir, f"{symbol.replace('/','')}_gene.json")
            genotype = Genotype.load(gene_path)
            
            if not genotype:
                logger.info(f"✨ Creating Genesis Genotype for {symbol}")
                genotype = Genotype(symbol=symbol)
                # Apply initial profile logic from Phase 7.2 here if needed
                # For now, default genesis
                genotype.save(gene_path)
            
            # 2. Instantiate Strategy (Inject Genotype)
            # Note: We need a wrapper or refactored strategy that accepts 'genotype'
            # For Phase 1, we might inject it after creation if __init__ isn't ready
            strategy = HybridScalpingStrategy(self.engine.data_provider, self.engine.events)
            strategy.symbol = symbol # FORCE SINGLE SYMBOL MODE
            strategy.genotype = genotype # Inject DNA
            
            # 3. Register with Engine
            self.engine.register_strategy(strategy)
            self.active_organisms[symbol] = strategy
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn {symbol}: {e}")

    def kill_organism(self, symbol: str):
        """
        Elimina un organismo del ecosistema.
        """
        if symbol in self.active_organisms:
            logger.info(f"💀 Killing Organism: {symbol}")
            strategy = self.active_organisms[symbol]
            
            # 1. Unregister
            self.engine.unregister_strategy(symbol)
            
            # 2. Cleanup
            del self.active_organisms[symbol]
