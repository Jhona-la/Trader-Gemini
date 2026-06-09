import time
import threading
import logging
import random
import copy
from typing import Dict, List, Any
from core.evolution.genome_registry import GenomeRegistry
from config import Config

class ShadowDarwin:
    """
    [ShadowDarwin] Evolution Daemon
    Motor asíncrono que muta parámetros técnicos buscando optimizar el Win Rate
    y reducir el Drawdown. Opera en un hilo secundario sin afectar latencia.
    """
    def __init__(self, config=None):
        self.config = config or Config
        self.logger = logging.getLogger("ShadowDarwin")
        self.registry = GenomeRegistry(self.config)
        self.active = False
        self._thread = None
        
        # Parámetros Evolutivos
        self.population_size = 5
        self.mutation_rate = 0.15
        self.cooldown_minutes = 60 # Mutar cada hora para no saturar
        
        # Base limits para mutaciones
        self.gene_bounds = {
            'rsi_buy': (25, 45),
            'rsi_sell': (55, 75),
            'bb_std': (1.2, 2.5),
            'ema_fast': (5, 12),
            'ema_slow': (15, 30),
            'atr_period': (5, 14),
            'atr_sl_mult': (1.0, 4.0),
            'atr_tp_mult': (1.0, 5.0)
        }

    def start(self):
        if self.active:
            return
        self.active = True
        self._thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self._thread.start()
        self.logger.info("🧬 [ShadowDarwin] Evolution Daemon started in background.")

    def stop(self):
        self.active = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self.logger.info("🧬 [ShadowDarwin] Evolution Daemon stopped.")

    def _evolution_loop(self):
        """Bucle infinito de mutación. 1 iteración por hora."""
        while self.active:
            try:
                self._run_generation()
            except Exception as e:
                self.logger.error(f"❌ [ShadowDarwin] Mutation error: {e}")
            
            # Cooldown sleep
            for _ in range(self.cooldown_minutes * 60):
                if not self.active:
                    break
                time.sleep(1)

    def _run_generation(self):
        """
        Ejecuta una generación de mutaciones.
        En producción real, esto leería del historial PnL (Alpha Leak) 
        para calcular el fitness. Aquí simulamos la inyección para la arquitectura.
        """
        self.logger.debug("🧬 [ShadowDarwin] Running new evolutionary generation...")
        
        for symbol in self.config.TRADING_PAIRS:
            for horizon in ["SCALPING", "SWING"]:
                # 1. Obtener genes actuales o fallbacks
                current_genes = self.registry.get_genes(symbol, horizon)
                if not current_genes:
                    # Usar base config
                    h_obj = getattr(self.config, "Horizons", None)
                    if not h_obj: continue
                    h_dict = getattr(h_obj, "Scalping" if horizon == "SCALPING" else "Swing", {})
                    current_genes = {k: v for k, v in h_dict.items() if k in self.gene_bounds}
                
                if not current_genes: continue

                # 2. Mutar (Simples Random Walks)
                mutated_genes = self._mutate(current_genes)
                
                # 3. Evaluar Fitness simulado (aquí conectaríamos con el PnL real del Portfolio)
                # Para la prueba arquitectónica, simplemente lo registramos como una mutación válida
                # con un fitness aleatorio superior para que se guarde.
                simulated_fitness = random.uniform(0.5, 2.0)
                
                # 4. Guardar si es mejor
                self.registry.update_genome(symbol, horizon, mutated_genes, simulated_fitness)

    def _mutate(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica mutaciones gaussianas limitadas por los bounds."""
        new_genes = copy.deepcopy(genes)
        for gene, value in new_genes.items():
            if gene in self.gene_bounds and random.random() < self.mutation_rate:
                min_val, max_val = self.gene_bounds[gene]
                
                # Mutación Gaussiana (+- 10%)
                if isinstance(value, int):
                    shift = int(value * 0.1) or 1
                    val = value + random.randint(-shift, shift)
                    new_genes[gene] = max(min_val, min(max_val, int(val)))
                else:
                    shift = value * 0.1
                    val = value + random.uniform(-shift, shift)
                    new_genes[gene] = max(min_val, min(max_val, float(val)))
        return new_genes
