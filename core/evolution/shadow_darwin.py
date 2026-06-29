import time
import threading
import logging
import random
import copy
from typing import Dict, List, Any
from core.evolution.genome_registry import GenomeRegistry
from config import Config

# QUÉ: Demonio de evolución continua ShadowDarwinDaemon.
# POR QUÉ: Permite optimizar los umbrales dinámicos (RSI, Bollinger) en tiempo real basado en historia reciente.
# PARA QUÉ: Evitar decaimiento del modelo (alpha decay) y asegurar adaptación al régimen actual.
# CÓMO: Hilo en background que invoca optuna/genetic search con data reciente cargada vía data_provider.
# CUÁNDO: Ejecutado durante la inicialización en engine.py.
# DÓNDE: core/evolution/shadow_darwin.py
# QUIÉN: Arquitecto Senior y Quant Developer.

class ShadowDarwinDaemon:
    """
    [ShadowDarwin] Evolution Daemon (Real-Time Background Optimizer)
    Motor asíncrono que muta parámetros técnicos buscando optimizar el Win Rate
    y reducir el Drawdown. Opera en un hilo secundario sin afectar latencia.
    """
    def __init__(self, config=None, data_provider=None):
        self.config = config or Config
        self.logger = logging.getLogger("ShadowDarwinDaemon")
        self.registry = GenomeRegistry(self.config)
        self.data_provider = data_provider
        self.active = False
        self._thread = None
        
        # Parámetros Evolutivos
        self.cooldown_minutes = 60 * 12 # Mutar cada 12 horas (cada mañana)
        
    def start(self):
        if self.active:
            return
        self.active = True
        self._thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self._thread.start()
        self.logger.info("🐉 [SHADOW-DARWIN] Real-time Evolution Daemon started in background.")

    def stop(self):
        self.active = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self.logger.info("🐉 [SHADOW-DARWIN] Evolution Daemon stopped.")

    def _evolution_loop(self):
        """Bucle infinito de mutación."""
        while self.active:
            try:
                self._run_generation()
            except Exception as e:
                self.logger.error(f"❌ [SHADOW-DARWIN] Mutation error: {e}")
            
            # Cooldown sleep
            for _ in range(self.cooldown_minutes * 60):
                if not self.active:
                    break
                time.sleep(1)

    def _run_generation(self):
        """
        Phase 2: Optuna en tiempo real.
        Debería usar self.data_provider para recolectar datos recientes y usar core.shadow_darwin.ShadowDarwin 
        """
        self.logger.info("🧬 [SHADOW-DARWIN] Waking up for evolutionary run (Background Optuna)...")
        try:
            from core.shadow_darwin import ShadowDarwin
            from core.simulation import SimDataProvider
        except ImportError:
            self.logger.warning("⚠️ [SHADOW-DARWIN] shadow_darwin or SimDataProvider not found.")
            return

        if not self.data_provider:
            self.logger.warning("⚠️ [SHADOW-DARWIN] No data_provider injected. Optuna cannot run on live data. Will use random walk fallback.")
            self._run_random_walk_fallback()
            return
            
        self.logger.info("🧬 [SHADOW-DARWIN] Running real-time Optuna optimization against recent data...")
        try:
            # Creamos un SimDataProvider on-the-fly con los ultimos datos del BinanceLoader
            # Asumiendo que self.data_provider es un BinanceLoader
            all_data = {}
            for symbol in self.config.TRADING_PAIRS:
                df = self.data_provider.get_historical_klines(symbol, self.config.TIMEFRAME.value, limit=1000)
                if df is not None and not df.empty:
                    all_data[symbol] = df
            
            if not all_data:
                self.logger.warning("⚠️ [SHADOW-DARWIN] No recent data fetched. Aborting Optuna.")
                return

            sim_provider = SimDataProvider(all_data)
            optimizer = ShadowDarwin(data_provider=sim_provider)
            
            for symbol in all_data.keys():
                self.logger.info(f"⚗️ [SHADOW-DARWIN] Optimizing {symbol} with Optuna TPE...")
                # Solo corremos ~10 trials para no sobrecargar el CPU en background
                res = optimizer.run_epoch_optuna(symbol, n_trials=10)
                
                if res and 'genotype' in res:
                    new_genes = res['genotype'].genes
                    fitness = res['best_fitness']
                    
                    # Update both SCALPING and SWING (or ideally separate them later)
                    self.registry.update_genome(symbol, "SCALPING", new_genes, fitness)
                    self.registry.update_genome(symbol, "SWING", new_genes, fitness)
                    
            self.logger.info("🏆 [SHADOW-DARWIN] Daily evolution complete.")
            
        except Exception as e:
            self.logger.error(f"❌ [SHADOW-DARWIN] Optuna integration failed: {e}")

    def _run_random_walk_fallback(self):
        """Fallback genético basado en random walk si falla la recolección de data real."""
        gene_bounds = {
            'rsi_buy': (25, 45), 'rsi_sell': (55, 75), 'bb_std': (1.2, 2.5),
            'ema_fast': (5, 12), 'ema_slow': (15, 30), 'atr_period': (5, 14),
            'atr_sl_mult': (1.0, 4.0), 'atr_tp_mult': (1.0, 5.0)
        }
        
        for symbol in self.config.TRADING_PAIRS:
            for horizon in ["SCALPING", "SWING"]:
                current_genes = self.registry.get_genes(symbol, horizon)
                if not current_genes:
                    h_obj = getattr(self.config, "Horizons", None)
                    if not h_obj: continue
                    h_dict = getattr(h_obj, "Scalping" if horizon == "SCALPING" else "Swing", {})
                    current_genes = {k: v for k, v in h_dict.items() if k in gene_bounds}
                
                if not current_genes: continue
                new_genes = copy.deepcopy(current_genes)
                for gene, value in new_genes.items():
                    if gene in gene_bounds and random.random() < 0.15:
                        min_val, max_val = gene_bounds[gene]
                        if isinstance(value, int):
                            shift = max(1, int(value * 0.1))
                            val = value + random.randint(-shift, shift)
                            new_genes[gene] = max(min_val, min(max_val, int(val)))
                        else:
                            shift = value * 0.1
                            val = value + random.uniform(-shift, shift)
                            new_genes[gene] = max(min_val, min(max_val, float(val)))
                
                simulated_fitness = random.uniform(0.5, 2.0)
                self.registry.update_genome(symbol, horizon, new_genes, simulated_fitness)
