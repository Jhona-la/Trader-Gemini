import os
import numpy as np
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from utils.logger import logger
from core.genotype import Genotype
from sophia.post_mortem import PostMortemResult
from core.gene_bank import gene_bank
from core.multiverse_simulator import multiverse_simulator
from core.sovereign_oracle import sovereign_oracle

class MetaOptimizer:
    """
    🧠 THE SOVEREIGN META-PREDICTOR Core.
    
    Encargado de la "Individuación" de cada moneda y la adaptación infinitesimal.
    """
    
    def __init__(self, mutation_strength: float = 1e-6):
        self.mutation_strength = mutation_strength
        self.history = {} # symbol -> List[PostMortemResult]
        self.telemetry_file = "data/fabric_telemetry.json"
        
        # Cargar telemetría existente
        self.telemetry = self._load_telemetry()
        
        logger.info(f"🧬 [META-OPTIMIZER] Initialized with Infinitesimal Strength: {mutation_strength}")

    def _load_telemetry(self) -> Dict[str, List[Dict]]:
        if os.path.exists(self.telemetry_file):
            try:
                with open(self.telemetry_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_fabric_telemetry(self, symbol: str, genes: Dict[str, float], result: PostMortemResult, causality: str):
        """
        Persiste la deriva de parámetros para visualización en Dashboard.
        """
        if symbol not in self.telemetry:
            self.telemetry[symbol] = []
            
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": result.trade_id,
            "outcome": result.actual_outcome,
            "pnl": float(result.actual_pnl),
            "brier": float(result.brier_score),
            "causality": causality,
            "parameters": {k: float(v) if isinstance(v, (float, np.float32, np.float64, int)) else v for k, v in genes.items() if isinstance(v, (float, np.float32, np.float64, int))}
        }
        
        self.telemetry[symbol].append(entry)
        
        # Limitar historial por símbolo (mantener últimos 100 ajustes)
        self.telemetry[symbol] = self.telemetry[symbol][-100:]
        
        try:
            os.makedirs(os.path.dirname(self.telemetry_file), exist_ok=True)
            with open(self.telemetry_file, 'w') as f:
                json.dump(self.telemetry, f, indent=2)
        except Exception as e:
            logger.error(f"❌ [META] Error saving telemetry: {e}")

    def process_trade_result(self, result: PostMortemResult, genotype: Optional[Genotype] = None):
        """
        Analiza el resultado de un trade y ajusta el Genoma con Causalidad.
        """
        symbol = result.symbol
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(result)
        
        # --- PHASE 47: SOVEREIGN ORACLE (Reasoning) ---
        reasoning = sovereign_oracle.reason_about_outcome(result)
        conviction_mod = reasoning["conviction_score"]
        
        if genotype is None:
            filename = f"data/genotypes/{symbol.replace('/','')}_gene.json"
            genotype = Genotype.load(filename)
            if not genotype:
                logger.warning(f"🧬 [META] No genotype found for {symbol}. Creating default.")
                genotype = Genotype(symbol=symbol)
        
        multiplier = 1.0 if result.actual_outcome == "WIN" else -1.0
        
        # 1. Definir Causalidad (Explicabilidad)
        causality = "REINFORCEMENT" if multiplier > 0 else "DEFENSIVE_ADAPTATION"
        if result.brier_score > 0.3:
            causality += "_HIGH_ERROR_CORRECTION"
            
        # --- PHASE 46.1: QUANTUM WEAVER ---
        # 2. Consultar Gene Bank si el fitness es bajo
        if genotype.fitness_score < 0.2:
            archetype = gene_bank.get_best_gene(symbol, "TRENDING_BULL") # Mock regime for now
            if archetype:
                logger.info(f"🧬 [META] Low fitness detected. Weaving Archetype for {symbol}.")
                # Mezclamos un 10% del arquetipo para evitar saltos bruscos
                for k, v in archetype.items():
                    if k in genotype.genes and isinstance(v, (int, float)) and isinstance(genotype.genes[k], (int, float)):
                        genotype.genes[k] = (genotype.genes[k] * 0.9) + (v * 0.1)

        genes = genotype.genes
        # Aplicamos el modificador de convicción del Oráculo a la fuerza de mutación
        effective_mutation = self.mutation_strength * conviction_mod
        drift = effective_mutation * (1.0 + result.brier_score) * multiplier
        
        tunable = ['tp_pct', 'sl_pct', 'strength_threshold', 'adx_threshold', 'chaos_dampening', 'certainty_floor']
        
        for p in tunable:
            if p in genes:
                try:
                    old_val = float(genes[p])
                    new_val = old_val * (1.0 + drift)
                    
                    # Sanity Check: Mantener escalas lógicas
                    if p.endswith('_pct'):
                        new_val = np.clip(new_val, 0.005, 0.10)
                    elif 'threshold' in p:
                        new_val = np.clip(new_val, 0.1, 0.95)
                    elif 'adx' in p:
                        new_val = np.clip(new_val, 10, 45)
                    elif p == 'chaos_dampening':
                        new_val = np.clip(new_val, 0.1, 1.5)
                    elif p == 'certainty_floor':
                        new_val = np.clip(new_val, 0.0, 0.90)
                        
                    genes[p] = new_val
                except (TypeError, ValueError):
                    # If it's not a float/int, skip or handle accordingly
                    logger.warning(f"🧬 [META] Skipping non-numeric gene {p} for {symbol}: {type(genes[p])}")
                    continue
                
        # 3. Evolución del Cerebro (Neural Weights)
        if 'brain_weights' in genes and genes['brain_weights']:
            aura = sovereign_oracle.get_causal_bias(symbol)
            weights = np.array(genes['brain_weights'])
            
            # Sesgo direccional: Si hay éxito, el ruido es menor. Si hay fracaso, el ruido es mayor.
            noise_scale = self.mutation_strength * result.brier_score * aura['drift_multiplier']
            noise = np.random.normal(0, noise_scale, size=weights.shape)
            
            # Aplicar sesgo de agresión al ruido si existe
            weights += (noise * multiplier) + (aura['aggression_bias'] * self.mutation_strength)
            genes['brain_weights'] = weights.tolist()

        logger.info(f"🧠 [META-INSIGHT] {symbol} {causality}. Brier={result.brier_score:.4f}. Gen={genotype.generation}")
        
        genotype.generation += 1
        genotype.fitness_score = self._calculate_rolling_fitness(symbol)
        
        # 4. Persistencia Genética
        filename = f"data/genotypes/{symbol.replace('/','')}_gene.json"
        os.makedirs("data/genotypes", exist_ok=True)
        genotype.save(filename)
        
        # 5. Telemetría de la Evolución
        self._save_fabric_telemetry(symbol, genes, result, causality)
        
        # 6. Salvar en Gene Bank si es Élite
        if genotype.fitness_score > 0.6: # Umbral de Élite
            # Detectar régimen (Simplificado por ahora)
            current_regime = "NORMAL" 
            gene_bank.save_elite_gene(genotype, current_regime)

    def _calculate_rolling_fitness(self, symbol: str) -> float:
        results = self.history[symbol]
        if not results: return 0.0
        window = results[-20:]
        win_rate = sum(1 for r in window if r.actual_outcome == "WIN") / len(window)
        avg_pnl = sum(r.actual_pnl for r in window) / len(window)
        return win_rate * (1.0 + avg_pnl)

meta_optimizer = MetaOptimizer()


def suggest_horizon_params(horizon_days: int, symbol: str = 'BTC/USDT') -> Dict[str, Any]:
    """
    Adaptive Evolution Protocol: Pre-Optimized Parameters per Horizon.
    
    QUÉ: Retorna un diccionario de parámetros de trading optimizados por horizonte.
    POR QUÉ: Un gen que funciona en 30D puede ser catastrófico en 1D. Los
              backtests revelaron que tp_pct, sl_pct, y los factores de
              dampening deben variar significativamente entre horizontes.
    PARA QUÉ: Inicializar el Genotype de cada símbolo con valores
               óptimos según el horizonte de operación actual.
    CÓMO: Basado en los hallazgos del Multi-Horizon Audit:
          - 1D: Targets ajustados, dampening completo
          - 15D: Targets intermedios, dampening reducido
          - 30D: Targets amplios, dampening mínimo
    CUÁNDO: Al iniciar una sesión de trading o backtest.
    DÓNDE: core/meta_optimizer.py
    QUIÉN: Engine.py, run_backtest.py.
    """
    if horizon_days <= 1:
        return {
            'tp_pct': 0.008,
            'sl_pct': 0.005,
            'chaos_dampening': 1.0,
            'certainty_floor': 0.0,
            'strength_threshold': 0.55,
            'adx_threshold': 25,
            'horizon': 'SCALPING'
        }
    elif horizon_days <= 7:
        return {
            'tp_pct': 0.010,
            'sl_pct': 0.006,
            'chaos_dampening': 0.7,
            'certainty_floor': 0.35,
            'strength_threshold': 0.50,
            'adx_threshold': 22,
            'horizon': 'SHORT_TERM'
        }
    elif horizon_days <= 15:
        return {
            'tp_pct': 0.012,
            'sl_pct': 0.008,
            'chaos_dampening': 0.5,
            'certainty_floor': 0.50,
            'strength_threshold': 0.45,
            'adx_threshold': 20,
            'horizon': 'MID_TERM'
        }
    else:
        return {
            'tp_pct': 0.015,
            'sl_pct': 0.008,
            'chaos_dampening': 0.3,
            'certainty_floor': 0.70,
            'strength_threshold': 0.40,
            'adx_threshold': 18,
            'horizon': 'MACRO'
        }
