"""
🧬 SOPHIA §6.2: Sovereign Oracle (Causal Reasoning Engine)

QUÉ: Motor de razonamiento de segundo orden y atribución de éxito.
POR QUÉ: Para que el bot entienda SI ganar fue por "Habilidad Genética" o "Suerte de Mercado".
PARA QUÉ: Modular la fuerza de la evolución (Mutation Strength) con sabiduría.
"""

from typing import Dict, Any, Optional
from utils.logger import logger
from sophia.post_mortem import PostMortemResult
from core.swarm_correlator import swarm_correlator

class SovereignOracle:
    """
    👁️ THE SIGHT BEYOND NUMBERS.
    Analiza el 'Por Qué' detrás del PnL para decidir el destino genético.
    """
    
    def __init__(self):
        self.last_reasoning = {} # symbol -> reasoning_metadata
        self.knowledge_base = {} # symbol -> list of reasoning_metadata

    def reason_about_outcome(self, result: PostMortemResult) -> Dict[str, Any]:
        """
        SOPHIA-INTELLIGENCE §6.1: The Sovereign Causal Reasoner.
        """
        symbol = result.symbol
        if symbol not in self.knowledge_base:
            self.knowledge_base[symbol] = []
        
        # 1. Análisis de Atribución (Skill vs Luck)
        # Skill = Low Brier Score (High prediction accuracy)
        # Luck = High Correlation with BTC in a winning trade (Swarm Beta)
        correlation = swarm_correlator.correlations.get(symbol, 0.0)
        
        attribution = "UNKNOWN"
        conviction = 0.5 # Default learning speed
        
        if result.actual_outcome == "WIN":
            if result.brier_score < 0.15 and correlation < 0.6:
                attribution = "GENETIC_PRECISION" # Win by skill
                conviction = 0.8 
            elif result.brier_score < 0.15 and correlation >= 0.6:
                attribution = "THE_FORMULA_SECRET" # Divine harmony (Skill + Swarm)
                conviction = 0.5 # Slow down to preserve perfection
            elif correlation > 0.8:
                attribution = "SWARM_BETA" # Riding the wave
                conviction = 1.0
        else:
            if result.brier_score > 0.4:
                attribution = "CALIBRATION_DRIFT" # Bot is confused
                conviction = 1.5 # Accelerate learning/correction
            elif correlation > 0.8:
                attribution = "SWARM_COLLAPSE" # Market took us down
                conviction = 0.7 # Don't overreact to market noise
            else:
                attribution = "GENETIC_FAILURE" # Strategy is wrong
                conviction = 2.0 # Critical correction needed

        reasoning = {
            "attribution": attribution,
            "conviction_score": conviction,
            "correlation": correlation,
            "brier": result.brier_score,
            "narrative": f"Oracle sees {attribution} for {symbol}. Conviction: {conviction:.2f}"
        }
        
        self.knowledge_base[symbol].append(reasoning)
        if len(self.knowledge_base[symbol]) > 50:
            self.knowledge_base[symbol].pop(0)
            
        logger.info(f"🧿 [ORACLE] {reasoning['narrative']}")
        return reasoning

    def get_causal_bias(self, symbol: str) -> Dict[str, float]:
        """
        Retorna el 'Aura' de la moneda: un vector que guía la mutación infinitesimal.
        """
        history = self.knowledge_base.get(symbol, [])
        if not history:
            return {"drift_multiplier": 1.0, "aggression_bias": 0.0}
            
        # Analizar últimos 10 trades
        recent = history[-10:]
        failures = [h for h in recent if "FAILURE" in h['attribution'] or "DRIFT" in h['attribution']]
        successes = [h for h in recent if "PRECISION" in h['attribution'] or "SECRET" in h['attribution']]
        
        bias = 0.0
        if len(failures) > len(successes):
            bias = -0.1 # Sesgo defensivo (reducir TP, aumentar SL filtros)
        elif len(successes) > 2:
            bias = 0.05 # Sesgo agresivo (confianza en los genes)
            
        avg_conviction = sum(h['conviction_score'] for h in recent) / len(recent)
        
        return {
            "drift_multiplier": avg_conviction,
            "aggression_bias": bias,
            "top_attribution": recent[-1]['attribution']
        }

    def get_mutation_mod(self, symbol: str) -> float:
        """Retorna el multiplicador de velocidad de aprendizaje."""
        # Use the latest conviction score from the knowledge base for mutation mod
        history = self.knowledge_base.get(symbol, [])
        if history:
            return history[-1]["conviction_score"]
        return 1.0

sovereign_oracle = SovereignOracle()
