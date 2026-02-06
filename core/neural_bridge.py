
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from utils.logger import logger

class NeuralBridge:
    """
    🧠 NEURAL BRIDGE (Phase 8): Shared Intelligence Hub
    ==================================================
    Permite que las estrategias compartan sus 'descubrimientos' y scores
    en tiempo real para forzar un consenso colectivo.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeuralBridge, cls).__new__(cls)
            cls._instance.blackboard = {} # {symbol: {strategy_id: {data}}}
        return cls._instance

    def publish_insight(self, strategy_id: str, symbol: str, insight: Dict[str, Any]):
        """
        Publica un score o descubrimiento sobre un símbolo.
        """
        if symbol not in self.blackboard:
            self.blackboard[symbol] = {}
        
        insight['timestamp'] = datetime.now(timezone.utc).timestamp()
        self.blackboard[symbol][strategy_id] = insight
        
        logger.debug(f"🧠 [BRIDGE] {strategy_id} published insight for {symbol}: {insight}")

    def query_insight(self, symbol: str, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Consulta el último descubrimiento de una estrategia específica.
        """
        return self.blackboard.get(symbol, {}).get(strategy_id)

    def get_collective_consensus(self, symbol: str) -> float:
        """
        Calcula un score promedio basado en todas las inteligencias activas.
        """
        insights = self.blackboard.get(symbol, {})
        if not insights:
            return 0.0
            
        scores = []
        now = datetime.now(timezone.utc).timestamp()
        
        for strat_id, data in insights.items():
            # Solo considerar insights frescos (menos de 5 minutos)
            if now - data.get('timestamp', 0) < 300:
                score = data.get('confidence', 0.0)
                # Si una estrategia detecta una dirección opuesta, el score es negativo
                if data.get('direction') == 'SHORT':
                    score = -score
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0

# Singleton instance
neural_bridge = NeuralBridge()
