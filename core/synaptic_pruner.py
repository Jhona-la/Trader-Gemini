import logging
import time
from typing import Tuple

logger = logging.getLogger("SynapticPruner")

class SynapticPruner:
    """
    🧠 [MUTACIÓN 35] Genetic Pruning (Cross-Fertilization Matrix)
    QUÉ: En vez de silenciar binariamente, muta el ADN (TP/SL) de las estrategias.
    POR QUÉ: Para maximizar el Win Rate en $13 USD, si el mercado cambia, la 
             estrategia debe adaptarse en tiempo real en lugar de apagarse.
    PARA QUÉ: Devolver un vector de mutación (Trust, TP_Mod, SL_Mod) al Engine.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SynapticPruner()
        return cls._instance

    def __init__(self):
        # strategy_id -> trust_score (0.0 a 2.0)
        self.trust_scores = {}
        # strategy_id -> (tp_modifier, sl_modifier)
        self.genetic_matrix = {}
        # Historial de trades recientes por estrategia: [1 (Win), -1 (Loss)]
        self.streak_tracker = {}
        self.last_punishment_time = {}

    def report_trade_result(self, strategy_id: str, pnl_pct: float):
        """Llamado por RiskManager al cerrar una operación."""
        if not strategy_id: return
        
        if strategy_id not in self.trust_scores:
            self.trust_scores[strategy_id] = 1.0
            self.genetic_matrix[strategy_id] = [1.0, 1.0] # TP, SL modifiers
            self.streak_tracker[strategy_id] = []
            
        is_win = 1 if pnl_pct > 0 else -1
        
        # Guardar últimos 5 trades
        self.streak_tracker[strategy_id].append(is_win)
        if len(self.streak_tracker[strategy_id]) > 5:
            self.streak_tracker[strategy_id].pop(0)
            
        recent_streak = sum(self.streak_tracker[strategy_id][-2:])
        
        if is_win == 1:
            # 🟢 DOPAMINA
            self.trust_scores[strategy_id] = min(2.0, self.trust_scores[strategy_id] + 0.3)
            # Restaurar ADN poco a poco
            self.genetic_matrix[strategy_id][0] = min(1.0, self.genetic_matrix[strategy_id][0] + 0.1)
            self.genetic_matrix[strategy_id][1] = min(1.0, self.genetic_matrix[strategy_id][1] + 0.1)
            logger.info(f"🧠🟢 [DOPAMINA] {strategy_id} ha ganado. Trust: {self.trust_scores[strategy_id]:.2f}")
        else:
            # 🔴 CORTISOL EXTREMO
            if recent_streak == -2:
                self.trust_scores[strategy_id] = 0.2
                # Muta el ADN: TP un 50% más rápido (asegurar ganancias mínimas), SL un 50% más estricto (cortar pérdidas de inmediato)
                self.genetic_matrix[strategy_id] = [0.5, 0.5]
                self.last_punishment_time[strategy_id] = time.time()
                logger.warning(f"🧠🔴 [MUTACIÓN] {strategy_id} perdió 2 seguidas. ADN mutado: TPx0.5, SLx0.5 para extrema supervivencia.")
            else:
                self.trust_scores[strategy_id] = max(0.1, self.trust_scores[strategy_id] - 0.4)
                # Leve mutación defensiva
                self.genetic_matrix[strategy_id][0] = 0.8
                self.genetic_matrix[strategy_id][1] = 0.8
                logger.info(f"🧠🔴 [CORTISOL] {strategy_id} ha perdido. Trust: {self.trust_scores[strategy_id]:.2f}")

    def get_genetic_modifiers(self, strategy_id: str) -> Tuple[float, float, float]:
        """Devuelve (trust_multiplier, tp_modifier, sl_modifier)."""
        if not strategy_id: return 1.0, 1.0, 1.0
        
        if strategy_id in self.last_punishment_time:
            time_since_punishment = time.time() - self.last_punishment_time[strategy_id]
            if time_since_punishment > 900: # 15 minutos
                if self.trust_scores[strategy_id] < 0.8:
                    self.trust_scores[strategy_id] = 0.8
                    # Restaurar ADN
                    self.genetic_matrix[strategy_id] = [1.0, 1.0]
                    logger.info(f"🧠♻️ [REHABILITACIÓN] {strategy_id} recupera ADN y Trust tras 15 min.")
                del self.last_punishment_time[strategy_id]
                
        trust = self.trust_scores[strategy_id]
        tp_mod, sl_mod = self.genetic_matrix[strategy_id]
        return trust, tp_mod, sl_mod
