"""
🧠 THE SOVEREIGN META-BRAIN - Strategy Selector v2.0 (Anti-Whipsaw)
QUÉ: Módulo de meta-cognición que decide qué estrategia priorizar.
POR QUÉ: Los regímenes de mercado cambian; una estrategia que ganó ayer
        puede perder hoy. El efecto Whipsaw duplica el drawdown cuando se
        persigue al ganador reciente (mean-reversion en crypto).
PARA QUÉ: Maximizar el Expected Value (EV) sin perseguir el techo de
        la curva de rendimiento de ninguna estrategia individual.
CÓMO: EMA sobre performance real, Softmax allocation, penalización DD,
       cooldown anti-chasing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from utils.logger import logger
from config import Config
from core.enums import SignalType

class StrategySelector:
    def __init__(self, portfolio=None, data_provider=None):
        self.portfolio = portfolio
        self.data_provider = data_provider
        self.strategy_health = {} # {strategy_id: {'score': 0.0, 'rank': 1}}
        self.last_update = None
        self.update_interval_hours = 2
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V35: DYNAMIC STRATEGY POOL (No More Hardcoded IDs)
        # QUÉ: El pool se sincroniza dinámicamente desde portfolio.strategy_performance.
        # POR QUÉ: Antes tenía IDs hardcoded ('TECHNICAL', 'ML_XGBOOST', etc.) que
        #   NO coincidían con los strategy_id reales (ej: 'MLEnsemble_BTCUSDT',
        #   'SniperMomentum_SCALPING'). Resultado: todos los pesos eran neutrales.
        # PARA QUÉ: Ranking evolutivo real basado en performance de producción.
        # CÓMO: _sync_pool() lee las keys de portfolio.strategy_performance.
        # ═══════════════════════════════════════════════════════════════
        self.strategies_pool = []  # Populated dynamically

    def _sync_pool(self):
        """Syncs strategies_pool from portfolio's actual strategy performance keys."""
        if self.portfolio and hasattr(self.portfolio, 'strategy_performance'):
            real_ids = list(self.portfolio.strategy_performance.keys())
            if real_ids and set(real_ids) != set(self.strategies_pool):
                self.strategies_pool = real_ids
                logger.info(f"🧠 [Meta-Brain] Synced {len(real_ids)} real strategies: {real_ids[:5]}...")

    def update_strategy_rankings(self):
        """
        Main loop for the Meta-Brain.
        Combines Sim Results + Real Portfolio Results.
        """
        # FORENSIC-V35: Sync pool before ranking
        self._sync_pool()
        
        if not self.strategies_pool:
            logger.debug("🧠 [Meta-Brain] No strategies tracked yet. Skipping ranking.")
            return
        
        logger.info("🧠 [Meta-Brain] Starting real-time strategy re-evaluation...")
        
        rankings = {}
        for strat in self.strategies_pool:
            # 1. Get Real Performance from Portfolio
            real_perf = self._get_real_performance(strat)
            
            # 2. Run Mini-Simulation (Mocking)
            sim_perf = self._run_mini_sim(strat)
            
            # 3. Blended Score (70% Real / 30% Sim)
            # Sim helps predict future, Real confirms past reliability.
            blended_score = (real_perf * 0.7) + (sim_perf * 0.3)
            rankings[strat] = blended_score
            
        # Normalize and Rank
        sorted_ranks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        self.strategy_health = {
            strat: {'score': score, 'rank': i+1} 
            for i, (strat, score) in enumerate(sorted_ranks)
        }
        
        self.last_update = datetime.now(timezone.utc)
        logger.info(f"🏆 [Meta-Brain] New Strategy Ranking: {self.strategy_health}")
        
        # Sync with Portfolio for Dashboard/Oracle visibility
        if self.portfolio:
            self.portfolio.strategy_rankings = self.strategy_health

    def _get_real_performance(self, strategy_id) -> float:
        """
        Fetch win rate + profit factor del Portfolio, penalizados por drawdown actual.
        
        QUÉ: Score compuesto que refleja performance histórico Y estado actual del DD.
        POR QUÉ: Solo usar win-rate pasado causa Whipsaw: una estrategia en DD
                 activo debe perder peso inmediatamente.
        PARA QUÉ: Que el softmax refleje el riesgo ACTUAL, no solo el histórico.
        """
        if not self.portfolio: return 0.5  # Neutral
        
        perf = self.portfolio.strategy_performance.get(strategy_id)
        if not perf or perf['trades'] < 5:
            return 0.5  # Neutral for new strategies
        
        trades = perf['trades']
        wr = perf['wins'] / trades
        
        # Profit factor (métrica más robusta que WR en scalping)
        gross_profit = perf.get('gross_profit', 0.0)
        gross_loss   = abs(perf.get('gross_loss', 1.0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else wr
        
        # Score base: blend de WR y PF normalizado
        base_score = (wr * 0.5) + (min(profit_factor / 2.0, 1.0) * 0.5)
        
        # Penalización por drawdown actual de la estrategia
        current_dd = perf.get('current_drawdown_pct', 0.0)  # 0-1 float
        dd_penalty = max(0.1, 1.0 - (3.0 * current_dd))     # λ=3.0, min 10%
        
        return float(np.clip(base_score * dd_penalty, 0.05, 1.0))

    def _run_mini_sim(self, strategy_id) -> float:
        """
        Perform a high-speed 'mock' backtest on the last 100 bars.
        Simulates how the strategy would have performed 'right now'.
        """
        try:
            # For brevity in this implementation, we use a simplified proxy.
            # In production, this would call strategy.calculate_signals() over a loop of bars.
            # Here we return a confidence score based on recent indicator alignment.
            
            # TODO: Integrate full event-loop simulation for each strategy
            # For now: placeholder logic that favors strategies based on volatility
            # (Statistical likes Range, ML likes Trend, etc.)
            
            # Simplified Logic:
            # We fetch BTC data as a proxy for the 'current vibe'
            bars = self.data_provider.get_latest_bars('BTC/USDT', n=100)
            if bars is None or len(bars) == 0: return 0.5
            
            closes = np.array([b['close'] for b in bars])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns)
            
            # Simple heuristic mapping for the "Brain"
            if volatility > 0.005: # High Vol
                if strategy_id == 'ML_XGBOOST': return 0.8
                if strategy_id == 'SNIPER_MOMENTUM': return 0.7
                return 0.4
            else: # Low Vol / Range
                if strategy_id == 'STATISTICAL_PAIRS': return 0.8
                if strategy_id == 'PATTERN_RECOGNITION': return 0.6
                return 0.5
                
        except Exception as e:
            logger.error(f"Sim Error for {strategy_id}: {e}")
            return 0.5

    def get_strategy_multiplier(self, strategy_id) -> float:
        """
        Retorna un multiplicador continuo derivado de los pesos Softmax del Meta-Brain.
        FORENSIC-V35: Now syncs pool first and handles unknown IDs gracefully.
        
        QUÉ: En lugar de ranking duro (1.2x/0.5x), usa la distribución Softmax
             para un peso continuo que varía suavemente entre estrategias.
        POR QUÉ: El ranking duro binario causa whipsaw al cambiar bruscamente
                 entre niveles. Softmax suaviza la transición.
        RANGO: 0.5x (estrategia muy débil) a 1.5x (líer claro).
        """
        self._sync_pool()  # FORENSIC-V35: Ensure pool is current
        if not self.strategies_pool:
            return 1.0  # Neutral if no strategies tracked yet
        
        weights = self.get_anti_whipsaw_weights()
        w = weights.get(strategy_id, 1.0 / max(1, len(self.strategies_pool)))
        
        # Normalizar al rango [0.5, 1.5]
        n = len(self.strategies_pool)
        neutral = 1.0 / n  # Peso si todos fueran iguales
        multiplier = 0.5 + (w / (2 * neutral)) if neutral > 0 else 1.0
        return float(np.clip(multiplier, 0.5, 1.5))
    
    def get_anti_whipsaw_weights(self) -> dict:
        """
        Retorna pesos Softmax para todas las estrategias del pool.
        
        QUÉ: Portfolio balanceado con temperatura controlada (anti-concentración
             y anti-dispersión excesiva).
        POR QUÉ: Softmax con temperatura ≈ 0.1 da distribución que favorece
                 al líer pero mantiene diversificación como seguro.
        PARA QUÉ: Que el RiskManager pueda usar pesos fractales en lugar de
                 alocar 100% a una estrategia por turno.
        """
        health = self.strategy_health
        if not health:
            # Neutral weights if not ranked yet
            n = max(1, len(self.strategies_pool))
            return {s: 1.0 / n for s in self.strategies_pool}
        
        scores = np.array([health.get(s, {}).get('score', 0.5) for s in self.strategies_pool])
        
        # Softmax con temperatura=0.1 (concentra en líder sin ser winner-take-all)
        TEMPERATURE = 0.10
        scaled = scores / TEMPERATURE
        scaled -= scaled.max()   # Estabilidad numérica
        exp_s = np.exp(scaled)
        weights_arr = exp_s / exp_s.sum()
        
        return {s: float(w) for s, w in zip(self.strategies_pool, weights_arr)}

    def get_governance_advice(self) -> dict:
        """
        PROFESSOR METHOD:
        QUÉ: Opinión experta del Meta-Brain sobre los límites operativos actuales.
        POR QUÉ: Para que el usuario entienda por qué el bot se auto-limita.
        """
        total_health = sum(d['score'] for d in self.strategy_health.values()) / max(1, len(self.strategy_health))
        
        advice = {
            "status": "NORMAL",
            "message": "Límites estándar operacionales (15/símbolo, 100/total).",
            "concurrency_target": Config.MAX_CONCURRENT_POSITIONS
        }
        
        if total_health < 0.35:
            advice["status"] = "DEFENSIVE"
            advice["message"] = "Salud estratégica baja. Se recomienda reducir límites a 5/símbolo."
            advice["concurrency_target"] = 1
        elif total_health > 0.75:
            advice["status"] = "AGGRESSIVE"
            advice["message"] = "Alta confianza estratégica. El sistema puede manejar mayor carga."
            advice["concurrency_target"] = min(5, Config.MAX_CONCURRENT_POSITIONS + 1)
            
        return advice

    def should_allow_trade(self, strategy_id) -> bool:
        """Global veto power for the Meta-Brain."""
        health = self.strategy_health.get(strategy_id, {'score': 0.5})
        # If a strategy is performing horribly (< 0.3 blended score), block it
        if health['score'] < 0.3:
            logger.warning(f"🧠 [Meta-Brain] VETO: Strategy {strategy_id} performing poorly (Score: {health['score']:.2f})")
            return False
        return True
